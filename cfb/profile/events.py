"""Source adapters for the player profile (PLAYER-PROFILE). Each adapter turns one data source into the same long
"events" table, so the feature engine (engine.py) never needs to know where a stat came from:

    season, week, game_id, play_id, offense, defense, player_id, opp, <outcome columns...>

`opp` is the opportunity type: dropback (passer), carry (rusher), target (receiver). Outcome columns are numeric
(0/1 for events, yards, EPA); a column that a source or season can't supply is NaN, and features that need it are
skipped for those rows. To add a data source: write an adapter that returns this shape (extra columns are fine) and
register it in ADAPTERS; to add a stat: add a feature to features.yml that names an outcome column.

Adapters here:
  plays      every season: down/distance/field position/yards/play type/EPA from the plays table, players from
             play_players (ingest/cfb/play_players.py)
  play_text  the richer play text CFBD's feed carries from 2025: throw depth and direction, formation, tempo, QB
             hurries. Joined onto `plays` events.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from football.common.filters import cfb_garbage_time

PASS_TYPES = {"Pass Reception", "Pass Incompletion", "Passing Touchdown", "Sack", "Interception", "Pass Interception Return",
              "Pass Interception"}
RUN_TYPES = {"Rush", "Rushing Touchdown"}
COMPLETE = {"Pass Reception", "Passing Touchdown"}
INTERCEPT = {"Interception", "Pass Interception Return", "Pass Interception"}


def _success(down, distance, gained) -> np.ndarray:
    """College success: 50% of the yards to go on 1st down, 70% on 2nd, all of it on 3rd and 4th."""
    need = np.select([down == 1, down == 2], [0.5 * distance, 0.7 * distance], distance)
    return (gained >= need).astype(float)


def plays(plays_df: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Every pass/run play outside garbage time, one row per (play, credited player, opportunity)."""
    p = plays_df[plays_df["playType"].isin(PASS_TYPES | RUN_TYPES)].copy()
    margin = (p["offenseScore"] - p["defenseScore"]).abs()
    p = p[~cfb_garbage_time(p["period"].to_numpy(), margin.to_numpy())]
    p["play_id"] = p["id"].astype(str)
    who = players[players["player_id"].notna() & players["role"].isin(["passer", "target", "rusher"])].drop_duplicates(["play_id", "role"])
    g, d, dist = p["yardsGained"].astype(float), p["down"].astype(float), p["distance"].astype(float)
    base = pd.DataFrame({
        "season": p["season"].to_numpy(), "week": p["week"].to_numpy(), "game_id": p["gameId"].to_numpy(),
        "play_id": p["play_id"].to_numpy(), "offense": p["offense"].to_numpy(), "defense": p["defense"].to_numpy(),
        "play_type": p["playType"].to_numpy(), "down": d.to_numpy(), "distance": dist.to_numpy(),
        "to_goal": p["yardsToGoal"].astype(float).to_numpy(), "yards": g.to_numpy(), "epa": p["ppa"].astype(float).to_numpy(),
        "margin": (p["offenseScore"] - p["defenseScore"]).astype(float).to_numpy(), "period": p["period"].to_numpy(),
    })
    is_pass = base["play_type"].isin(PASS_TYPES).to_numpy()
    base["kind"] = np.where(is_pass, "pass", "run")
    base["success"] = _success(d.to_numpy(), dist.to_numpy(), g.to_numpy())
    base["first_down"] = (g.to_numpy() >= dist.to_numpy()).astype(float)
    base["td"] = p["playType"].isin({"Passing Touchdown", "Rushing Touchdown"}).astype(float).to_numpy()
    base["third_down"] = (d.to_numpy() == 3).astype(float)
    base["early_down"] = (d.to_numpy() <= 2).astype(float)
    base["long_yardage"] = ((d.to_numpy() >= 3) & (dist.to_numpy() >= 7)).astype(float)
    base["short_yardage"] = ((d.to_numpy() >= 3) & (dist.to_numpy() <= 2)).astype(float)
    base["red_zone"] = (base["to_goal"] <= 20).astype(float)
    base["goal_line"] = (base["to_goal"] <= 5).astype(float)
    base["close_game"] = (base["margin"].abs() <= 8).astype(float)

    out = []
    pp = who.pivot(index="play_id", columns="role", values="player_id")
    base = base.merge(pp, left_on="play_id", right_index=True, how="left")
    for col in ("passer", "target", "rusher"):
        if col not in base:
            base[col] = np.nan
    # Dropbacks (passer): attempts, completions, sacks, interceptions, explosive completions.
    db = base[is_pass & base["passer"].notna().to_numpy()].copy()
    db["player_id"], db["opp"] = db["passer"], "dropback"
    db["sack"] = (db["play_type"] == "Sack").astype(float)
    db["attempt"] = 1.0 - db["sack"]
    db["complete"] = db["play_type"].isin(COMPLETE).astype(float)
    db["interception"] = db["play_type"].isin(INTERCEPT).astype(float)
    db["explosive"] = ((db["complete"] == 1) & (db["yards"] >= 20)).astype(float)
    out.append(db)
    # Targets (receiver).
    tg = base[is_pass & base["target"].notna().to_numpy()].copy()
    tg["player_id"], tg["opp"] = tg["target"], "target"
    tg["complete"] = tg["play_type"].isin(COMPLETE).astype(float)
    tg["explosive"] = ((tg["complete"] == 1) & (tg["yards"] >= 20)).astype(float)
    tg["interception"] = tg["play_type"].isin(INTERCEPT).astype(float)
    out.append(tg)
    # Carries (rusher), with whether the rusher is his team's passer that season (QB runs).
    ca = base[~is_pass & base["rusher"].notna().to_numpy()].copy()
    ca["player_id"], ca["opp"] = ca["rusher"], "carry"
    ca["stuffed"] = (ca["yards"] <= 0).astype(float)
    ca["run10"] = (ca["yards"] >= 10).astype(float)
    ca["explosive"] = (ca["yards"] >= 20).astype(float)
    out.append(ca)
    ev = pd.concat(out, ignore_index=True).drop(columns=["passer", "target", "rusher"])
    return ev


# ---- the richer play text (2025 on) ---------------------------------------------------------------------------------
DEPTH = re.compile(r"\bpass (?:complete|incomplete) (short|deep)\s*(left|right|middle)?", re.I)
RUN_DIR = re.compile(r"\brush (left|right|middle)", re.I)


def play_text(plays_raw: list[dict]) -> pd.DataFrame:
    """Per play id: deep (short/deep), dir_middle, shotgun, no_huddle, hurried. Every column NaN when the text doesn't
    carry it. (The "thrown to" spot is logged only on incompletions, so air yards and depth of target aren't derived:
    an average over incompletions alone would be biased deep.)"""
    rows = []
    for p in plays_raw:
        t = p.get("playText") or ""
        if p.get("playType") not in PASS_TYPES | RUN_TYPES or "#" not in t:
            continue
        r = {"play_id": str(p["id"])}
        m = DEPTH.search(t)
        if m:
            r["deep"] = float(m.group(1).lower() == "deep")
            if m.group(2):
                r["dir_middle"] = float(m.group(2).lower() == "middle")
        m = RUN_DIR.search(t)
        if m:
            r["dir_middle"] = float(m.group(1).lower() == "middle")
        r["shotgun"] = float("Shotgun" in t)
        r["no_huddle"] = float("No Huddle" in t)
        r["hurried"] = float("hurried by" in t)
        rows.append(r)
    return pd.DataFrame(rows)


def add_play_text(ev: pd.DataFrame, text: pd.DataFrame) -> pd.DataFrame:
    return ev if text.empty else ev.merge(text, on="play_id", how="left")


ADAPTERS = {"plays": plays, "play_text": play_text}
