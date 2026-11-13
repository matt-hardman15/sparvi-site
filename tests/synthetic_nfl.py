"""Synthetic nflverse files for testing the football dbt project in CI.

Real play-by-play isn't committed, so CI builds the project against this: the same columns
ingest/nfl/pull.py keeps, plausible values, a bye week (to test that standings carry forward),
two-point tries, kneel-downs, and spikes (which the efficiency rates must leave out), and team stats
computed with nflverse's own definitions (so the reconciliation test means something).

    write_nfl_fixture(out_dir, season) -> nfl/pbp_{season-1,season}.parquet, nfl/team_stats_*.parquet,
                                          nfl/schedules.parquet, nfl/teams.parquet
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from fixture_contract import write_conformed

TEAMS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
WEEKS = 7          # weeks 1-6 full; in week 7 two teams have a bye
PLAYS_PER_SIDE = 60


def _schedule(season: int, rng: np.random.Generator) -> pd.DataFrame:
    pairs = list(combinations(TEAMS, 2))
    rng.shuffle(pairs)
    rows, used = [], set()
    for week in range(1, WEEKS + 1):
        busy: set[str] = set()
        n_games = 4 if week < WEEKS else 3
        for a, b in pairs:
            if len([r for r in rows if r["week"] == week]) == n_games:
                break
            if (a, b) in used or a in busy or b in busy:
                continue
            used.add((a, b)); busy |= {a, b}
            home, away = (a, b) if rng.random() < 0.5 else (b, a)
            rows.append({"game_id": f"{season}_{week:02d}_{away}_{home}", "season": season, "game_type": "REG",
                         "week": week, "gameday": f"{season}-09-{week + 6:02d}", "home_team": home, "away_team": away,
                         "location": "Neutral" if (week == 3 and not any(r["location"] == "Neutral" for r in rows)) else "Home"})
    return pd.DataFrame(rows)


def _plays(sched: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    strength = {t: rng.normal(0, 0.08) for t in TEAMS}
    out = []
    for g in sched.itertuples():
        for off, dff in ((g.home_team, g.away_team), (g.away_team, g.home_team)):
            n = PLAYS_PER_SIDE
            is_pass = rng.random(n) < 0.58
            ptype = np.where(is_pass, "pass", "run").astype(object)
            two_pt = np.zeros(n, int); two_pt[-1] = 1                     # one two-point try per side
            kneel = np.zeros(n, int); spike = np.zeros(n, int)
            kneel[-2] = 1; ptype[-2] = "qb_kneel"; is_pass[-2] = False
            spike[-3] = 1; ptype[-3] = "qb_spike"; is_pass[-3] = True
            epa = rng.normal(strength[off] - strength[dff], 1.3, n)
            qtr = np.repeat([1, 2, 3, 4], n // 4)
            wp = np.clip(rng.normal(0.5, 0.3, n), 0.001, 0.999)
            wp[rng.random(n) < 0.03] = np.nan
            yards = np.round(rng.normal(np.where(is_pass, 6.5, 4.3), 8)).astype(int)
            out.append(pd.DataFrame({
                "game_id": g.game_id, "season": g.season, "week": g.week, "season_type": "REG",
                "game_date": g.gameday, "home_team": g.home_team, "away_team": g.away_team,
                "posteam": off, "defteam": dff, "posteam_type": np.where(off == g.home_team, "home", "away"),
                "qtr": qtr, "down": rng.integers(1, 5, n), "ydstogo": rng.integers(1, 15, n),
                "yardline_100": rng.integers(1, 99, n), "play_type": ptype,
                "pass": (is_pass & (kneel == 0)).astype(int), "rush": ((~is_pass) & (kneel == 0)).astype(int),
                "qb_dropback": is_pass.astype(int), "sack": (is_pass & (rng.random(n) < 0.06)).astype(int),
                "qb_scramble": (is_pass & (rng.random(n) < 0.03)).astype(int), "qb_kneel": kneel, "qb_spike": spike,
                "pass_attempt": is_pass.astype(int), "rush_attempt": ((~is_pass) | (kneel == 1)).astype(int),
                "complete_pass": (is_pass & (rng.random(n) < 0.65)).astype(int),
                "interception": (is_pass & (rng.random(n) < 0.02)).astype(int),
                "fumble_lost": (rng.random(n) < 0.008).astype(int), "two_point_attempt": two_pt,
                "penalty": 0, "yards_gained": yards, "air_yards": np.where(is_pass, rng.normal(8, 9, n), np.nan),
                "epa": epa, "qb_epa": epa, "success": (epa > 0).astype(float), "wp": wp, "vegas_wp": wp,
                "cpoe": np.where(is_pass, rng.normal(0, 40, n), np.nan),
                "xpass": np.where(rng.random(n) < 0.95, rng.uniform(0.2, 0.9, n), np.nan),
                "pass_oe": np.nan, "posteam_score": 0, "defteam_score": 0, "home_score": 0, "away_score": 0, "result": 0,
            }))
    p = pd.concat(out, ignore_index=True)
    p.insert(0, "play_id", p.groupby("game_id").cumcount() + 1)
    return p


def _team_stats(p: pd.DataFrame) -> pd.DataFrame:
    """nflverse's definitions: qb_epa on pass plays and spikes, epa on runs and kneels, 2-pt tries included."""
    pas = p[p["play_type"].isin(["pass", "qb_spike"])].groupby(["season", "posteam"])["qb_epa"].sum()
    run = p[p["play_type"].isin(["run", "qb_kneel"])].groupby(["season", "posteam"])["epa"].sum()
    t = pd.DataFrame({"passing_epa": pas, "rushing_epa": run}).reset_index().rename(columns={"posteam": "team"})
    t["season_type"] = "REG"
    return t


def write_nfl_fixture(out: Path, season: int, seed: int = 0) -> None:
    d = out / "nfl"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    scheds = []
    for s in (season - 1, season):
        sch = _schedule(s, rng)
        p = _plays(sch, rng)
        pts = p.groupby(["game_id", "posteam"])["epa"].sum().clip(lower=0).round().astype(int) * 2 + 10
        sch["home_score"] = [pts[(g, h)] for g, h in zip(sch["game_id"], sch["home_team"], strict=True)]
        sch["away_score"] = [pts[(g, a)] + (1 if pts[(g, a)] == pts[(g, h)] else 0)   # no ties
                             for g, a, h in zip(sch["game_id"], sch["away_team"], sch["home_team"], strict=True)]
        sch["result"] = sch["home_score"] - sch["away_score"]
        sch["overtime"] = 0
        write_conformed(p, d / f"pbp_{s}.parquet", "nfl", "pbp")
        write_conformed(_team_stats(p), d / f"team_stats_{s}.parquet", "nfl", "team_stats")
        scheds.append(sch)
    write_conformed(pd.concat(scheds), d / "schedules.parquet", "nfl", "schedules")
    write_conformed(pd.DataFrame({"team_abbr": TEAMS, "team_name": [f"Team {t}" for t in TEAMS], "team_nick": TEAMS,
                                  "team_conf": "AFC", "team_division": "AFC Test", "team_color": "#000000"}),
                    d / "teams.parquet", "nfl", "teams")
