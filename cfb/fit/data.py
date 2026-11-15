"""SYSTEM-FIT data layer (F0): events including FCS-only games, player experience, and role availability.

season_events     one season's play events (cfb/profile/events.py) from every game in the plays table: FBS games and,
                  from 2022, FCS-only games (dated by week, ingest/snapshot.py), read through the point-in-time loader.
experience        seasons of college football per player, derived (CFBD's roster class field is unusable): seasons since
                  his recruiting class when linked to one, otherwise 1 + earlier seasons on a roster.
role_availability for each team and position group, as of a decision date: last season's workload share (dropbacks,
                  carries, targets) of players who have entered the portal, whose eligibility is likely exhausted, or
                  who return. Shown next to fit, never mixed into it (PLAN-fit, decided Nov 14).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfb.fit.portal import link
from cfb.profile import engine as G
from cfb.profile import events as E

GROUP_OPP = {"QB": "dropback", "RB": "carry", "WR": "target", "TE": "target"}


def season_events(loader, s: int, as_of) -> tuple[pd.DataFrame, pd.Series]:
    plays = loader.load(f"plays_{s}", as_of)
    ev = E.plays(plays, loader.load(f"play_players_{s}", as_of))
    ros = loader.load(f"players_roster_{s}", as_of, columns=["id", "position"])
    pos = ros.assign(id=ros["id"].astype(str)).drop_duplicates("id").set_index("id")["position"]
    return ev, pos


def experience(s: int, rosters: dict[int, pd.DataFrame], recruits: pd.DataFrame) -> pd.Series:
    """player_id -> seasons of college football including season s (1-6)."""
    ids = rosters[s]["id"].astype(str).drop_duplicates()
    seen = pd.concat([rosters[y]["id"].astype(str).drop_duplicates() for y in rosters if y < s] or [pd.Series(dtype=str)])
    count = seen.value_counts()
    rec = (recruits[recruits["athleteId"].notna()].assign(pid=lambda d: d["athleteId"].astype(str))
           .sort_values("year").drop_duplicates("pid", keep="first").set_index("pid")["year"])
    by_rec = s - ids.map(rec) + 1
    by_ros = 1 + ids.map(count).fillna(0)
    exp = by_rec.where(by_rec.between(1, 6), by_ros).clip(1, 6)
    return pd.Series(exp.to_numpy(), index=ids.to_numpy())


def workload(ev: pd.DataFrame, pos: pd.Series) -> pd.DataFrame:
    """Per team, position group, player: his share of the team's opportunities of the group's type last season."""
    out = []
    grp = ev["player_id"].map(pos).map(G.POSITION_GROUP)
    for g, opp in GROUP_OPP.items():
        r = ev[(ev["opp"] == opp) & (grp == g)]
        if r.empty:
            continue
        team_total = ev[ev["opp"] == opp].groupby("offense")["play_id"].nunique()
        n = r.groupby(["offense", "player_id"]).size().rename("n").reset_index()
        n["share"] = n["n"] / n["offense"].map(team_total)
        out.append(n.assign(group=g).rename(columns={"offense": "team"}))
    return pd.concat(out, ignore_index=True)


def leave_rates(history: list[tuple[pd.DataFrame, set, pd.Series, set]]) -> dict[int, float]:
    """P(a player not in the portal is gone the next season | seasons of experience), from earlier seasons.
    history: [(workload, portal ids, experience, (team, player) pairs who played the next season)]."""
    rows = []
    for w, gone_portal, exp, played_next in history:
        x = w[~w["player_id"].isin(gone_portal) & (w["share"] >= 0.05)]
        rows.append(pd.DataFrame({"exp": x["player_id"].map(exp).fillna(1).clip(1, 6).astype(int).to_numpy(),
                                  "gone": [(t, p) not in played_next for t, p in zip(x["team"], x["player_id"], strict=True)]}))
    d = pd.concat(rows)
    return {int(e): float(g["gone"].mean()) for e, g in d.groupby("exp")}


def role_availability(prev_ev: pd.DataFrame, prev_pos: pd.Series, portal: pd.DataFrame, exp_prev: pd.Series,
                      as_of: pd.Timestamp, p_leave: dict[int, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per team and group: last season's workload share that is leaving via the portal (entries filed by as_of,
    certain), expected to leave otherwise (share x P(leave | experience), estimated on earlier seasons: exhausted
    eligibility, the NFL, other exits), and expected to return. portal: linked entries for the coming season."""
    w = workload(prev_ev, prev_pos)
    filed = portal[pd.to_datetime(portal["transferDate"], utc=True).dt.tz_localize(None) <= pd.Timestamp(as_of)]
    w["portal"] = w["player_id"].isin(set(filed["player_id"].dropna()))
    w["p_leave"] = np.where(w["portal"], 1.0, w["player_id"].map(exp_prev).fillna(1).clip(1, 6).astype(int).map(p_leave).fillna(0.2))
    agg = w.groupby(["team", "group"]).apply(lambda d: pd.Series({
        "lost_portal": float(d.loc[d["portal"], "share"].sum()),
        "expected_other_exits": float((d.loc[~d["portal"], "share"] * d.loc[~d["portal"], "p_leave"]).sum()),
        "players": int(len(d))}), include_groups=False).reset_index()
    agg["open_share"] = (agg["lost_portal"] + agg["expected_other_exits"]).clip(0, 1)
    agg["returning"] = 1 - agg["open_share"]
    return agg, w


__all__ = ["season_events", "experience", "workload", "leave_rates", "role_availability", "link"]
