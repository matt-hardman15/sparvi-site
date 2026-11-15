"""Transfer history and success labels (SYSTEM-FIT F3), with the definitions Matt approved on Nov 14 (PLAN-fit).

A transfer into season t: a player whose roster team in season t differs from his roster team in season t-1. Rosters,
not the portal list, so it covers 2018-2020 (before CFBD's portal records) and players who never saw the field at the
new school (they didn't win the role).

Success in season t at the new school = won the role AND produced above average:
  won the role   QB: at least 50% of the team's dropbacks. RB: the team's largest carry share or at least 35%.
                 WR: among the team's top three in targets with at least 12% of them. TE: top two, at least 8%.
  above average  his efficiency composite (mean of the within-position z-scores of opponent-adjusted EPA and success
                 rate per opportunity) above the average composite of players who won the role at his position, at
                 the destination's competition level (Power, Group of Five, FCS), that season.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfb.profile import engine as G

POWER = {"ACC", "Big Ten", "Big 12", "SEC"}
POWER_THROUGH_2023 = {"Pac-12"}
INDEPENDENT_POWER = {"Notre Dame"}
EFFICIENCY = {"QB": ("qb_epa", "qb_success"), "RB": ("rb_epa", "rb_success"), "WR": ("rec_epa", "rec_success"),
              "TE": ("rec_epa", "rec_success")}
SHARE = {"QB": "qb_dropback_share", "RB": "rb_carry_share", "WR": "rec_target_share", "TE": "rec_target_share"}


def tier(team: str, season: int, conf: dict[str, str]) -> str:
    c = conf.get(team)
    if c is None:
        return "FCS"
    if c in POWER or (c in POWER_THROUGH_2023 and season <= 2023) or team in INDEPENDENT_POWER:
        return "Power"
    return "G5"


def role_and_output(profile: pd.DataFrame, season: int, conf: dict[str, str]) -> pd.DataFrame:
    """One row per player and position group in a season: team, share, won_role, composite, tier, above_average."""
    p = profile[profile["season"] == season]
    rows = []
    for g in ("QB", "RB", "WR", "TE"):
        q = p[p["group"] == g]
        share = q[q["feature"] == SHARE[g]].set_index("player_id")[["team", "raw"]].rename(columns={"raw": "share"})
        if share.empty:
            continue
        rank = share.groupby("team")["share"].rank(ascending=False, method="min")
        if g == "QB":
            won = share["share"] >= 0.5
        elif g == "RB":
            won = (rank == 1) | (share["share"] >= 0.35)
        elif g == "WR":
            won = (rank <= 3) & (share["share"] >= 0.12)
        else:
            won = (rank <= 2) & (share["share"] >= 0.08)
        eff = q[q["feature"].isin(EFFICIENCY[g])].pivot_table(index="player_id", columns="feature", values="adj")
        z = (eff - eff.mean()) / eff.std()
        d = share.assign(won_role=won, composite=z.mean(axis=1).reindex(share.index), group=g, season=season)
        rows.append(d.reset_index())
    out = pd.concat(rows, ignore_index=True)
    out["tier"] = [tier(t, season, conf) for t in out["team"]]
    starters = out[out["won_role"] & out["composite"].notna()]
    bar = starters.groupby(["group", "tier"])["composite"].mean()
    out["bar"] = [bar.get((g, t), np.nan) for g, t in zip(out["group"], out["tier"], strict=True)]
    out["above_average"] = out["composite"] > out["bar"]
    return out


def transfers(rosters: dict[int, pd.DataFrame], t: int, portal: pd.DataFrame | None = None) -> pd.DataFrame:
    """Transfers into season t (QB, RB, WR, TE by the season-t roster position), from three sources, first match wins:
      id      the same athlete id on a different roster team than in t-1
      reid    a new id in season t whose normalized full name matches exactly one t-1 roster player of the same position
              group on a different team (CFBD sometimes issues a new id on transfer, e.g. Bo Nix in 2022)
      portal  a portal entry for season t with a destination, matched by name to the destination's season-t roster
              (catches players missing from the origin roster, e.g. Quinn Ewers in 2022); origin from the entry
    prev_id is the player's id in season t-1 where known (to link his prior profile); player_id is his season-t id."""
    from ingest.cfb.play_players import norm
    prev = rosters[t - 1].assign(player_id=rosters[t - 1]["id"].astype(str)).drop_duplicates("player_id")
    cur = rosters[t].assign(player_id=rosters[t]["id"].astype(str)).drop_duplicates("player_id")
    for d in (prev, cur):
        d["full"] = (d["firstName"].fillna("") + " " + d["lastName"].fillna("")).map(norm)
        d["group"] = d["position"].map(G.POSITION_GROUP)
    cur = cur[cur["group"].notna()]
    m = prev[["player_id", "team"]].merge(cur[["player_id", "team", "position", "group", "firstName", "lastName"]], on="player_id",
                                          suffixes=("_from", "_to"))
    m = m[m["team_from"] != m["team_to"]].assign(prev_id=lambda d: d["player_id"], source="id")
    out = [m]
    new = cur[~cur["player_id"].isin(prev["player_id"])]
    by_name = prev.groupby(["full", "group"])
    cand = new.merge(prev[["full", "group", "player_id", "team"]].rename(columns={"player_id": "prev_id", "team": "team_from"}),
                     on=["full", "group"])
    cand = cand[cand["team_from"] != cand["team"]]
    cand = cand[cand.groupby("player_id")["prev_id"].transform("size") == 1]
    cand = cand[cand.apply(lambda r: len(by_name.get_group((r["full"], r["group"]))) == 1, axis=1)] if len(cand) else cand
    out.append(cand.rename(columns={"team": "team_to"})[["player_id", "prev_id", "team_from", "team_to", "position", "group", "firstName", "lastName"]]
               .assign(source="reid"))
    if portal is not None and len(portal):
        pe = portal[(portal["season"] == t) & portal["destination"].notna()].copy()
        pe["full"] = (pe["firstName"].fillna("") + " " + pe["lastName"].fillna("")).map(norm)
        hit = pe.merge(cur[["player_id", "team", "full", "position", "group", "firstName", "lastName"]],
                       left_on=["destination", "full"], right_on=["team", "full"], suffixes=("_p", ""))
        hit = hit[hit.groupby(["destination", "full"])["player_id"].transform("size") == 1]
        hit = hit.assign(team_from=hit["origin"], team_to=hit["destination"], source="portal")
        uniq = prev[~prev.duplicated(["team", "full"], keep=False)].set_index(["team", "full"])["player_id"]
        hit["prev_id"] = [uniq.get((o, f)) for o, f in zip(hit["origin"], hit["full"], strict=True)]
        out.append(hit[["player_id", "prev_id", "team_from", "team_to", "position", "group", "firstName", "lastName", "source"]])
    allm = pd.concat(out, ignore_index=True).drop_duplicates("player_id", keep="first")
    return allm.assign(season=t)


def label(transfers_t: pd.DataFrame, outcomes_t: pd.DataFrame) -> pd.DataFrame:
    """success = won the role at the new school AND above average there. A transfer with no season-t record at the new
    school (never recorded a dropback, carry, or target there) didn't win the role."""
    o = outcomes_t.rename(columns={"team": "team_played"})
    d = transfers_t.merge(o[["player_id", "group", "team_played", "share", "won_role", "composite", "tier", "bar", "above_average"]],
                          on=["player_id", "group"], how="left")
    at_new = d["team_played"] == d["team_to"]
    d["won_role"] = at_new & d["won_role"].fillna(False).astype(bool)
    d["above_average"] = at_new & d["above_average"].fillna(False).astype(bool)
    d["success"] = d["won_role"] & d["above_average"]
    d["played"] = at_new
    return d
