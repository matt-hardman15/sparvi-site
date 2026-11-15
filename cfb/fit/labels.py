"""Transfer history and success labels (SYSTEM-FIT F3), with the definitions Matt approved on Nov 14 (PLAN-fit).

A transfer into season t: a player whose roster team in season t differs from his roster team in season t-1. Rosters,
not the portal list, so it covers 2018-2020 (before CFBD's portal records) and players who never saw the field at the
new school (they didn't win the role).

Success in season t at the new school = won the role AND produced above average:
  won the role   shares are of the team's whole season (his opportunities / all the team's pass plays, run plays, or targets), not
                 only the games he played (the profile's share features use those, which let a backup who
                 played a few games count as the starter).
                 QB: at least 50% of the team's dropbacks. RB: the team's largest carry share or at least 35%.
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


def team_plays(ev: pd.DataFrame) -> pd.Series:
    """Each team's season denominators for the role rule, counted by play id. Index (team, kind):
      pass    pass plays (QB dropbacks; the same count the profile's share features use per game)
      run     run plays (RB carries)
      target  pass plays with a parsed target (WR/TE: the approved rule is a share of the team's targets)"""
    pr = ev[ev["kind"].isin(["pass", "run"])].drop_duplicates("play_id").groupby(["offense", "kind"]).size()
    tg = ev[ev["opp"] == "target"].drop_duplicates("play_id").groupby("offense").size()
    tg.index = pd.MultiIndex.from_tuples([(t, "target") for t in tg.index], names=["offense", "kind"])
    return pd.concat([pr, tg])


def role_and_output(profile: pd.DataFrame, season: int, conf: dict[str, str], plays: pd.Series) -> pd.DataFrame:
    """One row per player and position group in a season: team, share (of the team's season), won_role, composite,
    tier, above_average. plays: team_plays() for the season."""
    p = profile[profile["season"] == season]
    rows = []
    for g in ("QB", "RB", "WR", "TE"):
        q = p[p["group"] == g]
        share = q[q["feature"] == SHARE[g]].set_index("player_id")[["team", "n"]]
        if share.empty:
            continue
        kind = {"QB": "pass", "RB": "run"}.get(g, "target")
        den = np.array([plays.get((t, kind), 0) for t in share["team"]], dtype=float)
        share = share.assign(share=np.minimum(share["n"] / np.maximum(den, 1), 1.0)).drop(columns="n")
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


def _one_team(r: pd.DataFrame, played: dict[str, str] | None) -> pd.DataFrame:
    """One roster row per player id. An id listed on two teams in a season (195 in 2021, stale listings) keeps the team
    he recorded plays for; with no plays to settle it, he is dropped rather than guessed."""
    r = r.assign(player_id=r["id"].astype(str)).drop_duplicates(["player_id", "team"])
    multi = r["player_id"].duplicated(keep=False)
    if not multi.any():
        return r
    played = played or {}
    ok = [played.get(p) == t for p, t in zip(r.loc[multi, "player_id"], r.loc[multi, "team"], strict=True)]
    return pd.concat([r[~multi], r[multi][ok]]).drop_duplicates("player_id", keep=False)


def transfers(rosters: dict[int, pd.DataFrame], t: int, portal: pd.DataFrame | None = None,
              played: dict[int, dict[str, str]] | None = None) -> pd.DataFrame:
    """Transfers into season t (QB, RB, WR, TE by the season-t roster position), from three sources, first match wins:
      id      the same athlete id on a different roster team than in t-1
      reid    a new id in season t whose normalized full name matches exactly one t-1 roster player of the same position
              group on a different team (CFBD sometimes issues a new id on transfer, e.g. Bo Nix in 2022)
      portal  a portal entry for season t with a destination, matched by name to the destination's season-t roster
              (catches players missing from the origin roster, e.g. Quinn Ewers in 2022); origin from the entry.
              Withdrawn entries (destination = his t-1 team, e.g. Jake Haener at Fresno State in 2022) are skipped
    prev_id is the player's id in season t-1 where known (to link his prior profile); player_id is his season-t id.
    played: season -> {player_id: team he recorded plays for}, to settle ids listed on two rosters (see _one_team)."""
    from ingest.cfb.play_players import norm
    played = played or {}
    prev = _one_team(rosters[t - 1], played.get(t - 1))
    cur = _one_team(rosters[t], played.get(t))
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
        # A withdrawn entry lists his own school as the destination: not a transfer if he was already there in t-1.
        was_id = set(zip(prev["player_id"], prev["team"], strict=True))
        was_name = set(zip(prev["full"], prev["team"], strict=True))
        stayed = [(p, d) in was_id or (f, d) in was_name or o == d
                  for p, f, o, d in zip(hit["player_id"], hit["full"], hit["origin"], hit["destination"], strict=True)]
        hit = hit[~np.array(stayed, dtype=bool)]
        uniq = prev[~prev.duplicated(["team", "full"], keep=False)].set_index(["team", "full"])["player_id"]
        hit["prev_id"] = [uniq.get((o, f)) for o, f in zip(hit["origin"], hit["full"], strict=True)]
        out.append(hit[["player_id", "prev_id", "team_from", "team_to", "position", "group", "firstName", "lastName", "source"]])
    allm = pd.concat(out, ignore_index=True).drop_duplicates("player_id", keep="first")
    return allm.assign(season=t)


def follow(labeled: pd.DataFrame, rosters: dict[int, pd.DataFrame], outcomes: dict[int, pd.DataFrame], last: int,
           played: dict[int, dict[str, str]] | None = None) -> pd.DataFrame:
    """Each transfer's seasons at his destination: year 1 is the transfer season (the `label` row), year k is season
    t+k-1, kept while he is on the destination's roster and didn't record plays for another team. The first season he isn't (transferred again, left, or no
    roster) ends his record there; what he does at a later school doesn't count. Seasons after `last` are unobserved.
    One row per transfer and year: transfer_season, year, season, plus label's columns for that season.
    played: season -> {player_id: team he recorded plays for} (as in transfers); defaults to the outcome tables."""
    out = [labeled.assign(transfer_season=labeled["season"], year=1)]
    keep = ["player_id", "prev_id", "group", "position", "team_from", "team_to", "source", "firstName", "lastName"]
    alive = labeled[[*keep, "season"]].rename(columns={"season": "transfer_season"})
    k = 1
    while len(alive):
        k += 1
        alive = alive.assign(season=alive["transfer_season"] + k - 1)
        alive = alive[alive["season"] <= last]
        rows = []
        for s, a in alive.groupby("season"):
            r = rosters.get(s)
            if r is None:
                continue
            on = set(zip(r["id"].astype(str), r["team"], strict=True))
            elsewhere = (played or {}).get(s) or dict(zip(outcomes[s]["player_id"], outcomes[s]["team"], strict=True))
            a = a[[(p, t) in on and elsewhere.get(p, t) == t for p, t in zip(a["player_id"], a["team_to"], strict=True)]]
            if len(a):
                rows.append(label(a.drop(columns="season"), outcomes[s]).assign(season=s, year=k))
        if not rows:
            break
        nxt = pd.concat(rows, ignore_index=True)
        out.append(nxt)
        alive = nxt[[*keep, "transfer_season"]]
    return pd.concat(out, ignore_index=True)


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
