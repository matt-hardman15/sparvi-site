"""DATA-QA: id-linkage audit across CFB sources (Matt, Nov 16: are ids mapped correctly between sources?). For each link
the projection, profile, and fit models rely on: how many records link, how many are ambiguous or duplicated, and a
precision proxy (a linked record agrees with what its other records say: team, position group, season). Aggregates only
-> cfb/qa/output/ids.json.

    python -m cfb.qa.ids
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cfb.fit.portal import link
from cfb.profile import engine as G
from ingest.cfb.play_players import norm
from sparvi.config import RAW, ROOT

P = RAW / "cfb" / "players"
OUT = ROOT / "cfb" / "qa" / "output"
SEASONS = range(2014, 2027)


def _r(x: float) -> float:
    return round(float(x), 4)


def rosters() -> dict:
    """Roster -> roster: placeholder ids, a name listed twice on one team, one id on two teams, an id whose name changes."""
    out, prev = {}, None
    for s in SEASONS:
        r = pd.read_parquet(P / f"roster_{s}.parquet")
        r = r.assign(pid=r["id"].astype(str), full=(r["firstName"].fillna("") + " " + r["lastName"].fillna("")).map(norm))
        dup = r[r.duplicated(["team", "full"], keep=False) & (r["full"] != "")]
        two_teams = r.groupby("pid")["team"].nunique()
        row = {"rows": int(len(r)), "placeholder_ids": int(r["pid"].str.startswith("-").sum()),
               "name_twice_on_a_team": int(dup.groupby(["team", "full"]).ngroups),
               "id_on_two_teams": int((two_teams > 1).sum())}
        if prev is not None:
            j = prev.drop_duplicates("pid").merge(r.drop_duplicates("pid"), on="pid", suffixes=("_a", "_b"))
            last_a = j["full_a"].str.split().str[-1]
            last_b = j["full_b"].str.split().str[-1]
            row["ids_carried_from_last_season"] = int(len(j))
            row["carried_id_last_name_changed"] = int((last_a != last_b).sum())
        out[s] = row
        prev = r
    return out


def play_text() -> dict:
    """Play text -> roster: matched share by role, and unmatched names that still have two or more same-name roster rows."""
    out = {}
    for s in SEASONS:
        x = pd.read_parquet(P / f"play_players_{s}.parquet")
        r = pd.read_parquet(P / f"roster_{s}.parquet")
        r = r.assign(full=(r["firstName"].fillna("") + " " + r["lastName"].fillna("")).map(norm))
        counts = r.groupby(["team", "full"]).size()
        un = x[x["player_id"].isna() & x["role"].isin(["passer", "rusher", "target"])]
        amb = [counts.get((t, norm(n)), 0) >= 2 for t, n in zip(un["team"], un["name"], strict=True)]
        # Precision proxy: the matched player's roster position fits the role (a passer is mostly a QB; a receiver a WR,
        # TE, or RB). Trick plays exist, so this is a floor on wrong matches, not a count of them.
        rpos = r.assign(pid=r["id"].astype(str)).drop_duplicates("pid").set_index("pid")["position"]
        fits = {"passer": {"QB"}, "rusher": {"QB", "RB", "FB", "WR", "ATH"}, "target": {"WR", "TE", "RB", "FB", "ATH"}}
        prec = {}
        for role, ok in fits.items():
            m = x[(x["role"] == role) & x["player_id"].notna()]
            prec[f"{role}_position_fits"] = _r(m["player_id"].map(rpos).isin(ok).mean()) if len(m) else None
        out[s] = {"rows": int(len(x)), **{f"{role}_matched": _r(x.loc[x["role"] == role, "player_id"].notna().mean())
                                          for role in ("passer", "rusher", "target")}, **prec,
                  "unmatched_still_ambiguous": int(sum(amb)), "unmatched_rows": int(len(un))}
    return out


def portal() -> dict:
    """Portal -> roster: link methods, and precision proxies: the linked player's previous team is the entry's origin, and
    his roster position group matches the portal's."""
    out = {}
    rost = {s: pd.read_parquet(P / f"roster_{s}.parquet") for s in SEASONS}
    for s in range(2021, 2027):
        f = P / f"portal_{s}.parquet"
        if not f.exists():
            continue
        p = pd.read_parquet(f).assign(season=s)
        lk = link(p, {y: rost[y] for y in (s - 2, s - 1)})
        m = lk[lk["player_id"].notna()].copy()
        ros = pd.concat([rost[s - 1].assign(sea=s - 1), rost[s - 2].assign(sea=s - 2)])
        ros = ros.assign(pid=ros["id"].astype(str)).sort_values("sea", ascending=False).drop_duplicates("pid").set_index("pid")
        m["r_team"] = m["player_id"].astype(str).map(ros["team"])
        m["r_grp"] = m["player_id"].astype(str).map(ros["position"]).map(G.POSITION_GROUP)
        m["p_grp"] = m["position"].map(G.POSITION_GROUP)
        both = m[m["r_grp"].notna() & m["p_grp"].notna()]
        out[s] = {"entries": int(len(p)), "linked": _r(lk["player_id"].notna().mean()),
                  "by_method": lk["link_method"].fillna("unlinked").value_counts().to_dict(),
                  "origin_agrees": _r((m["r_team"] == m["origin"]).mean()),
                  "position_group_agrees": _r((both["r_grp"] == both["p_grp"]).mean()) if len(both) else None,
                  "one_id_two_entries": int(m["player_id"].duplicated().sum())}
    return out


def recruits() -> dict:
    """Recruits -> roster: the share of signees with an athlete id that appears on a roster within two seasons, and
    position-group agreement for skill positions."""
    rost = pd.concat([pd.read_parquet(P / f"roster_{s}.parquet").assign(season=s) for s in SEASONS], ignore_index=True)
    rost = rost.assign(pid=rost["id"].astype(str))
    first = rost.sort_values("season").drop_duplicates("pid").set_index("pid")
    pos_map = {"PRO": "QB", "DUAL": "QB", "QB": "QB", "RB": "RB", "APB": "RB", "WR": "WR", "TE": "TE"}
    out = {}
    for s in range(2015, 2027):
        f = P / f"recruits_{s}.parquet"
        if not f.exists():
            continue
        r = pd.read_parquet(f)
        r = r[r["athleteId"].notna()].assign(pid=lambda d: d["athleteId"].astype(int).astype(str))
        on = r["pid"].isin(first.index)
        soon = on & (r["pid"].map(first["season"]) <= s + 1)
        sk = r[r["position"].isin(pos_map) & on]
        grp = sk["pid"].map(first["position"]).map(G.POSITION_GROUP)
        out[s] = {"signees_with_id": int(len(r)), "on_a_roster": _r(on.mean()), "on_a_roster_within_a_season": _r(soon.mean()),
                  "skill_position_group_agrees": _r((grp == sk["position"].map(pos_map)).mean()) if len(sk) else None,
                  "one_id_two_signees": int(r["pid"].duplicated().sum())}
    return out


def coordinators() -> dict:
    p = ROOT / "cfb" / "fit" / "output" / "oc.json"
    v = json.loads(p.read_text(encoding="utf-8"))["validation"] if p.exists() else {}
    return {"team_seasons": v.get("team_seasons"), "with_coordinator": v.get("with_oc"), "head_coach_matches_cfbd": v.get("head_coach_matches_cfbd")}


def main() -> None:
    res = {"rosters": rosters(), "play_text": play_text(), "portal": portal(), "recruits": recruits(), "coordinators": coordinators()}
    OUT.mkdir(parents=True, exist_ok=True)
    Path(OUT / "ids.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in res.items() if k != "rosters"}, indent=1, default=str)[:6000])
    print("rosters", {s: v for s, v in res["rosters"].items() if s in (2014, 2019, 2022, 2025, 2026)})


if __name__ == "__main__":
    main()
