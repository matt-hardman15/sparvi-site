"""SYSTEM-FIT F0 checks (PLAN-fit "done when"), written to cfb/fit/output/f0.json:

  fcs           share of each season's events from FCS-only games (now included)
  portal_link   share of all portal entries linked to an athlete id, and the share of movers (player on a different team
                the next season, 25+ plays) whose portal entry is linked, split by FBS and FCS origin
  availability  P(not back in the role | experience) estimated on the two prior seasons, calibration bins, and Brier
                score against a flat rate, for FBS teams' players with 5%+ of a workload

    python cfb/fit/f0_report.py
"""
from __future__ import annotations

import glob
import json

import pandas as pd

from backtest.asof import Loader
from cfb.fit import data as D
from cfb.fit.portal import link
from sparvi.config import RAW, ROOT

OUT = ROOT / "cfb" / "fit" / "output"
P = RAW / "cfb" / "players"


def main() -> None:
    L = Loader()
    port = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(P / "portal_*.parquet")))], ignore_index=True)
    rosters = {s: pd.read_parquet(P / f"roster_{s}.parquet") for s in range(2016, 2027)}
    recruits = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(P / "recruits_*.parquet")))], ignore_index=True)
    links = link(port, rosters)
    res = {"portal_link": {"entries": int(len(links)), "linked": round(float(links["player_id"].notna().mean()), 4),
                           "methods": links["link_method"].fillna("unlinked").value_counts().to_dict(), "movers": {}},
           "fcs": {}, "availability": {}}
    cache = {}

    def season(t):
        if t not in cache:
            ev, pos = D.season_events(L, t - 1, pd.Timestamp(f"{t}-02-01"))
            ev_t, _ = D.season_events(L, t, pd.Timestamp(f"{t}-12-31"))
            cache[t] = (ev, pos, D.experience(t - 1, rosters, recruits), set(zip(ev_t["offense"], ev_t["player_id"], strict=True)), ev_t)
        return cache[t]

    for t in range(2022, 2026):
        ev, pos, exp, played, ev_t = season(t)
        fbs = set(pd.read_parquet(RAW / "cfb" / f"teams_{t - 1}.parquet")["school"])
        games = set(pd.read_parquet(RAW / "cfb" / f"games_{t - 1}.parquet")["id"])
        res["fcs"][t - 1] = round(float((~ev["game_id"].isin(games)).mean()), 4)
        n = ev.groupby("player_id").size()
        active = set(n[n >= 25].index)
        a, b = rosters[t - 1][["id", "team"]].astype({"id": str}), rosters[t][["id", "team"]].astype({"id": str})
        m = a.merge(b, on="id", suffixes=("_p", "_c"))
        m = m[(m["team_p"] != m["team_c"]) & m["id"].isin(active)]
        linked = set(links.loc[links["season"] == t, "player_id"].dropna())
        res["portal_link"]["movers"][t] = {k: {"n": int(len(x)), "linked": round(float(x["id"].isin(linked).mean()), 4)}
                                           for k, x in (("fbs_origin", m[m["team_p"].isin(fbs)]), ("fcs_origin", m[~m["team_p"].isin(fbs)]))}
        if t < 2024:
            continue
        hist = []
        for h in (t - 2, t - 1):
            hev, hpos, hexp, hplayed, _ = season(h)
            hist.append((D.workload(hev, hpos), set(links.loc[links["season"] == h, "player_id"].dropna()), hexp, hplayed))
        pl = D.leave_rates(hist)
        _, w = D.role_availability(ev, pos, links[links["season"] == t], exp, pd.Timestamp(f"{t}-02-01"), pl)
        w = w[w["team"].isin(fbs) & (w["share"] >= 0.05)].copy()
        w["gone"] = [(tm, p) not in played for tm, p in zip(w["team"], w["player_id"], strict=True)]
        bins = pd.cut(w["p_leave"], [0, .1, .3, .5, .7, .9, 1.0], include_lowest=True)
        cal = w.groupby(bins, observed=True).agg(n=("gone", "size"), predicted=("p_leave", "mean"), actual=("gone", "mean"))
        res["availability"][t] = {
            "leave_rate_by_experience": {k: round(v, 3) for k, v in pl.items()},
            "brier": round(float(((w["p_leave"] - w["gone"]) ** 2).mean()), 4),
            "brier_flat": round(float(((w["gone"].mean() - w["gone"]) ** 2).mean()), 4),
            "calibration": [{"bin": str(i), "n": int(r.n), "predicted": round(r.predicted, 3), "actual": round(r.actual, 3)} for i, r in cal.iterrows()]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "f0.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(res, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
