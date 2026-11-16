"""DATA-QA: season-drift report. League-level values of the inputs the CFB models use, per completed season, each
flagged when it sits far from the other seasons (|z| > 3 against the leave-one-out mean and spread). A source or parser
change shows up here before it shows up in a model (receivers' yards per target went 8.1 -> 11.2 -> 5.8 from 2017 to
2025 before CFB-RECV-FIX). Aggregates only -> cfb/qa/output/drift.json.

    python -m cfb.qa.drift
"""
from __future__ import annotations

import json

import pandas as pd

from backtest.asof import Loader
from cfb.proj import pit
from sparvi.config import ROOT

OUT = ROOT / "cfb" / "qa" / "output"
SEASONS = range(2014, 2026)            # completed seasons
Z_FLAG = 3.0


def season_values(L, s: int) -> dict:
    d = pit.season(L, s)
    t, p = d["team"], d["p"]
    ok = t[t["games"] >= 5]

    def rate(g):
        x = p[p["group"] == g]
        return float(x["yards"].sum() / max(x["n"].sum(), 1))
    return {
        "teams": int(len(t)), "games_per_team": float(ok["games"].mean()),
        "pass_plays_per_game": float(ok["pass_pg"].mean()), "run_plays_per_game": float(ok["run_pg"].mean()),
        "matched_dropbacks_per_pass_play": float((ok["dropbacks"] / ok["pass"].clip(lower=1)).mean()),
        "receiving_yards_per_pass_play": float(ok["rec_ypd"].mean()), "rb_share_of_runs": float(ok["rb_frac"].mean()),
        "qb_yards_per_dropback": rate("QB"), "rb_yards_per_carry": rate("RB"), "yards_per_reception": rate("REC"),
        "qbs_per_team_with_50_dropbacks": float((p[(p["group"] == "QB") & (p["n"] >= 50)].groupby("team").size()).mean()),
        "receivers_per_team_with_20_catches": float((p[(p["group"] == "REC") & (p["n"] >= 20)].groupby("team").size()).mean()),
    }


def main() -> None:
    L = Loader()
    rows = {s: season_values(L, s) for s in SEASONS}
    df = pd.DataFrame(rows).T
    flags = []
    for col in df.columns:
        for s in df.index:
            rest = df.loc[df.index != s, col]
            sd = rest.std()
            z = (df.loc[s, col] - rest.mean()) / sd if sd > 0 else 0.0
            if abs(z) > Z_FLAG and s != 2020:              # 2020 (short COVID schedules) is known to be different
                flags.append({"season": int(s), "value": col, "z": round(float(z), 2), "this": round(float(df.loc[s, col]), 3),
                              "others_mean": round(float(rest.mean()), 3)})
    res = {"z_flag": Z_FLAG, "note": "2020 is excluded from flags (short COVID schedules)",
           "seasons": {int(s): {k: round(float(v), 4) for k, v in r.items()} for s, r in df.iterrows()}, "flags": flags}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "drift.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(df.round(3).to_string())
    print("flags:", json.dumps(flags, indent=1) if flags else "none")


if __name__ == "__main__":
    main()
