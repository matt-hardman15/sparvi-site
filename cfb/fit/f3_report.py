"""SYSTEM-FIT F3 report: the transfer history 2018-2025 and its success labels (cfb/fit/labels.py), summarized in
cfb/fit/output/f3.json. The player-level label table is written to data/raw/cfb/fit_labels.parquet (private: it's built
from CFBD rosters; never committed or published).

    python cfb/fit/f3_report.py
"""
from __future__ import annotations

import glob
import json

import pandas as pd

from backtest.asof import Loader
from cfb.fit import data as D
from cfb.fit import labels as LB
from cfb.profile import engine as G
from sparvi.config import RAW, ROOT

OUT = ROOT / "cfb" / "fit" / "output"
FIRST, LAST = 2018, 2025
# Well-known transfers, checked by hand against what happened (season = the season at the new school).
SPOT = [("Joe", "Burrow", 2018), ("Jalen", "Hurts", 2019), ("Joe", "Milton III", 2021), ("Hendon", "Hooker", 2021),
        ("Jameson", "Williams", 2021), ("Caleb", "Williams", 2022), ("Bo", "Nix", 2022), ("Quinn", "Ewers", 2022),
        ("Jayden", "Daniels", 2022), ("Michael", "Penix Jr.", 2022), ("Jaxson", "Dart", 2022), ("Ricky", "Pearsall", 2022),
        ("Kedon", "Slovis", 2022), ("Cam", "Ward", 2022), ("Tez", "Johnson", 2023), ("Kedon", "Slovis", 2023),
        ("Cam", "Ward", 2024), ("Dillon", "Gabriel", 2024)]


def build() -> pd.DataFrame:
    L = Loader()
    P = RAW / "cfb" / "players"
    rosters = {s: pd.read_parquet(P / f"roster_{s}.parquet") for s in range(FIRST - 1, LAST + 1)}
    portal = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(P / "portal_*.parquet")))], ignore_index=True)
    out = []
    for t in range(FIRST, LAST + 1):
        ev, pos = D.season_events(L, t, pd.Timestamp(f"{t + 1}-02-01"))
        conf = dict(pd.read_parquet(RAW / "cfb" / f"teams_{t}.parquet")[["school", "conference"]].values)
        out.append(LB.label(LB.transfers(rosters, t, portal), LB.role_and_output(G.season_profile(ev, pos, t), t, conf)))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    d = build()
    d.to_parquet(RAW / "cfb" / "fit_labels.parquet", index=False)
    rate = lambda x: round(float(x.mean()), 4)  # noqa: E731
    res = {"seasons": [FIRST, LAST], "transfers": int(len(d)), "with_prior_id": rate(d["prev_id"].notna()),
           "by_source": d["source"].value_counts().to_dict(),
           "rates": {"played_at_new_school": rate(d["played"]), "won_role": rate(d["won_role"]), "success": rate(d["success"])},
           "by_season": {int(s): {"n": int(len(g)), "success": rate(g["success"])} for s, g in d.groupby("season")},
           "by_group": {g: {"n": int(len(x)), "won_role": rate(x["won_role"]), "success": rate(x["success"])} for g, x in d.groupby("group")},
           "by_destination_tier": {t: {"n": int(len(x)), "success": rate(x["success"])} for t, x in d[d["played"]].groupby("tier")},
           "spot_checks": []}
    for f, last, t in SPOT:
        x = d[(d["firstName"] == f) & (d["lastName"] == last) & (d["season"] == t)]
        for r in x.itertuples():
            res["spot_checks"].append({"player": f"{f} {last}", "season": t, "from": r.team_from, "to": r.team_to, "source": r.source,
                                       "won_role": bool(r.won_role), "composite": None if pd.isna(r.composite) else round(float(r.composite), 3),
                                       "bar": None if pd.isna(r.bar) else round(float(r.bar), 3), "success": bool(r.success)})
        if x.empty:
            res["spot_checks"].append({"player": f"{f} {last}", "season": t, "found": False})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "f3.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in res.items() if k != "spot_checks"}, indent=1))
    for c in res["spot_checks"]:
        print(" ", c)


if __name__ == "__main__":
    main()
