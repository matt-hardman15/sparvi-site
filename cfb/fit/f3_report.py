"""SYSTEM-FIT F3 report: the transfer history 2018-2025 and its success labels (cfb/fit/labels.py), summarized in
cfb/fit/output/f3.json, plus each transfer's later years at the same school (labels.follow). The player-level tables
are written to data/raw/cfb/fit_labels.parquet and fit_labels_years.parquet (private: built from CFBD rosters; never
committed or published).

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


def build(with_outcomes: bool = False):
    """(year-1 labels, later years), and with_outcomes the per-season role/value tables (FIRST-1 .. LAST) too."""
    L = Loader()
    P = RAW / "cfb" / "players"
    rosters = {s: pd.read_parquet(P / f"roster_{s}.parquet") for s in range(FIRST - 1, LAST + 1)}
    portal = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(P / "portal_*.parquet")))], ignore_index=True)
    outcomes, played = {}, {}
    for t in range(FIRST - 1, LAST + 1):
        ev, pos = D.season_events(L, t, pd.Timestamp(f"{t + 1}-02-01"))
        played[t] = ev.groupby("player_id")["offense"].agg(lambda s: s.mode().iloc[0]).to_dict()
        conf = dict(pd.read_parquet(RAW / "cfb" / f"teams_{t}.parquet")[["school", "conference"]].values)
        outcomes[t] = LB.role_and_output(G.season_profile(ev, pos, t), t, conf, LB.team_plays(ev))
    d = pd.concat([LB.label(LB.transfers(rosters, t, portal, played), outcomes[t]) for t in range(FIRST, LAST + 1)], ignore_index=True)
    y = LB.follow(d, rosters, outcomes, LAST, played)
    return (d, y, outcomes) if with_outcomes else (d, y)


def by_year(y: pd.DataFrame, flag: str, k: int) -> dict:
    """Share of transfers with `flag` in any of their first k years at the school, among cohorts with k observable
    seasons (transfer season + k - 1 <= LAST). A player who left before year k counts as not reaching it."""
    c = y[y["transfer_season"] + k - 1 <= LAST]
    hit = c[c["year"] <= k].groupby(["transfer_season", "player_id", "group"])[flag].any()
    return {"cohorts": [int(c["transfer_season"].min()), LAST - k + 1], "n": int(len(hit)), "rate": round(float(hit.mean()), 4)}


def main() -> None:
    d, y = build()
    d.to_parquet(RAW / "cfb" / "fit_labels.parquet", index=False)
    y.to_parquet(RAW / "cfb" / "fit_labels_years.parquet", index=False)
    y["efficient"] = y["above_average"] & (y["share"] >= 0.10)
    rate = lambda x: round(float(x.mean()), 4)  # noqa: E731
    res = {"seasons": [FIRST, LAST], "transfers": int(len(d)), "with_prior_id": rate(d["prev_id"].notna()),
           "by_source": d["source"].value_counts().to_dict(),
           "rates": {"played_at_new_school": rate(d["played"]), "won_role": rate(d["won_role"]), "success": rate(d["success"])},
           "by_season": {int(s): {"n": int(len(g)), "success": rate(g["success"])} for s, g in d.groupby("season")},
           "by_group": {g: {"n": int(len(x)), "won_role": rate(x["won_role"]), "success": rate(x["success"])} for g, x in d.groupby("group")},
           "by_destination_tier": {t: {"n": int(len(x)), "success": rate(x["success"])} for t, x in d[d["played"]].groupby("tier")},
           "spot_checks": []}
    stay = y.groupby(["transfer_season", "player_id", "group"])["year"].max()
    first = y[y["success"]].groupby(["transfer_season", "player_id", "group"])["year"].min()
    res["later_years"] = {
        "years_at_school": {int(k): int(v) for k, v in stay.value_counts().sort_index().items()},
        "success_by_year": {k: by_year(y, "success", k) for k in (1, 2, 3, 4)},
        "won_role_by_year": {k: by_year(y, "won_role", k) for k in (1, 2, 3, 4)},
        "efficient_by_year": {k: by_year(y, "efficient", k) for k in (1, 2, 3, 4)},
        "first_success_year": {int(k): int(v) for k, v in first.value_counts().sort_index().items()},
        "success_by_year_and_group": {g: {k: by_year(x, "success", k) for k in (1, 2, 3)} for g, x in y.groupby("group")},
        "notes": "efficient = above the starters' average with at least 10% of the team's dropbacks, carries or targets"}
    for f, last, t in SPOT:
        x = d[(d["firstName"] == f) & (d["lastName"] == last) & (d["season"] == t)]
        for r in x.itertuples():
            res["spot_checks"].append({"player": f"{f} {last}", "season": t, "from": r.team_from, "to": r.team_to, "source": r.source,
                                       "won_role": bool(r.won_role), "composite": None if pd.isna(r.composite) else round(float(r.composite), 3),
                                       "bar": None if pd.isna(r.bar) else round(float(r.bar), 3), "success": bool(r.success),
                                       "later": [{"year": int(z.year), "season": int(z.season), "won_role": bool(z.won_role), "success": bool(z.success)}
                                                 for z in y[(y["player_id"] == r.player_id) & (y["transfer_season"] == t) & (y["year"] > 1)].itertuples()]})
        if x.empty:
            res["spot_checks"].append({"player": f"{f} {last}", "season": t, "found": False})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "f3.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in res.items() if k != "spot_checks"}, indent=1))
    for c in res["spot_checks"]:
        print(" ", c)


if __name__ == "__main__":
    main()
