"""SYSTEM-FIT F1 report: team styles, style dimensions, system labels, and the PLAN-fit checks, written to
cfb/fit/output/f1.json and cfb/fit/output/team_styles.csv (team-season style features, dimensions, label probabilities).

Checks:
  stability     split-half (even vs. odd seasons) refits: |cosine| between each dimension's two versions; adjusted Rand
                index between the two halves' hard system labels for K = 3..10
  persistence   year-to-year correlation of each dimension for the same team, with the same head coach vs. after a
                change; median style distance for those pairs vs. random pairs of teams

    python cfb/fit/f1_report.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from backtest.asof import Loader
from cfb.fit import data as D
from cfb.fit import systems as S
from sparvi.config import RAW, ROOT, cfb

OUT = ROOT / "cfb" / "fit" / "output"


def team_styles(loader, seasons) -> pd.DataFrame:
    rows = []
    for s in seasons:
        as_of = pd.Timestamp(f"{s + 1}-02-01")
        ev, pos = D.season_events(loader, s, as_of)
        st = S.style(loader.load(f"plays_{s}", as_of), ev, pos)
        rows.append(st.assign(season=s).reset_index().rename(columns={"offense": "team"}))
    return pd.concat(rows).set_index(["season", "team"]).sort_index()


def main() -> None:
    seasons = [s for s in cfb().seasons if s < cfb().season]
    st = team_styles(Loader(), seasons)
    even = st[st.index.get_level_values("season") % 2 == 0]
    odd = st[st.index.get_level_values("season") % 2 == 1]
    dims = S.dimensions(st)
    de, do = S.dimensions(even), S.dimensions(odd)
    res = {"team_seasons": int(len(st)), "seasons": [seasons[0], seasons[-1]],
           "dimensions": {"explained": [round(x, 4) for x in dims["explained"]],
                          "loadings": dims["loadings"].round(3).to_dict(),
                          "split_half_cosine": [round(abs(float(de["components"][i] @ do["components"][i])), 3) for i in range(S.N_DIMS)]},
           "systems": {}}
    for k in S.K_GRID:
        fe, fo, full = S.fit(even, k), S.fit(odd, k), S.fit(st, k)
        ari = adjusted_rand_score(S.predict(fe, st).to_numpy().argmax(1), S.predict(fo, st).to_numpy().argmax(1))
        res["systems"][k] = {"split_half_ari": round(float(ari), 3), "bic": round(full["bic"][k]),
                             "sizes": np.bincount(full["probs"].to_numpy().argmax(1), minlength=k).tolist()}
    labels = S.fit(st, S.LABEL_K)
    res["labels"] = {"k": S.LABEL_K, "means": labels["means"].round(2).to_dict(orient="index")}
    # Persistence by head-coach change.
    co = pd.read_parquet(RAW / "cfb" / "coaches.parquet")
    coach = co.sort_values("games").groupby(["season", "school"])["coach_id"].last().rename("coach").reset_index().rename(columns={"school": "team"})
    X = dims["coords"].reset_index().merge(coach, on=["season", "team"])
    pair = X.merge(X.assign(season=X["season"] - 1), on=["season", "team"], suffixes=("", "_next"))
    pair["same_coach"] = pair["coach"] == pair["coach_next"]
    cols = list(dims["coords"].columns)
    pair["dist"] = np.linalg.norm(pair[cols].to_numpy() - pair[[c + "_next" for c in cols]].to_numpy(), axis=1)
    x = dims["coords"].to_numpy()
    idx = np.random.default_rng(0).integers(0, len(x), (5000, 2))
    res["persistence"] = {
        "pairs": {"same_coach": int(pair["same_coach"].sum()), "new_coach": int((~pair["same_coach"]).sum())},
        "median_distance": {"same_coach": round(float(pair.loc[pair["same_coach"], "dist"].median()), 3),
                            "new_coach": round(float(pair.loc[~pair["same_coach"], "dist"].median()), 3),
                            "random_teams": round(float(np.median(np.linalg.norm(x[idx[:, 0]] - x[idx[:, 1]], axis=1))), 3)},
        "year_to_year_r": {c: {k: round(float(np.corrcoef(q[c], q[c + "_next"])[0, 1]), 3)
                               for k, q in (("same_coach", pair[pair["same_coach"]]), ("new_coach", pair[~pair["same_coach"]]))} for c in cols}}
    OUT.mkdir(parents=True, exist_ok=True)
    table = st.join(dims["coords"].round(4)).join(labels["probs"].round(4))
    table.round(5).to_csv(OUT / "team_styles.csv")
    (OUT / "f1.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("labels",)}, indent=1, default=str)[:3500])


if __name__ == "__main__":
    main()
