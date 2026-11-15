"""SYSTEM-FIT F2 report: player types per fit group (QB, RB, receivers), written to cfb/fit/output/f2.json and
cfb/fit/output/player_types.csv (player-season dimensions and label probabilities).

Checks:
  stability     even vs. odd-season refits: |cosine| per dimension; adjusted Rand index of hard labels for K = 2..8
  persistence   year-to-year correlation of each dimension for the same player, stayed on his team vs. moved
                (does the type travel with the player, or change with the system?)

    python cfb/fit/f2_report.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from backtest.asof import Loader
from cfb.fit import data as D
from cfb.fit import player_types as T
from cfb.profile import engine as G
from sparvi.config import RAW, ROOT, cfb

OUT = ROOT / "cfb" / "fit" / "output"


def main() -> None:
    seasons = [s for s in cfb().seasons if s < cfb().season]
    L = Loader()
    prof = pd.concat([G.season_profile(*D.season_events(L, s, pd.Timestamp(f"{s + 1}-02-01")), s) for s in seasons], ignore_index=True)
    rosters = {s: pd.read_parquet(RAW / "cfb" / "players" / f"roster_{s}.parquet") for s in seasons}
    team = prof.drop_duplicates(["season", "player_id"]).set_index(["season", "player_id"])["team"]
    res, tables = {"seasons": [seasons[0], seasons[-1]], "groups": {}}, []
    for g in T.FIT_GROUPS:
        m = T.group_matrix(prof, rosters, g)
        ev, od = m[m.index.get_level_values(0) % 2 == 0], m[m.index.get_level_values(0) % 2 == 1]
        d, de, do = T.dimensions(m), T.dimensions(ev), T.dimensions(od)
        r = {"player_seasons": int(len(m)), "features": list(m.columns),
             "explained": [round(x, 4) for x in d["explained"]], "loadings": d["loadings"].round(3).to_dict(),
             "split_half_cosine": [round(abs(float(de["components"][i] @ do["components"][i])), 3) for i in range(T.N_DIMS)],
             "label_ari": {k: round(float(adjusted_rand_score(T.predict_labels(T.fit_labels(ev, k), m).to_numpy().argmax(1),
                                                               T.predict_labels(T.fit_labels(od, k), m).to_numpy().argmax(1))), 3) for k in T.K_GRID}}
        X = d["coords"].reset_index()
        X["team"] = [team.get((s, p)) for s, p in zip(X["season"], X["player_id"], strict=True)]
        pr = X.merge(X.assign(season=X["season"] - 1), on=["season", "player_id"], suffixes=("", "_n"))
        pr["mover"] = pr["team"] != pr["team_n"]
        r["persistence"] = {"pairs": {"stayed": int((~pr["mover"]).sum()), "moved": int(pr["mover"].sum())},
                            "year_to_year_r": {c: {k: round(float(np.corrcoef(q[c], q[c + "_n"])[0, 1]), 3)
                                                   for k, q in (("stayed", pr[~pr["mover"]]), ("moved", pr[pr["mover"]]))}
                                               for c in d["coords"].columns}}
        tab = d["coords"].round(4).assign(group=g)
        if T.LABEL_K[g]:
            lab = T.fit_labels(m, T.LABEL_K[g])
            r["labels"] = {"k": T.LABEL_K[g], "means": lab["means"].round(2).to_dict(orient="index")}
            tab = tab.join(lab["probs"].round(4))
        res["groups"][g] = r
        tables.append(tab)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(tables).to_csv(OUT / "player_types.csv")
    (OUT / "f2.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    for g, r in res["groups"].items():
        print(g, r["player_seasons"], "cos", r["split_half_cosine"], "ARI", r["label_ari"], "persistence", r["persistence"]["year_to_year_r"])


if __name__ == "__main__":
    main()
