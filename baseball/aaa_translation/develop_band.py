"""AAA-BAND: how should the 80% band around the published line (150+ PA/BF in Triple-A) be sized? (development)

The published band is the fitted pairs' own residual quantiles, weighted by the smaller of each pair's two samples.
On the development folds it covered 0.66 to 0.75 of hitters' outcomes on six metrics (PRED-AUDIT section 2). Three
things could make it narrow: the weights favor large-sample pairs, whose residuals are smaller than a typical
pair's; the residuals are in-sample; and the level shifts between seasons.

DEVELOPMENT ONLY, same folds as develop.py: an MLB outcome season t before the current one, everything fit on pairs
whose MLB season is before t. Nothing reads an MLB line from the current season.

Bands, fixed before running. Each is the 10th and 90th percentile of a set of residuals of the published line:
    published      in-sample, weighted by the smaller sample (what the site shows)
    unweighted     in-sample, every pair counted once
    cv             out-of-fold (five folds of players inside the training pairs), every pair counted once
    season         out-of-season (each training season predicted from the others), every pair counted once;
                   needs two training seasons, so it exists only where the fold has them
The line itself is the published one in every case. Coverage is the share of the fold's pairs (150+ in Triple-A,
100+ in MLB) whose MLB rate falls inside; with about 130 to 280 pairs a fold its standard error is 0.025 to 0.035.

    python baseball/aaa_translation/develop_band.py        # about two minutes

Writes output/development_band.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from develop import CFG, METRICS, MIN_AAA, OUT, SEED, _wls, _wquantile, ages, build_lines, pairs, seasons
from sklearn.model_selection import GroupKFold

from backtest.asof import Loader

BANDS = ["published", "unweighted", "cv", "season"]
TARGET = (0.77, 0.83)


def _line(d: pd.DataFrame, a: str, y: str) -> np.ndarray:
    w = np.minimum(d["pa_aaa"], d["pa_mlb"]).to_numpy(float)
    return _wls(np.column_stack([np.ones(len(d)), d[a]]), d[y].to_numpy(float), w)


def _resid(fit: pd.DataFrame, on: pd.DataFrame, a: str, y: str) -> np.ndarray:
    b = _line(fit, a, y)
    return on[y].to_numpy(float) - (b[0] + b[1] * on[a].to_numpy(float))


def bands(tr: pd.DataFrame, a: str, y: str) -> dict:
    """{band: (lo, hi)} offsets from the line fit on `tr` (the published pool's pairs with this metric)."""
    one = np.ones(len(tr))
    res = _resid(tr, tr, a, y)
    w = np.minimum(tr["pa_aaa"], tr["pa_mlb"]).to_numpy(float)
    out = {"published": (_wquantile(res, w, 0.1), _wquantile(res, w, 0.9)),
           "unweighted": (_wquantile(res, one, 0.1), _wquantile(res, one, 0.9))}
    g = tr["player_id"].to_numpy()
    if len(np.unique(g)) >= 5:
        oof = np.full(len(tr), np.nan)
        for f, h in GroupKFold(n_splits=5).split(tr, groups=g):
            oof[h] = _resid(tr.iloc[f], tr.iloc[h], a, y)
        out["cv"] = (_wquantile(oof, one, 0.1), _wquantile(oof, one, 0.9))
    ss = sorted(tr["outcome_season"].unique())
    if len(ss) >= 2:
        oos = np.concatenate([_resid(tr[tr["outcome_season"].ne(s)], tr[tr["outcome_season"].eq(s)], a, y) for s in ss])
        out["season"] = (_wquantile(oos, np.ones(len(oos)), 0.1), _wquantile(oos, np.ones(len(oos)), 0.9))
    return out


def main() -> None:
    S = seasons()
    assert all(s < CFG.season for ss in S.values() for s in ss), "a development fold may not use the current season"
    lines = build_lines(Loader())
    born = ages()
    res = {"seasons": S, "min_aaa": MIN_AAA, "bands": BANDS, "target": list(TARGET), "seed": SEED}
    for side in METRICS:
        P = pairs(lines, side, born)
        P = P[P["pa_aaa"].ge(MIN_AAA)]
        folds = [int(t) for t in sorted(P["outcome_season"].unique()) if (P["outcome_season"] < t).any()]
        R = {"folds": {}}
        for t in folds:
            F = {"metrics": {}}
            for m in METRICS[side]:
                a, y = f"{m}_aaa", f"{m}_mlb"
                tr = P[P["outcome_season"] < t].dropna(subset=[a, y, "age"])
                te = P[P["outcome_season"] == t].dropna(subset=[a, y, "age"])
                b = _line(tr, a, y)
                r = te[y].to_numpy(float) - (b[0] + b[1] * te[a].to_numpy(float))
                sd = float(tr[y].std())
                F["metrics"][m] = {"n_train": int(len(tr)), "train_seasons": int(tr["outcome_season"].nunique()), "n": int(len(te)),
                                   "level_shift_sd": float(r.mean() / sd),
                                   **{k: {"coverage": float(((r >= lo) & (r <= hi)).mean()), "below": float((r < lo).mean()),
                                          "above": float((r > hi).mean()), "width_sd": float((hi - lo) / sd)}
                                      for k, (lo, hi) in bands(tr, a, y).items()}}
            F["summary"] = {k: {"min": min(v[k]["coverage"] for v in F["metrics"].values() if k in v),
                                "mean": float(np.mean([v[k]["coverage"] for v in F["metrics"].values() if k in v])),
                                "max": max(v[k]["coverage"] for v in F["metrics"].values() if k in v),
                                "in_target": int(sum(TARGET[0] <= v[k]["coverage"] <= TARGET[1] for v in F["metrics"].values() if k in v)),
                                "metrics": len(F["metrics"])}
                            for k in BANDS if any(k in v for v in F["metrics"].values())}
            R["folds"][str(t)] = F
            print(f"\n== {side}, outcome {t}")
            for m, v in F["metrics"].items():
                print(f"  {m:14s} n {v['n']:3d} (fit {v['n_train']:3d}) shift {v['level_shift_sd']:+.2f} sd | " + "  ".join(
                    f"{k} {v[k]['coverage']:.2f} (w {v[k]['width_sd']:.2f})" for k in BANDS if k in v))
            print("  summary: " + "; ".join(f"{k} {s['min']:.2f}-{s['max']:.2f}, mean {s['mean']:.3f}, in target {s['in_target']}/{s['metrics']}"
                                            for k, s in F["summary"].items()))
        res[side] = R
    (OUT / "development_band.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
