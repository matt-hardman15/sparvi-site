"""Automated leakage tests (PLAN-backtest §4). Run on every backtest; any failure fails the run.

- timestamp (in runner.py): the latest available_at a fold's fit / predict touched is no later than
  its train_until / decision date.
- future_poisoning: rerun the fold with every row available after the decision date corrupted
  (numbers shuffled and noised). Predictions must be bit-identical; any path to future data shows up.
- label_permutation: train on shuffled targets (the model's `fit(..., permute_seed=)` hook; the spec's `permutation_seeds`
  averages several shuffles, for models whose single shuffle is noisy). The model
  must then have no skill: the 90% interval of its correlation with the outcome includes zero. For metrics
  other than correlation (board recall), the 90% interval of its edge over the spec's `null_baseline` (random)
  includes zero.
- too_good: the prediction's correlation with the outcome stays under a ceiling (default 0.9) that no
  honest model of a noisy sports outcome reaches; for other metrics, the metric itself stays under the
  spec's `too_good` ceiling.
- entity_overlap: if the spec groups by entity, no entity is in both the training and test sets.
- universe: predictions cover everyone in the as-of universe; how many later dropped out (no outcome,
  so unscored) is reported, since that's where survivorship bias would hide.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from backtest import metrics
from backtest.asof import Loader


def corrupt_after(cutoff: pd.Timestamp, seed: int = 0):
    """Loader poison hook: rows available after `cutoff` get shuffled and noised numeric columns."""
    def poison(df: pd.DataFrame, avail: pd.Series, as_of: pd.Timestamp) -> pd.DataFrame:
        late = (avail > cutoff).to_numpy()
        if not late.any():
            return df
        df = df.copy()
        rng = np.random.default_rng(seed)
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
                df[c] = df[c].astype("float64")          # corrupted values aren't whole numbers
                v = df.loc[late, c].to_numpy(dtype=float, copy=True)
                rng.shuffle(v)
                df.loc[late, c] = v * (1 + rng.normal(0, 0.5, len(v))) + rng.normal(0, 1, len(v))
        return df
    return poison


def future_poisoning(mod, fold, clean_pred: pd.DataFrame) -> dict:
    loader = Loader(poison=corrupt_after(fold.decision_date))
    pred = mod.predict(mod.fit(loader, fold), loader, fold)
    a = clean_pred.sort_values("entity_id").reset_index(drop=True)
    b = pred.sort_values("entity_id").reset_index(drop=True)
    same = a["entity_id"].tolist() == b["entity_id"].tolist() and np.array_equal(a["point"].to_numpy(float), b["point"].to_numpy(float), equal_nan=True)
    return {"passed": bool(same), "cutoff": fold.decision_date, "n": int(len(b))}


def label_permutation(mod, fold, seed: int = 1, n_boot: int = 1000, spec: dict | None = None) -> dict:
    if "permute_seed" not in inspect.signature(mod.fit).parameters:
        return {"passed": False, "reason": "model has no permute_seed hook in fit()"}
    loader = Loader()
    pred = mod.predict(mod.fit(loader, fold, permute_seed=seed), loader, fold)
    df = pred.merge(mod.outcome(loader, fold), on="entity_id").reset_index(drop=True)
    metric = (spec or {}).get("primary_metric", "pearson_r")
    if metric != "pearson_r":
        null = spec["null_baseline"]
        p = metrics.paired_bootstrap(df, metric, "point", null, "y", spec.get("block"), n=n_boot, seed=seed)
        lo, hi = p["ci90"]
        return {"passed": lo <= 0 <= hi, "edge_over": null, "est": p["est"], "ci90": [lo, hi], "n": int(len(df))}
    seeds = int((spec or {}).get("permutation_seeds", 1))
    if seeds > 1:
        return _permutation_mean(mod, fold, loader, df, seed, seeds)
    rng = np.random.default_rng(seed)
    r = metrics.pearson(df["point"], df["y"])
    draws = [metrics.pearson(*(df.loc[idx, c] for c in ("point", "y"))) for idx in
             (rng.integers(0, len(df), len(df)) for _ in range(n_boot))]
    lo, hi = float(np.percentile(draws, 5)), float(np.percentile(draws, 95))
    return {"passed": lo <= 0 <= hi, "r": r, "ci90": [lo, hi], "n": int(len(df))}


def _permutation_mean(mod, fold, loader, first: pd.DataFrame, seed: int, seeds: int) -> dict:
    """Shuffled-label correlation over `seeds` permutations (the spec's `permutation_seeds`; Matt 2026-11-15). One
    permutation of a model whose rows cluster by player (the profile: ~7 features per player) swings far more than a
    bootstrap of that permutation shows, so a single seed fails often with nothing wrong. A leak shows in every seed:
    pass when the 90% interval of the mean across seeds (t, seeds - 1 df) includes zero."""
    from scipy import stats
    rs = [metrics.pearson(first["point"], first["y"])]
    for k in range(1, seeds):
        pred = mod.predict(mod.fit(loader, fold, permute_seed=seed + k), loader, fold)
        df = pred.merge(mod.outcome(loader, fold), on="entity_id")
        rs.append(metrics.pearson(df["point"], df["y"]))
    m, sd = float(np.mean(rs)), float(np.std(rs, ddof=1))
    half = float(stats.t.ppf(0.95, seeds - 1)) * sd / np.sqrt(seeds)
    lo, hi = m - half, m + half
    return {"passed": bool(lo <= 0 <= hi), "r": m, "ci90": [float(lo), float(hi)], "seeds": seeds, "r_by_seed": [round(r, 4) for r in rs],
            "n": int(len(first))}


def run_all(name: str, mod, spec: dict, fold, scored: pd.DataFrame) -> dict:
    loader = Loader()
    clean = mod.predict(mod.fit(loader, fold), loader, fold)
    out = {"future_poisoning": future_poisoning(mod, fold, clean),
           "label_permutation": label_permutation(mod, fold, spec=spec)}
    metric = spec.get("primary_metric", "pearson_r")
    if metric == "pearson_r":
        ceiling = spec.get("too_good_r", 0.9)
        r = metrics.pearson(scored["point"], scored["y"])
        out["too_good"] = {"passed": abs(r) < ceiling, "r": r, "ceiling": ceiling}
    else:
        ceiling = spec["too_good"]
        v = metrics.score(metric, scored["point"], scored["y"])
        out["too_good"] = {"passed": v < ceiling, metric: v, "ceiling": ceiling}
    if str(spec.get("group_by", "")).startswith("entity"):
        train_ids = set(getattr(mod, "train_entities", lambda *_: set())(loader, fold))
        overlap = train_ids & set(clean["entity_id"])
        out["entity_overlap"] = {"passed": not overlap, "n_overlap": len(overlap)}
    else:
        out["entity_overlap"] = {"passed": True, "not_applicable": f"group_by: {spec.get('group_by')}"}
    n_univ, n_scored = int(clean["entity_id"].nunique()), int(scored[scored["fold"] == fold.test_season]["entity_id"].nunique())
    out["universe"] = {"passed": True, "predicted": n_univ, "scored": n_scored, "no_outcome": n_univ - n_scored,
                       "note": "entities without an outcome (retired, injured, too few pitches) can't be scored; "
                               "they're counted here so survivorship is visible"}
    return out
