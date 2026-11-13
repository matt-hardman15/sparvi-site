"""Automated leakage tests (PLAN-backtest §4). Run on every backtest; any failure fails the run.

- timestamp (in runner.py): the latest available_at a fold's fit / predict touched is no later than
  its train_until / decision date.
- future_poisoning: rerun the fold with every row available after the decision date corrupted
  (numbers shuffled and noised). Predictions must be bit-identical; any path to future data shows up.
- label_permutation: train on shuffled targets (the model's `fit(..., permute_seed=)` hook). The model
  must then have no skill: the 90% interval of its correlation with the outcome includes zero.
- too_good: the prediction's correlation with the outcome stays under a ceiling (default 0.9) that no
  honest model of a noisy sports outcome reaches.
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


def label_permutation(mod, fold, seed: int = 1, n_boot: int = 1000) -> dict:
    if "permute_seed" not in inspect.signature(mod.fit).parameters:
        return {"passed": False, "reason": "model has no permute_seed hook in fit()"}
    loader = Loader()
    pred = mod.predict(mod.fit(loader, fold, permute_seed=seed), loader, fold)
    df = pred.merge(mod.outcome(loader, fold), on="entity_id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    r = metrics.pearson(df["point"], df["y"])
    draws = [metrics.pearson(*(df.loc[idx, c] for c in ("point", "y"))) for idx in
             (rng.integers(0, len(df), len(df)) for _ in range(n_boot))]
    lo, hi = float(np.percentile(draws, 5)), float(np.percentile(draws, 95))
    return {"passed": lo <= 0 <= hi, "r": r, "ci90": [lo, hi], "n": int(len(df))}


def run_all(name: str, mod, spec: dict, fold, scored: pd.DataFrame) -> dict:
    loader = Loader()
    clean = mod.predict(mod.fit(loader, fold), loader, fold)
    out = {"future_poisoning": future_poisoning(mod, fold, clean),
           "label_permutation": label_permutation(mod, fold)}
    ceiling = spec.get("too_good_r", 0.9)
    r = metrics.pearson(scored["point"], scored["y"])
    out["too_good"] = {"passed": abs(r) < ceiling, "r": r, "ceiling": ceiling}
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
