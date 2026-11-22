"""Automated leakage tests (PLAN-backtest §4). Run on every backtest; any failure fails the run.

- timestamp (in runner.py): the latest available_at a fold's fit / predict touched is no later than
  its train_until / decision date.
- future_poisoning: rerun the fold with every row available after the decision date corrupted
  (numbers shuffled and noised). Predictions must be bit-identical; any path to future data shows up.
- label_permutation: train on shuffled targets (the model's `fit(..., permute_seed=)` hook; the spec's `permutation_seeds`
  averages several shuffles, for models whose single shuffle is noisy). The model
  must then have no skill: the 90% interval of its correlation with the outcome includes zero. For metrics
  other than correlation (board recall), the 90% interval of its edge over the spec's `null_baseline` (random)
  includes zero. Error metrics over several shuffles are one-sided, and their interval counts two sources of
  chance: which rows are in the test fold and which shuffles were drawn (A5-PERM).
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
    if metric != "pearson_r" and int((spec or {}).get("permutation_seeds", 1)) > 1:
        return _permutation_mean_loss(mod, fold, loader, df, metric, spec, seed, n_boot)
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


ROW_LOSS = {"brier": lambda p, y: (p - y) ** 2, "mae": lambda p, y: np.abs(p - y), "rmse": lambda p, y: (p - y) ** 2}


def _permutation_mean_loss(mod, fold, loader, first: pd.DataFrame, metric: str, spec: dict, seed: int, n_boot: int) -> dict:
    """Shuffled-label test over `permutation_seeds` seeds for loss metrics (F4, Matt 2026-11-18): each row's loss minus
    the null's, averaged over the seeds. One seed's noise function can line up with a strong real predictor by chance in
    either direction; a leak lines up in every seed. One-sided (Matt, 2026-11-18): a model fit to noise is worse than a
    constant on average (its predictions add variance), so the test fails only when the shuffled-label model is
    significantly *better* than the null (the 90% interval of the loss difference entirely below zero).

    The interval has two parts, added in quadrature (A5-PERM, 2026-11-22). A block bootstrap of the seed-averaged
    difference measures the chance in which rows are in the fold. It cannot see the chance in which shuffles were drawn:
    a shuffled fit makes an error every test row shares (one random slope for everyone), so that error does not average
    out across rows, only across shuffles. That part is a t interval on the per-shuffle means (seeds - 1 df), as
    `_permutation_mean` uses for correlations. With the bootstrap alone, the translation's 2026 fold failed on 5
    shuffles and passed on 20 with nothing wrong either way: per-shuffle values swung by 0.011 inside an interval 0.004
    wide."""
    from scipy import stats
    null, block, seeds = spec["null_baseline"], spec.get("block"), int(spec["permutation_seeds"])
    loss = ROW_LOSS[metric]
    key = first[["entity_id"]].copy()
    diffs, per_seed = [], []
    for k in range(seeds):
        df = first if k == 0 else mod.predict(mod.fit(loader, fold, permute_seed=seed + k), loader, fold).merge(
            mod.outcome(loader, fold), on="entity_id")
        df = key.merge(df, on="entity_id", how="left")
        d = loss(df["point"].to_numpy(float), df["y"].to_numpy(float)) - loss(df[null].to_numpy(float), df["y"].to_numpy(float))
        diffs.append(d)
        per_seed.append(float(np.nanmean(d)))
    mean_d = np.nanmean(np.vstack(diffs), axis=0)
    groups = first[block].to_numpy() if block else np.arange(len(first))
    g = pd.Series(mean_d).groupby(groups)
    sums, counts = g.sum().to_numpy(), g.count().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    est = float(np.nanmean(mean_d))
    rows_lo, rows_hi = float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))
    shuffles = float(stats.t.ppf(0.95, seeds - 1) * np.std(per_seed, ddof=1) / np.sqrt(seeds))
    lo, hi = est - float(np.hypot(est - rows_lo, shuffles)), est + float(np.hypot(rows_hi - est, shuffles))
    return {"passed": hi >= 0, "rule": "one-sided: fail if the shuffled-label model beats the null",
            "edge_over": null, "est": est, "ci90": [lo, hi],
            "interval": "rows (block bootstrap) and shuffles (t across per-shuffle means), in quadrature",
            "ci90_rows_only": [rows_lo, rows_hi], "half_width_shuffles": shuffles,
            "seeds": seeds, "per_seed": per_seed, "n": int(len(first))}


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


NOT_APPLICABLE_OK = {"label_permutation"}        # the timestamp and future-poisoning tests can never be waived


def not_applicable(test: str, declared: dict) -> dict | None:
    """A test the card declares not applicable, with its written reason (Matt, 2026-11-16: for structural models such as
    the CFB projection, whose skill comes from roster composition rather than from what it learns from targets, label
    permutation can't reach zero; timestamp and future poisoning still check that no later data is read)."""
    if test not in declared:
        return None
    reason = str(declared[test] or "").strip()
    if test not in NOT_APPLICABLE_OK or len(reason) < 20:
        return {"passed": False, "reason": f"{test} can't be declared not applicable" if test not in NOT_APPLICABLE_OK
                else "not_applicable needs a written reason"}
    return {"passed": True, "not_applicable": reason}


def run_all(name: str, mod, spec: dict, fold, scored: pd.DataFrame) -> dict:
    loader = Loader()
    clean = mod.predict(mod.fit(loader, fold), loader, fold)
    na = spec.get("not_applicable") or {}
    out = {"future_poisoning": future_poisoning(mod, fold, clean),
           "label_permutation": not_applicable("label_permutation", na) or label_permutation(mod, fold, spec=spec)}
    metric = spec.get("primary_metric", "pearson_r")
    if metric == "pearson_r":
        ceiling = spec.get("too_good_r", 0.9)
        r = metrics.pearson(scored["point"], scored["y"])
        out["too_good"] = {"passed": abs(r) < ceiling, "r": r, "ceiling": ceiling}
    else:   # too good = beyond the spec's bound in the metric's better direction (above it, or below it for errors)
        bound = spec["too_good"]
        v = metrics.score(metric, scored["point"], scored["y"])
        ok = v < bound if metrics.higher_is_better(metric) else v > bound
        out["too_good"] = {"passed": ok, metric: v, "bound": bound,
                           "direction": "must stay below" if metrics.higher_is_better(metric) else "must stay above"}
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
