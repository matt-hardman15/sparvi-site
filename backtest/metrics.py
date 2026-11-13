"""Scoring for backtests (PLAN-backtest §3): point metrics and paired comparisons with a block bootstrap.

Blocks are whatever groups make the resamples honest: a player for season targets (his rows move
together), a week for game targets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pearson(x, y) -> float:
    return float(pd.Series(np.asarray(x, float)).corr(pd.Series(np.asarray(y, float))))


def mae(pred, y) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(pred, float))))


def rmse(pred, y) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(pred, float)) ** 2)))


METRICS = {"pearson_r": (pearson, True), "mae": (mae, False), "rmse": (rmse, False)}


def score(metric: str, pred, y) -> float:
    fn, _ = METRICS[metric]
    return fn(pred, y)


def higher_is_better(metric: str) -> bool:
    return METRICS[metric][1]


def paired_bootstrap(df: pd.DataFrame, metric: str, a: str, b: str, y: str, block: str | None,
                     n: int = 2000, seed: int = 7) -> dict:
    """metric(a) - metric(b) on the same rows, with a 90% interval from resampling blocks (rows if None).
    Positive means `a` is better for metrics where higher is better; negative otherwise."""
    rng = np.random.default_rng(seed)
    groups = [df.index[df[block] == g].to_numpy() for g in df[block].unique()] if block else [np.array([i]) for i in df.index]
    est = score(metric, df[a], df[y]) - score(metric, df[b], df[y])
    draws = np.empty(n)
    for i in range(n):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        s = df.loc[idx]
        draws[i] = score(metric, s[a], s[y]) - score(metric, s[b], s[y])
    return {"est": est, "ci90": [float(np.percentile(draws, 5)), float(np.percentile(draws, 95))], "n": int(len(df)),
            "blocks": len(groups), "bootstrap": n, "seed": seed}
