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


def recall(credit, y) -> float:
    """Board recall: the mean hit credit (0-1, e.g. the chance of ranking in a board's top 10) over the positive rows."""
    c, y = np.asarray(credit, float), np.asarray(y, float)
    return float(c[y == 1].mean())


def brier(p, y) -> float:
    """Mean squared error of a probability against a 0/1 outcome (lower is better)."""
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def log_loss(p, y) -> float:
    """Mean negative log likelihood of a 0/1 outcome; probabilities are clipped to [1e-6, 1 - 1e-6]."""
    p, y = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6), np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration(p, y, bins: int = 10) -> list[dict]:
    """Reliability table (reported, not gating): equal-count bins of the prediction, mean predicted vs. observed."""
    d = pd.DataFrame({"p": np.asarray(p, float), "y": np.asarray(y, float)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False)
    g = d.groupby("bin").agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    return [{k: (int(v) if k == "n" else round(float(v), 4)) for k, v in r.items()} for r in g.to_dict("records")]


def ece(p, y, bins: int = 10) -> float:
    """Expected calibration error: the count-weighted mean |predicted - observed| over equal-count bins."""
    t = calibration(p, y, bins)
    n = sum(r["n"] for r in t)
    return float(sum(r["n"] * abs(r["predicted"] - r["observed"]) for r in t) / n)


def ece_noise(p, q: float = 0.9, draws: int = 2000, seed: int = 7, bins: int = 10) -> dict:
    """The ECE that perfectly calibrated predictions would show on these same rows: outcomes drawn as Bernoulli(p),
    `draws` times. Returns the q-quantile (the calibration gate's bar) and the median."""
    p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    sims = np.array([ece(p, (rng.random(len(p)) < p).astype(float), bins) for _ in range(draws)])
    return {"q": q, "draws": draws, "bar": float(np.quantile(sims, q)), "median": float(np.median(sims))}


METRICS = {"pearson_r": (pearson, True), "mae": (mae, False), "rmse": (rmse, False), "recall": (recall, True),
           "brier": (brier, False), "log_loss": (log_loss, False), "ece": (ece, False)}


def score(metric: str, pred, y) -> float:
    fn, _ = METRICS[metric]
    return fn(pred, y)


def higher_is_better(metric: str) -> bool:
    return METRICS[metric][1]


def partial_r(x, y, z) -> float:
    """Correlation of x with y once z is taken out of both."""
    rxy, rxz, ryz = pearson(x, y), pearson(x, z), pearson(y, z)
    return float((rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)))


def partial_bootstrap(df: pd.DataFrame, a: str, given: str, y: str, block: str | None, n: int = 2000, seed: int = 7) -> dict:
    """partial_r(a, y | given) with a 90% interval from resampling blocks (rows if None). Reported, never gated."""
    rng = np.random.default_rng(seed)
    groups = [df.index[df[block] == g].to_numpy() for g in df[block].unique()] if block else [np.array([i]) for i in df.index]
    draws = np.empty(n)
    for i in range(n):
        s = df.loc[np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])]
        draws[i] = partial_r(s[a], s[y], s[given])
    return {"given": given, "est": partial_r(df[a], df[y], df[given]),
            "ci90": [float(np.percentile(draws, 5)), float(np.percentile(draws, 95))], "n": int(len(df)),
            "blocks": len(groups), "bootstrap": n, "seed": seed}


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
