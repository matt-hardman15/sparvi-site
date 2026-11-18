"""NFL win probability (F4) in the backtest harness. Rows: every scrimmage play of season t in a decided game.
Baseline: logistic regression on the score margin scaled by time left, plus home field."""
from __future__ import annotations

import numpy as np

from nfl.fourth import harness as H
from nfl.fourth import models as M
from nfl.fourth.data import scrimmage


def fit(loader, fold, permute_seed: int | None = None) -> dict:
    tr = scrimmage(H.train(loader, fold))
    if permute_seed is not None:   # label-permutation leakage test: outcomes shuffled across games
        g = tr["game_id"].unique()
        flip = dict(zip(g, np.random.default_rng(permute_seed).integers(0, 2, len(g)), strict=True))
        tr["win"] = np.where(tr["game_id"].map(flip) == 1, 1 - tr["win"], tr["win"])
    return {"m": M.fit_wp(tr), "b": M.fit_wp_baseline(tr), "rate": float(tr["win"].mean())}


def predict(fitted, loader, fold):
    d = scrimmage(H.test(loader, fold))
    return H.rows(d, M.predict_wp(fitted["m"], d), baseline=M.predict_wp_baseline(fitted["b"], d), null=fitted["rate"])


def outcome(loader, fold):
    return H.outcome(scrimmage(H.load_plays_resolved(loader, fold)), "win")
