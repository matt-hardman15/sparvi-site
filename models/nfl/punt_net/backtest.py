"""NFL net punt yards (F4) in the backtest harness. Rows: every unblocked punt of season t with a kick distance.
Baseline: the training seasons' mean net by 5-yard bucket of field position."""
from __future__ import annotations

import numpy as np

from nfl.fourth import harness as H
from nfl.fourth import models as M


def _punts(d):
    return d[d["punt_net"].notna()].reset_index(drop=True)


def fit(loader, fold, permute_seed: int | None = None) -> dict:
    tr = _punts(H.train(loader, fold))
    if permute_seed is not None:
        tr["punt_net"] = np.random.default_rng(permute_seed).permutation(tr["punt_net"].to_numpy())
    return {"m": M.fit_punt(tr), "b": M.fit_punt_baseline(tr), "median": float(tr["punt_net"].median())}


def predict(fitted, loader, fold):
    d = _punts(H.test(loader, fold))
    return H.rows(d, M.predict_punt(fitted["m"], d), baseline=M.predict_punt_baseline(fitted["b"], d), null=fitted["median"])


def outcome(loader, fold):
    return H.outcome(_punts(H.load_plays_resolved(loader, fold)), "punt_net")
