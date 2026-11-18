"""NFL 4th-down conversion (F4) in the backtest harness: the training seasons' conversion rate by yards to go (1-10
each, 11+ pooled), chosen in development over every richer model tried. Gated on calibration. Rows: every 4th-down
run or pass of season t (fakes included; kneels, spikes and plays wiped out by penalty excluded)."""
from __future__ import annotations

import numpy as np

from nfl.fourth import harness as H
from nfl.fourth import models as M


def _tries(d):
    return d[d["converted"].notna()].reset_index(drop=True)


def fit(loader, fold, permute_seed: int | None = None) -> dict:
    tr = _tries(H.train(loader, fold))
    if permute_seed is not None:
        tr["converted"] = np.random.default_rng(permute_seed).permutation(tr["converted"].to_numpy())
    return {"m": M.fit_conv_baseline(tr)}


def predict(fitted, loader, fold):
    d = _tries(H.test(loader, fold))
    return H.rows(d, M.predict_conv_baseline(fitted["m"], d), ydstogo=d["ydstogo"].to_numpy())


def outcome(loader, fold):
    return H.outcome(_tries(H.load_plays_resolved(loader, fold)), "converted")
