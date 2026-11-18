"""NFL field-goal make probability (F4) in the backtest harness: logistic regression on kick distance, chosen in
development over models adding roof, temperature, wind and season. Gated on calibration. Rows: every field-goal
attempt of season t (blocked kicks count as misses)."""
from __future__ import annotations

import numpy as np

from nfl.fourth import harness as H
from nfl.fourth import models as M


def _kicks(d):
    return d[d["fg_made"].notna()].reset_index(drop=True)


def fit(loader, fold, permute_seed: int | None = None) -> dict:
    tr = _kicks(H.train(loader, fold))
    if permute_seed is not None:
        tr["fg_made"] = np.random.default_rng(permute_seed).permutation(tr["fg_made"].to_numpy())
    return {"m": M.fit_fg_baseline(tr)}


def predict(fitted, loader, fold):
    d = _kicks(H.test(loader, fold))
    return H.rows(d, M.predict_fg_baseline(fitted["m"], d), kick_distance=d["kick_distance"].to_numpy())


def outcome(loader, fold):
    return H.outcome(_kicks(H.load_plays_resolved(loader, fold)), "fg_made")
