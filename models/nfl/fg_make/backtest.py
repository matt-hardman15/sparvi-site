"""NFL field-goal make probability (F4) in the backtest harness. 2.0.0 (F4-V2): logistic regression on kick distance
and its square, fit on the last three seasons before the test season (nfl/fourth/fg.py). Gated on calibration. Rows:
every field-goal attempt of season t (blocked kicks count as misses)."""
from __future__ import annotations

import numpy as np

from nfl.fourth import fg as F
from nfl.fourth import harness as H


def _kicks(d):
    return d[d["fg_made"].notna()].reset_index(drop=True)


def fit(loader, fold, permute_seed: int | None = None) -> dict:
    tr = _kicks(H.train(loader, fold))
    if permute_seed is not None:
        tr["fg_made"] = np.random.default_rng(permute_seed).permutation(tr["fg_made"].to_numpy())
    return {"m": F.fit(tr)}


def predict(fitted, loader, fold):
    d = _kicks(H.test(loader, fold))
    return H.rows(d, F.predict(fitted["m"], d["kick_distance"]), kick_distance=d["kick_distance"].to_numpy())


def outcome(loader, fold):
    return H.outcome(_kicks(H.load_plays_resolved(loader, fold)), "fg_made")
