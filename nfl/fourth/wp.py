"""NFL win probability, 2.0.0 (F4-WP2): the 1.0.x trees, boosted from a logistic baseline that includes field
position.

Why: on the 2020-2025 seasons it hadn't seen, 1.0.1 rated a team with the ball inside its own 6 about 4 points too
high (2,200 plays: won 46.6%, rated 51.0%; 90% interval of the gap 2.4 to 6.4 points, resampling games) and slightly
underrated one inside the opponent's 6. Field position barely moved it at either end: there are few plays there, and
150 small trees spend their splits on score and time. That made a fourth-down conversion deep in a team's own end
look better than it is (found in the Nov. 2026 audit of the fourth-down chart, pull request #104).

What changed: a logistic regression on the score margin scaled by time, home field, team strength, who receives
the 2nd-half kickoff, and a cubic in field position is fit first; the trees (same features, settings and monotone
constraints as 1.0.1) then start from its log-odds instead of from the base rate. The baseline carries a smooth
field-position slope all the way to both goal lines; the trees correct it where there are plays to learn from.

Chosen by Matt (Nov 20, 2026) after the 2020-2025 backtest of 1.0.1 was read, so those seasons are not a clean test
of this version: the 2026 season is. On the development seasons (validation 2018 and 2019, nfl/fourth/develop.py) it
scored a Brier of 0.1470 against 0.1491 for 1.0.1.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from nfl.fourth import models as M

BASE = ["diff_per_time", "home", "strength", "recv_2h"]      # the baseline's terms, besides the cubic in field position
_COL = {c: M.WP_FEATURES.index(c) for c in [*BASE, "yardline_100"]}


def _base_x(X: np.ndarray) -> np.ndarray:
    y = X[:, _COL["yardline_100"]] / 100
    return np.column_stack([*(X[:, _COL[c]] for c in BASE), y, y ** 2, y ** 3])


class Model:
    """The baseline and the trees fit on top of it. `predict` takes the WP_FEATURES matrix, as a LightGBM booster
    does, so models.predict_wp reads either version."""

    def __init__(self, base, trees):
        self.base, self.trees = base, trees

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        return 1 / (1 + np.exp(-(self.base.decision_function(_base_x(X)) + self.trees.predict(X, raw_score=True))))


def fit(train: pd.DataFrame, rounds: int = M.WP_ROUNDS, params: dict | None = None, monotone: bool = True) -> Model:
    """`train`: scrimmage plays (data.scrimmage). `monotone=False` drops the trees' constraints (the backtest's
    shuffled-label test only)."""
    t = M._prep(train)
    X, y = t[M.WP_FEATURES].to_numpy(float), t["win"].to_numpy(float)
    # Effectively unpenalized (seven terms, hundreds of thousands of plays); lbfgs is deterministic.
    base = LogisticRegression(C=1e4, max_iter=5000).fit(_base_x(X), y)
    p = dict(M.WP_PARAMS | (params or {}), boost_from_average=False)
    if monotone:
        p["monotone_constraints"] = M.WP_MONOTONE
    ds = lgb.Dataset(X, y, init_score=base.decision_function(_base_x(X)), free_raw_data=False)
    return Model(base, lgb.train(p, ds, num_boost_round=rounds))


predict = M.predict_wp
