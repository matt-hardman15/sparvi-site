"""The four fourth-down models (F4) and their pre-registered baselines. Settings are fixed from development on the
2016-2019 seasons only (nfl/fourth/develop.py, output/development.json); the backtest folds (2020-2025) and the 2026
vault never informed them.

    win_prob        P(offense wins) from any scrimmage state          LightGBM, monotone in score margin
    conversion      P(a 4th-down attempt converts)                    logistic regression
    fg_make         P(a field goal is good)                           logistic regression
    punt_net        net punt yards                                    LightGBM regression

Every fit is deterministic (single thread, fixed seed).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

SEED = 7

WP_FEATURES = ["score_differential", "game_seconds_remaining", "half_seconds_remaining", "half2", "ot", "down",
               "ydstogo", "yardline_100", "posteam_timeouts_remaining", "defteam_timeouts_remaining", "home", "recv_2h",
               "strength", "diff_per_time"]
WP_MONOTONE = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
WP_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.9,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=1, deterministic=True, seed=SEED,
                 verbose=-1)
WP_ROUNDS = 150

CONV_FEATURES = ["ydstogo", "yardline_100", "goal_to_go", "season_c", "off_rating", "def_rating", "indoors"]
CONV_SETS = {   # development variants (develop.py); the chosen one is CONV_SET
    "full": CONV_FEATURES,
    "ytg_cat_strength": ["ytg_1", "ytg_2", "ytg_3", "ytg_4", "ytg_5", "ytg_6_10", "goal_to_go", "strength"],
    "ytg_cat": ["ytg_1", "ytg_2", "ytg_3", "ytg_4", "ytg_5", "ytg_6_10", "goal_to_go"],
    "with3rd": ["ytg_1", "ytg_2", "ytg_3", "ytg_4", "ytg_5", "ytg_6_10", "goal_to_go", "is_4th", "is_4th_x_ytg", "strength"],
}
WITH_THIRD = {"with3rd"}   # sets trained on 3rd and 4th downs (scored on 4th downs only)
CONV_SET = "full"
FG_FEATURES = ["kick_distance", "indoors", "temp", "wind", "season_c"]
FG_SETS = {"full": FG_FEATURES, "dist_season": ["kick_distance", "season_c"]}
FG_SET = "full"
PUNT_FEATURES = ["yardline_100", "indoors", "wind", "season_c"]
PUNT_PARAMS = dict(objective="regression_l1", learning_rate=0.05, num_leaves=8, min_data_in_leaf=100, lambda_l2=1.0,
                   num_threads=1, deterministic=True, seed=SEED, verbose=-1)
PUNT_ROUNDS = 75
ERA = 2020                   # season_c = season - ERA (a trend term; centered so coefficients stay small)


def _prep(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["diff_per_time"] = d["score_differential"] / np.sqrt(d["game_seconds_remaining"].clip(lower=0) / 60 + 1)
    d["goal_to_go"] = (d["ydstogo"] >= d["yardline_100"]).astype(float)
    d["season_c"] = d["season"] - ERA
    if "ydstogo" in d:
        y = d["ydstogo"]
        for k in range(1, 6):
            d[f"ytg_{k}"] = (y == k).astype(float)
        d["ytg_6_10"] = y.between(6, 10).astype(float)
    for c in ("temp", "wind", "off_rating", "def_rating", "strength"):
        if c in d:
            d[c] = d[c].fillna(d[c].median() if d[c].notna().any() else 0.0)
    return d


# ---- win probability ---------------------------------------------------------------------------------------------
def fit_wp(train: pd.DataFrame, rounds: int = WP_ROUNDS, params: dict | None = None):
    t = _prep(train)
    ds = lgb.Dataset(t[WP_FEATURES].to_numpy(float), t["win"].to_numpy(float), free_raw_data=False)
    p = dict(WP_PARAMS | (params or {}), monotone_constraints=WP_MONOTONE)
    return lgb.train(p, ds, num_boost_round=rounds)


def predict_wp(model, d: pd.DataFrame) -> np.ndarray:
    return model.predict(_prep(d)[WP_FEATURES].to_numpy(float))


def fit_wp_baseline(train: pd.DataFrame):
    """Baseline: logistic regression on the score margin scaled by time left, plus home field."""
    t = _prep(train)
    return LogisticRegression(C=1.0, max_iter=1000).fit(t[["diff_per_time", "home"]], t["win"])


def predict_wp_baseline(m, d: pd.DataFrame) -> np.ndarray:
    return m.predict_proba(_prep(d)[["diff_per_time", "home"]])[:, 1]


# ---- conversion --------------------------------------------------------------------------------------------------
def conv_rows(d: pd.DataFrame, features: str | None = None) -> pd.DataFrame:
    """Training rows for a conversion variant: 4th-down attempts, plus 3rd downs for the WITH_THIRD sets."""
    four = d[d["converted"].notna()].assign(is_4th=1.0)
    if (features or CONV_SET) not in WITH_THIRD:
        return four
    three = d[d["converted_3rd"].notna()].assign(is_4th=0.0, converted=lambda x: x["converted_3rd"])
    return pd.concat([four, three], ignore_index=True)


def fit_conv(train: pd.DataFrame, C: float = 1.0, features: str | None = None):
    """`train` is all plays; conv_rows picks the rows for the variant."""
    cols = CONV_SETS[features or CONV_SET]
    t = _prep(conv_rows(train, features))
    t["is_4th_x_ytg"] = t["is_4th"] * np.log1p(t["ydstogo"])
    t = t.assign(ydstogo=np.log1p(t["ydstogo"]))
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(t[cols], t["converted"])
    m.sparvi_cols = cols
    return m


def predict_conv(m, d: pd.DataFrame) -> np.ndarray:
    t = _prep(d).assign(is_4th=1.0)
    t["is_4th_x_ytg"] = np.log1p(t["ydstogo"])
    return m.predict_proba(t.assign(ydstogo=np.log1p(t["ydstogo"]))[m.sparvi_cols])[:, 1]


def fit_conv_baseline(train: pd.DataFrame) -> pd.Series:
    """Baseline: the training seasons' conversion rate by yards to go (1-10 each, 11+ pooled)."""
    t = train.assign(b=train["ydstogo"].clip(upper=11))
    return t.groupby("b")["converted"].mean()


def predict_conv_baseline(rates: pd.Series, d: pd.DataFrame) -> np.ndarray:
    return d["ydstogo"].clip(upper=11).map(rates).fillna(rates.mean()).to_numpy(float)


# ---- field goals -------------------------------------------------------------------------------------------------
def _fg_spline():
    return SplineTransformer(n_knots=5, degree=3)


def fit_fg(train: pd.DataFrame, C: float = 1.0, features: str | None = None):
    from sklearn.compose import ColumnTransformer
    cols = FG_SETS[features or FG_SET]
    t = _prep(train)
    ct = ColumnTransformer([("dist", _fg_spline(), ["kick_distance"]),
                            ("rest", StandardScaler(), [c for c in cols if c != "kick_distance"])])
    m = make_pipeline(ct, LogisticRegression(C=C, max_iter=2000)).fit(t[cols], t["fg_made"])
    m.sparvi_cols = cols
    return m


def predict_fg(m, d: pd.DataFrame) -> np.ndarray:
    return m.predict_proba(_prep(d)[m.sparvi_cols])[:, 1]


def fit_fg_baseline(train: pd.DataFrame):
    """Baseline: logistic regression on kick distance alone."""
    return LogisticRegression(C=1.0, max_iter=1000).fit(train[["kick_distance"]], train["fg_made"])


def predict_fg_baseline(m, d: pd.DataFrame) -> np.ndarray:
    return m.predict_proba(d[["kick_distance"]])[:, 1]


# ---- punts -------------------------------------------------------------------------------------------------------
def fit_punt(train: pd.DataFrame, rounds: int = PUNT_ROUNDS):
    t = _prep(train)
    ds = lgb.Dataset(t[PUNT_FEATURES].to_numpy(float), t["punt_net"].to_numpy(float), free_raw_data=False)
    return lgb.train(PUNT_PARAMS, ds, num_boost_round=rounds)


def predict_punt(m, d: pd.DataFrame) -> np.ndarray:
    return m.predict(_prep(d)[PUNT_FEATURES].to_numpy(float))


def fit_punt_baseline(train: pd.DataFrame) -> pd.Series:
    """Baseline: the training seasons' mean net by 5-yard bucket of field position."""
    return train.groupby((train["yardline_100"] // 5).clip(upper=19))["punt_net"].mean()


def predict_punt_baseline(means: pd.Series, d: pd.DataFrame) -> np.ndarray:
    return (d["yardline_100"] // 5).clip(upper=19).map(means).fillna(means.mean()).to_numpy(float)
