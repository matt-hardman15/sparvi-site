"""Stuff+ in the backtest harness (A5): the out-of-sample year-over-year test, as in
baseball/pitch_quality/robustness.py, with every row read through the point-in-time loader.

Fold (season t): fit the Stuff+ models on season t-2 (train_until = its last day), grade season t-1
pitchers as of the decision date (March 1 of season t), and score against season t run value.
Baseline: last season's run value (RV/100). Pitchers with MIN_PITCHES+ pitches in both seasons.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from sparvi.config import ROOT

sys.path.insert(0, str(ROOT / "baseball" / "pitch_quality"))
from features import build_features  # noqa: E402
from train import MIN_PITCHES, SEED, fit_kinds, pitcher_season, predict_all  # noqa: E402

KINDS = ["stuff"]


def _heights(loader, as_of) -> pd.Series:
    h = loader.load("batter_heights", as_of)
    return (h.set_index("batter")["height_in"] / 12).astype(float)


def _pitches(loader, season: int, as_of) -> pd.DataFrame:
    return build_features(loader.load(f"statcast_mlb_{season}", as_of), _heights(loader, as_of))


def fit(loader, fold) -> dict:
    tr = pd.concat([_pitches(loader, s, fold.train_until) for s in fold.train_seasons], ignore_index=True)
    rng = np.random.default_rng(SEED)            # same seeded 15% pitcher split for early stopping as train.py
    val_p = set(rng.choice(tr["pitcher"].unique(), size=int(0.15 * tr["pitcher"].nunique()), replace=False))
    return {"models": fit_kinds(tr, tr["pitcher"].isin(val_p), kinds=KINDS)}


def predict(fitted, loader, fold) -> pd.DataFrame:
    d = _pitches(loader, fold.test_season - 1, fold.decision_date)
    predict_all(fitted["models"], d)
    ps = pitcher_season(d).query("pitches >= @MIN_PITCHES")
    return pd.DataFrame({"entity_id": ps["pitcher"], "point": ps["xrv100"], "last_season": ps["rv100"],
                         "name": ps["name"], "pitches_prev": ps["pitches"]})


def outcome(loader, fold) -> pd.DataFrame:
    d = _pitches(loader, fold.test_season, fold.resolve_after)
    ps = pitcher_season(d).query("pitches >= @MIN_PITCHES")
    return pd.DataFrame({"entity_id": ps["pitcher"], "y": ps["rv100"]})
