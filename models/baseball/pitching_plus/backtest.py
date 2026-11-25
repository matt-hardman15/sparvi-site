"""Pitching+ in the backtest harness (A5-port-PITCHING): the out-of-sample year-over-year test, as Stuff+'s
(models/baseball/stuff_plus/backtest.py), with every row read through the point-in-time loader.

Fold (season t): fit Pitching+ on the two seasons ending at t-2 (as the published model trains), grade season t-1
pitchers as of the decision date (March 1 of season t), and score against season t run value per 100 pitches.
Pitchers with MIN_PITCHES+ pitches in both seasons.

The gate is the partial correlation with the outcome given Stuff+ (Matt, Nov 25 2026): Pitching+ is Stuff+'s
features plus location and count, so the question is what it knows that Stuff+ doesn't. Stuff+ is fit on the same
fold for that. Last season's results and Stuff+ are reported baselines.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from sparvi.config import ROOT

sys.path.insert(0, str(ROOT / "baseball" / "pitch_quality"))
from features import build_features  # noqa: E402
from train import MIN_PITCHES, SEED, fit_kinds, pitcher_season, predict_all, xrv100  # noqa: E402, F401  (MIN_PITCHES: query strings)

KINDS = ["pitching"]
GIVEN = ["stuff"]                # what the gate takes out of both sides


def _heights(loader, as_of) -> pd.Series:
    h = loader.load("batter_heights", as_of)
    return (h.set_index("batter")["height_in"] / 12).astype(float)


def _pitches(loader, season: int, as_of) -> pd.DataFrame:
    return build_features(loader.load(f"statcast_mlb_{season}", as_of), _heights(loader, as_of))


def fit(loader, fold, permute_seed: int | None = None) -> dict:
    tr = pd.concat([_pitches(loader, s, fold.train_until) for s in fold.train_seasons], ignore_index=True)
    if permute_seed is not None:   # label-permutation leakage test: the target shuffled across pitches
        tr["rv_x"] = np.random.default_rng(permute_seed).permutation(tr["rv_x"].to_numpy())
    # The leakage refits check Pitching+'s own grade ("point"), so they skip the Stuff+ fit (the slow one).
    clean = permute_seed is None and loader.poison is None
    return {"models": fit_kinds(tr, kinds=KINDS + (GIVEN if clean else []))}   # bagged, same settings as train.py


def predict(fitted, loader, fold) -> pd.DataFrame:
    d = _pitches(loader, fold.test_season - 1, fold.decision_date)
    predict_all(fitted["models"], d)
    ps = pitcher_season(d).query("pitches >= @MIN_PITCHES")
    nan = pd.Series(np.nan, index=ps.index)
    return pd.DataFrame({"entity_id": ps["pitcher"], "point": ps[xrv100("pitching")], "stuff": ps.get(xrv100("stuff"), nan),
                         "last_season": ps["rv100"], "name": ps["name"], "pitches_prev": ps["pitches"]})


def outcome(loader, fold) -> pd.DataFrame:
    d = _pitches(loader, fold.test_season, fold.resolve_after)
    ps = pitcher_season(d).query("pitches >= @MIN_PITCHES")
    return pd.DataFrame({"entity_id": ps["pitcher"], "y": ps["rv100"]})
