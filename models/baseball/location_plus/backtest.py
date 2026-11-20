"""The proposed Location+ (3.0.0, "location value given shape") in the backtest harness (LOC-TYPE).

The published Location+ (2.0.0) sees only where a pitch crossed the plate and the count, and it doesn't predict
(development folds: baseball/pitch_quality/output/development_location.json). The candidate grades the location
given the pitch: the prediction of a model on shape, count, batter side and location, minus the prediction of the
same model without location, averaged over a pitcher's pitches. Both models are trained as Location+ is today (a bag
of 10 over pitcher folds, learning rate 0.03). This module tests the candidate; train.py is unchanged until it passes.

Fold (season t): fit on the two seasons ending at t-2, grade season t-1 pitchers as of March 1 of season t, score
against season t run value per 100 pitches. Baselines on the same pitchers: the published Location+ features (the
gate), Stuff+ and last season's results (reported). Pitchers with MIN_PITCHES+ pitches in both seasons.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from sparvi.config import ROOT

sys.path.insert(0, str(ROOT / "baseball" / "pitch_quality"))
from features import FEATURES, PITCHING_FEATURES, build_features  # noqa: E402
from train import (  # noqa: E402, F401  (MIN_PITCHES: query strings)
    GROUPS,
    LEARNING_RATE,
    MIN_PITCHES,
    N_BAGS,
    SEED,
    bag_folds,
    fit_group,
    fit_kinds,
    pitcher_season,
    predict_all,
)
from train import predict as bag_predict  # noqa: E402

NO_LOC = FEATURES + ["balls", "strikes", "same_hand"]        # everything the full model sees except the location
PARTS = {"full": PITCHING_FEATURES, "no_loc": NO_LOC}
BASELINE_KINDS = ["stuff", "loc"]                            # Stuff+ and the published Location+, as train.py fits them


def _heights(loader, as_of) -> pd.Series:
    h = loader.load("batter_heights", as_of)
    return (h.set_index("batter")["height_in"] / 12).astype(float)


def _pitches(loader, season: int, as_of) -> pd.DataFrame:
    return build_features(loader.load(f"statcast_mlb_{season}", as_of), _heights(loader, as_of))


def _bag(tr: pd.DataFrame, feats: list[str]) -> dict:
    """{group: bag}, trained as train.fit_kinds trains Location+."""
    fold = bag_folds(tr["pitcher"], SEED, N_BAGS)
    return {g: [fit_group(tr[tr["group"].eq(g) & fold.ne(f)], tr[tr["group"].eq(g) & fold.eq(f)], feats,
                          seed=SEED * 100 + f, learning_rate=LEARNING_RATE["loc"]) for f in range(N_BAGS)] for g in GROUPS}


def fit(loader, fold, permute_seed: int | None = None) -> dict:
    tr = pd.concat([_pitches(loader, s, fold.train_until) for s in fold.train_seasons], ignore_index=True)
    if permute_seed is not None:   # label-permutation leakage test: the target shuffled across pitches
        tr["rv_x"] = np.random.default_rng(permute_seed).permutation(tr["rv_x"].to_numpy())
    fitted = {"parts": {name: _bag(tr, feats) for name, feats in PARTS.items()}}
    # The leakage refits check the candidate's own grade ("point"), so they skip the baselines' models.
    if permute_seed is None and loader.poison is None:
        fitted["baselines"] = fit_kinds(tr, kinds=BASELINE_KINDS)
    return fitted


def predict(fitted, loader, fold) -> pd.DataFrame:
    d = _pitches(loader, fold.test_season - 1, fold.decision_date)
    d["value"] = bag_predict(fitted["parts"]["full"], d, PARTS["full"]) - bag_predict(fitted["parts"]["no_loc"], d, PARTS["no_loc"])
    if "baselines" in fitted:
        predict_all(fitted["baselines"], d)
    ps = pitcher_season(d).query("pitches >= @MIN_PITCHES")
    grade = (100 * d.groupby("pitcher")["value"].mean()).rename("point").reset_index()
    ps = ps.merge(grade, on="pitcher")
    nan = pd.Series(np.nan, index=ps.index)
    return pd.DataFrame({"entity_id": ps["pitcher"], "point": ps["point"],
                         "published_location": ps.get("xrv100_loc", nan), "stuff": ps.get("xrv100", nan),
                         "last_season": ps["rv100"], "name": ps["name"], "pitches_prev": ps["pitches"]})


def outcome(loader, fold) -> pd.DataFrame:
    d = _pitches(loader, fold.test_season, fold.resolve_after)
    ps = pitcher_season(d).query("pitches >= @MIN_PITCHES")
    return pd.DataFrame({"entity_id": ps["pitcher"], "y": ps["rv100"]})
