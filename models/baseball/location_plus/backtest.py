"""Location+ in the backtest harness: the proposed 4.0.0, the same grade trained to move less between seeds (LOC-STABLE).

Location+ (3.0.0, LOC-TYPE) grades the location given the pitch: the prediction of a model on shape, count, batter side
and location, minus the prediction of the same model without location, averaged over a pitcher's pitches. As a
difference of two models it carries the seed noise of both: 0.70 points between two seeds on the development folds,
24% of pitchers moving more than 1 (baseball/pitch_quality/output/development_location_stability.json). The candidate
trains both models at learning rate 0.005 in bags of 20 (CANDIDATE), which cut that to 0.29 points and 2% with the same
r against next-season results. This module tests the candidate; train.py is unchanged until it passes.

Fold (season t): fit on the two seasons ending at t-2, grade season t-1 pitchers as of March 1 of season t, score
against season t run value per 100 pitches. Baselines on the same pitchers: the location-only features of Location+
2.0.0 (the gate), Stuff+ and last season's results (reported), and the same grade as 3.0.x trains it (V3: reported,
the direct comparison). Pitchers with MIN_PITCHES+ pitches in both seasons.
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
    MIN_PITCHES,
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
BASELINE_KINDS = ["stuff", "loc"]                            # Stuff+ and the location-only model, as train.py fits them
# (learning rate, bag members) for both models. Written out here, not read from train.py, so the test of a proposed
# training setup doesn't change when train.py does.
CANDIDATE = (0.005, 20)
V3 = (0.03, 10)                                              # Location+ 3.0.x


def _heights(loader, as_of) -> pd.Series:
    h = loader.load("batter_heights", as_of)
    return (h.set_index("batter")["height_in"] / 12).astype(float)


def _pitches(loader, season: int, as_of) -> pd.DataFrame:
    return build_features(loader.load(f"statcast_mlb_{season}", as_of), _heights(loader, as_of))


def _bag(tr: pd.DataFrame, feats: list[str], setup: tuple[float, int]) -> dict:
    """{group: bag}, trained as train.fit_kinds trains a model family, at this setup's learning rate and bag size."""
    rate, n_bags = setup
    fold = bag_folds(tr["pitcher"], SEED, n_bags)
    return {g: [fit_group(tr[tr["group"].eq(g) & fold.ne(f)], tr[tr["group"].eq(g) & fold.eq(f)], feats,
                          seed=SEED * 100 + f, learning_rate=rate) for f in range(n_bags)] for g in GROUPS}


def _value(parts: dict, d: pd.DataFrame) -> pd.Series:
    return bag_predict(parts["full"], d, PARTS["full"]) - bag_predict(parts["no_loc"], d, PARTS["no_loc"])


def fit(loader, fold, permute_seed: int | None = None) -> dict:
    tr = pd.concat([_pitches(loader, s, fold.train_until) for s in fold.train_seasons], ignore_index=True)
    if permute_seed is not None:   # label-permutation leakage test: the target shuffled across pitches
        tr["rv_x"] = np.random.default_rng(permute_seed).permutation(tr["rv_x"].to_numpy())
    fitted = {"parts": {name: _bag(tr, feats, CANDIDATE) for name, feats in PARTS.items()}}
    # The leakage refits check the candidate's own grade ("point"), so they skip the baselines' models.
    if permute_seed is None and loader.poison is None:
        fitted["baselines"] = fit_kinds(tr, kinds=BASELINE_KINDS)
        fitted["v3"] = {name: _bag(tr, feats, V3) for name, feats in PARTS.items()}
    return fitted


def predict(fitted, loader, fold) -> pd.DataFrame:
    d = _pitches(loader, fold.test_season - 1, fold.decision_date)
    d["value"] = _value(fitted["parts"], d)
    d["value_v3"] = _value(fitted["v3"], d) if "v3" in fitted else np.nan
    if "baselines" in fitted:
        predict_all(fitted["baselines"], d)
    ps = pitcher_season(d).query("pitches >= @MIN_PITCHES")
    grade = (100 * d.groupby("pitcher")[["value", "value_v3"]].mean()).rename(columns={"value": "point", "value_v3": "v3_setup"})
    ps = ps.merge(grade.reset_index(), on="pitcher")
    nan = pd.Series(np.nan, index=ps.index)
    return pd.DataFrame({"entity_id": ps["pitcher"], "point": ps["point"], "v3_setup": ps["v3_setup"],
                         "published_location": ps.get("xrv100_loc", nan), "stuff": ps.get("xrv100", nan),
                         "last_season": ps["rv100"], "name": ps["name"], "pitches_prev": ps["pitches"]})


def outcome(loader, fold) -> pd.DataFrame:
    d = _pitches(loader, fold.test_season, fold.resolve_after)
    ps = pitcher_season(d).query("pitches >= @MIN_PITCHES")
    return pd.DataFrame({"entity_id": ps["pitcher"], "y": ps["rv100"]})
