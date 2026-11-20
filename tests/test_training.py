"""Stuff+/Location+/Pitching+ training must be repeatable: same data, same model, same numbers.

The published model numbers (Stuff+, Location+, Pitching+, and the article figures built on them)
come from LightGBM. Multithreaded training without deterministic settings can round differently from
run to run, which moved prose numbers between two refreshes on unchanged data. These train the real
fit functions twice on synthetic rows and require identical models, and check the bag's folds (v2.0.0):
every pitcher in exactly one held-out fold, each model early-stopping on pitchers it never trained on.
"""
import sys

import numpy as np
import pandas as pd

from sparvi.config import ROOT

sys.path.insert(0, str(ROOT / "baseball" / "pitch_quality"))
import train  # noqa: E402


def _rows(n: int, seed: int, n_pitchers: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(train.FEATURES))), columns=train.FEATURES)
    df[train.TARGET] = 0.02 * df[train.FEATURES[0]] + rng.normal(0, 0.25, n)
    df["pitcher"] = rng.integers(0, n_pitchers, n)
    df["group"] = rng.choice(train.GROUPS, n)
    return df


def test_params_pin_determinism():
    p = train.PARAMS
    assert p["deterministic"] is True
    assert p.get("force_row_wise") or p.get("force_col_wise")
    assert isinstance(p.get("num_threads"), int) and p["num_threads"] > 0


def test_same_data_same_model():
    tr, va = _rows(20_000, 1), _rows(4_000, 2)
    a = train.fit_group(tr, va).model_to_string()
    b = train.fit_group(tr, va).model_to_string()
    assert a == b


def test_bag_folds_hold_out_each_pitcher_once():
    df = _rows(5_000, 3)
    fold = train.bag_folds(df["pitcher"], seed=7, n_bags=5)
    per_pitcher = fold.groupby(df["pitcher"]).nunique()
    assert (per_pitcher == 1).all(), "a pitcher's pitches must all sit in one fold"
    assert sorted(fold.unique()) == [0, 1, 2, 3, 4]
    sizes = df.groupby(fold)["pitcher"].nunique()
    assert sizes.max() - sizes.min() <= 1, "folds split the pitchers evenly"
    assert fold.equals(train.bag_folds(df["pitcher"], seed=7, n_bags=5))
    assert not fold.equals(train.bag_folds(df["pitcher"], seed=8, n_bags=5))


def test_same_data_same_bag():
    df = _rows(30_000, 4)
    a = train.fit_kinds(df, kinds=["stuff"], n_bags=3)
    b = train.fit_kinds(df, kinds=["stuff"], n_bags=3)
    for g in train.GROUPS:
        assert [m.model_to_string() for m in a["stuff"][g]] == [m.model_to_string() for m in b["stuff"][g]]
    pa, pb = train.predict(a["stuff"], df, train.FEATURE_SETS["stuff"]), train.predict(b["stuff"], df, train.FEATURE_SETS["stuff"])
    assert pa.notna().all() and pa.equals(pb)


def test_location_value_is_the_full_model_minus_the_model_without_location():
    """Location+ 3.0.0: predict_all derives xrv_loc from the two fitted parts, and only when both are there."""
    rng = np.random.default_rng(5)
    df = _rows(30_000, 5)
    for c in set(train.FEATURE_SETS["pitching"]) - set(train.FEATURES):
        df[c] = rng.normal(size=len(df))
    df[train.TARGET] += 0.05 * df["loc_x"]
    assert set(train.FEATURE_SETS["pitching"]) - set(train.FEATURE_SETS["no_loc"]) == {"loc_x", "loc_z"}
    assert "loc" not in train.MODELS and "loc" in train.KINDS
    train.predict_all(train.fit_kinds(df, kinds=["no_loc", "pitching"], n_bags=3), df)
    assert df["xrv_loc"].equals(df["xrv_pitching"] - df["xrv_no_loc"])
    assert df["xrv_loc"].corr(df["loc_x"]) > 0.5, "the location's value should follow the location"
    only = _rows(5_000, 6)
    train.predict_all(train.fit_kinds(only, kinds=["stuff"], n_bags=3), only)
    assert "xrv_loc" not in only
