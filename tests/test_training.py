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


# ---- stored models (MODEL-ARTIFACTS) ---------------------------------------------------------------
def _raw(tmp_path, seasons=(2024, 2025), extra_batter=None):
    """A stand-in data/raw: one small file per season with the batters in it, and the heights file."""
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    for y in seasons:
        pd.DataFrame({"batter": [1, 2, 3, y], "x": [0.1, 0.2, 0.3, 0.4]}).to_parquet(raw / f"statcast_mlb_{y}.parquet", index=False)
    batters = [1, 2, 3, *seasons] + ([extra_batter] if extra_batter else [])
    pd.DataFrame({"batter": batters, "height_in": [70.0 + i for i in range(len(batters))]}).to_parquet(raw / "batter_heights.parquet", index=False)
    return raw


def _get(tmp_path, monkeypatch, fit, raw, official=False, store=None, **kw):
    import artifacts
    monkeypatch.setenv("SPARVI_MODELS", "official" if official else "")
    return artifacts.get("production", [2024, 2025], fit, store=store, raw=raw, local=tmp_path / "store",
                         record=tmp_path / "model_artifacts.json", log=lambda s: None, **kw)


def test_stored_models_predict_exactly_what_the_fitted_ones_did(tmp_path, monkeypatch):
    """A fit is kept and reused: the second call never trains, and the reloaded bags predict bit for bit what the
    in-memory ones did, with the same trees and the same gain importance."""
    df = _rows(30_000, 5)
    for i, c in enumerate(sorted({c for cols in train.FEATURE_SETS.values() for c in cols} - set(df.columns))):
        df[c] = np.random.default_rng(50 + i).normal(size=len(df))
    fresh = train.fit_kinds(df, kinds=["stuff", "no_loc", "pitching"], n_bags=3)
    raw = _raw(tmp_path)
    first = _get(tmp_path, monkeypatch, lambda: fresh, raw)

    def no_fit():
        raise AssertionError("trained although a stored fit exists")
    again = _get(tmp_path, monkeypatch, no_fit, raw)
    for models in (first, again):
        a, b = df.copy(), df.copy()
        train.predict_all(fresh, a)
        train.predict_all(models, b)
        for col in ("xrv", "xrv_no_loc", "xrv_pitching", "xrv_loc"):
            assert np.array_equal(a[col].to_numpy(), b[col].to_numpy()), col
        for k in fresh:
            for g in train.GROUPS:
                for m, r in zip(fresh[k][g], models[k][g], strict=True):
                    assert r.best_iteration == r.num_trees() == m.best_iteration
                    assert np.array_equal(m.feature_importance("gain", iteration=m.best_iteration), r.feature_importance("gain", iteration=r.best_iteration))


def test_model_key_follows_training_inputs_only(tmp_path):
    """The key changes with a training season's file, the model code or a training batter's height, and not with a
    batter who joins the heights file later (a call-up during the season)."""
    import artifacts
    raw = _raw(tmp_path)
    code = tmp_path / "code"
    code.mkdir()
    for f in artifacts.KEY_CODE:
        (code / f).write_text("x = 1\n", encoding="utf-8")
    key = lambda: artifacts.key_of(artifacts.inputs("production", [2024, 2025], raw, code))  # noqa: E731
    base = key()
    assert key() == base and artifacts.key_of(artifacts.inputs("oos", [2024], raw, code)) != base
    _raw(tmp_path, extra_batter=999)                                   # a new batter in the heights file only
    assert key() == base
    (code / "features.py").write_text("x = 2\n", encoding="utf-8")
    changed = key()
    assert changed != base
    (code / "features.py").write_text("x = 1\r\n", encoding="utf-8", newline="")   # line endings don't count
    assert key() == base
    h = pd.read_parquet(raw / "batter_heights.parquet")
    h.loc[h["batter"].eq(2), "height_in"] = 99.0
    h.to_parquet(raw / "batter_heights.parquet", index=False)
    assert key() != base
    _raw(tmp_path)
    pd.DataFrame({"batter": [1, 2, 3, 2025], "x": [0.1, 0.2, 0.3, 0.5]}).to_parquet(raw / "statcast_mlb_2025.parquet", index=False)
    assert key() != base


def test_official_runs_upload_record_and_refuse_a_different_model(tmp_path, monkeypatch):
    """The refresh: a new fit goes to the bucket and onto the record; the next run loads it from the bucket without
    training; a bundle that doesn't match the record stops the run; nothing is uploaded or recorded elsewhere."""
    import json

    import artifacts

    from ingest.common.snapshots import LocalStore
    df = _rows(20_000, 6)
    a = train.fit_kinds(df, kinds=["stuff"], n_bags=2)
    b = train.fit_kinds(_rows(20_000, 7), kinds=["stuff"], n_bags=2)
    raw, bucket, record = _raw(tmp_path), LocalStore(tmp_path / "bucket"), tmp_path / "model_artifacts.json"

    _get(tmp_path, monkeypatch, lambda: a, raw)                                    # a laptop: local store only
    assert not record.exists() and bucket.list("models") == []
    (tmp_path / "store").rename(tmp_path / "laptop_store")

    fits = []
    def fit():  # noqa: E306
        fits.append(1)
        return a
    _get(tmp_path, monkeypatch, fit, raw, official=True, store=bucket, verify=True)
    rec = json.loads(record.read_text(encoding="utf-8"))["production"]
    assert len(fits) == 2, "an official fit with verify=True trains twice"
    assert bucket.list("models") == [rec["object"]] and rec["sha256"] == artifacts._sha(bucket.get(rec["object"]))

    for f in (tmp_path / "store").iterdir():                                         # a new runner: empty local store
        f.unlink()
    got = _get(tmp_path, monkeypatch, lambda: (_ for _ in ()).throw(AssertionError("trained")), raw, official=True, store=bucket)
    assert got["stuff"]["fastball"][0].model_to_string() == artifacts.unbundle(bucket.get(rec["object"]))["stuff"]["fastball"][0].model_to_string()

    for f in (tmp_path / "store").iterdir():                                         # a different model under the recorded key
        f.write_bytes(artifacts.bundle(b))
    import pytest
    with pytest.raises(artifacts.ModelMismatch, match="only scores with the model on record"):
        _get(tmp_path, monkeypatch, fit, raw, official=True, store=None)

    flip = iter([a, b])
    for f in (tmp_path / "store").iterdir():
        f.unlink()
    (tmp_path / "bucket2").mkdir()
    record.unlink()
    with pytest.raises(artifacts.ModelMismatch, match="not repeatable"):
        _get(tmp_path, monkeypatch, lambda: next(flip), raw, official=True, store=LocalStore(tmp_path / "bucket2"), verify=True)
