"""Stuff+/Location+/Pitching+ training must be repeatable: same data, same model, same numbers.

The published model numbers (Stuff+, Location+, Pitching+, and the article figures built on them)
come from LightGBM. Multithreaded training without deterministic settings can round differently from
run to run, which moved prose numbers between two refreshes on unchanged data. This trains the real
fit function twice on synthetic rows and requires identical models.
"""
import sys

import numpy as np
import pandas as pd

from sparvi.config import ROOT

sys.path.insert(0, str(ROOT / "baseball" / "pitch_quality"))
import train  # noqa: E402


def _rows(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(train.FEATURES))), columns=train.FEATURES)
    df[train.TARGET] = 0.02 * df[train.FEATURES[0]] + rng.normal(0, 0.25, n)
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
