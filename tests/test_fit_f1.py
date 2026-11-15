"""SYSTEM-FIT F1: style standardization, dimensions (sign convention, projection), and system labels."""
import numpy as np
import pandas as pd

from cfb.fit import systems as S


def _styles(n=300, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, len(S.STYLE)))
    base[:, S.STYLE.index("early_pass_rate")] = base[:, S.STYLE.index("pass_rate")] + rng.normal(0, 0.3, n)
    idx = pd.MultiIndex.from_tuples([(2014 + i % 10, f"T{i}") for i in range(n)], names=["season", "team"])
    return pd.DataFrame(base, index=idx, columns=S.STYLE)


def test_standardize_clips_extremes():
    st = _styles()
    st.iloc[0, 0] = 1e6
    z = S.standardize(st, st)
    assert z.to_numpy().max() <= S.CLIP and z.to_numpy().min() >= -S.CLIP


def test_dimensions_signs_and_projection():
    st = _styles()
    d = S.dimensions(st)
    assert d["loadings"].loc["pass_rate", "D1"] > 0 and d["loadings"].loc["wr_target_share", "D2"] > 0
    np.testing.assert_allclose(S.project(d, st).to_numpy(), d["coords"].to_numpy(), atol=1e-9)
    assert abs(sum(d["explained"])) <= 1


def test_system_labels_are_deterministic_and_ordered():
    st = _styles()
    a, b = S.fit(st, 3), S.fit(st, 3)
    assert a["probs"].equals(b["probs"])
    assert np.allclose(a["probs"].sum(axis=1), 1)
    pr = a["means"]["pass_rate"].to_numpy()
    assert (np.diff(pr) >= 0).all()          # numbered from most run-heavy to most pass-heavy
