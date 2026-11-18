"""SYSTEM-FIT F2: player-type matrices (qualification, size cleaning, pooled receivers) and dimensions."""
import numpy as np
import pandas as pd

from cfb.fit import player_types as T


def _profile():
    rows = []
    rng = np.random.default_rng(0)
    for s in (2023, 2024):
        for i in range(40):
            pid, grp = f"p{i}", ("WR" if i < 30 else "TE")
            n_main = 30 if i % 10 else 5                       # every tenth player below the target minimum
            rows.append({"season": s, "player_id": pid, "group": grp, "feature": "rec_target_share", "n": n_main, "shrunk": rng.uniform(.05, .3)})
            for f in T.STYLE["WR"][1:]:
                rows.append({"season": s, "player_id": pid, "group": grp, "feature": f, "n": n_main, "shrunk": rng.normal()})
    return pd.DataFrame(rows)


def _rosters():
    out = {}
    for s in (2023, 2024):
        out[s] = pd.DataFrame({"id": [f"p{i}" for i in range(40)], "height": [73] * 39 + [20], "weight": [200] * 39 + [900]})
    return out


def test_matrix_qualifies_and_cleans_size():
    m = T.matrix(_profile(), _rosters(), "WR")
    assert len(m) == 2 * 27                                     # 30 WRs per season, 3 below the minimum
    assert list(m.columns) == [*T.STYLE["WR"], *T.SIZE]
    rec = T.group_matrix(_profile(), _rosters(), "REC")
    assert set(rec["is_te"]) == {0.0, 1.0} and len(rec) == 2 * 36
    te39 = rec.loc[(2023, "p39")]
    assert te39["height"] == 73 and te39["weight"] == 200        # impossible size replaced by the median


def test_dimensions_project_back():
    m = T.group_matrix(_profile(), _rosters(), "REC")
    d = T.dimensions(m)
    np.testing.assert_allclose(T.project(d, m).to_numpy(), d["coords"].to_numpy(), atol=1e-9)
    for c in d["loadings"].columns:                             # sign convention: each dimension's largest loading positive
        col = d["loadings"][c]
        assert col[col.abs().idxmax()] > 0
