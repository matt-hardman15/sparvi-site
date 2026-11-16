"""DATA-QA safeguards (Matt, Nov 16): range assertions fail the build, positions are inferred only when the roster's is
unusable, and a fit that doesn't converge raises instead of passing silently."""
import numpy as np
import pandas as pd
import pytest

from cfb.fit import data as D
from cfb.proj import pit
from cfb.proj import shares as S


def _team(**over):
    row = {"team": "A", "games": 12, "pass_pg": 30.0, "run_pg": 32.0, "rec_ypd": 7.0, "rb_frac": 0.7}
    row.update(over)
    return pd.DataFrame([row])


def _p():
    return pd.DataFrame({"team": ["A"], "player_id": ["1"], "group": ["QB"], "n": [300], "rate": [6.5]})


def _work(share=0.6):
    return pd.DataFrame({"team": ["A", "A"], "player_id": ["1", "2"], "group": ["REC", "REC"], "share": [share, 0.3]})


def test_plausible_season_passes():
    pit.validate_season(2024, _team(), _p(), _work())


@pytest.mark.parametrize("bad", [{"rec_ypd": 1980.0}, {"pass_pg": 1.0}, {"run_pg": 8.0}, {"rb_frac": 1.2}])
def test_implausible_team_values_fail(bad):
    """UL Monroe 2022's 1,980 receiving yards per dropback and Mississippi State's 8 run plays a game would have failed."""
    with pytest.raises(ValueError, match="range checks"):
        pit.validate_season(2022, _team(**bad), _p(), _work())


def test_shares_over_one_fail():
    with pytest.raises(ValueError, match="shares sum"):
        pit.validate_season(2015, _team(), _p(), _work(share=0.9))


def test_projection_bounds():
    c = pd.DataFrame({"team": ["A", "A"], "group": ["QB", "QB"], "share": [0.7, 0.2], "proj": [3000.0, 9000.0], "key": ["1", "2"]})
    with pytest.raises(ValueError, match="projection"):
        pit.validate_projection(c)


def test_positions_inferred_only_when_unusable():
    """'?' and blank are replaced by the role he played most; a listed position (even defensive) is kept."""
    pos = pd.Series({"a": "?", "b": None, "c": "LB", "d": "WR"})
    ev = pd.DataFrame({"player_id": ["a"] * 12 + ["b"] * 12 + ["c"] * 12 + ["d"] * 12,
                       "opp": ["carry"] * 12 + ["dropback"] * 12 + ["carry"] * 12 + ["target"] * 12})
    out = D.infer_positions(pos, ev)
    assert out["a"] == "RB" and out["b"] == "QB" and out["c"] == "LB" and out["d"] == "WR"


def test_share_fit_raises_when_not_converged(monkeypatch):
    class Bad:
        success, message, nit, x = False, "ABNORMAL", 5000, np.zeros(3)
    monkeypatch.setattr(S, "minimize", lambda *a, **k: Bad())
    with pytest.raises(RuntimeError, match="did not converge"):
        S.fit(np.zeros((2, 2)), np.array(["k", "k"]), np.array([0.5, 0.5]), {"k": 0.0})
