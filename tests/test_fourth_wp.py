"""Win probability 2.0.0 (nfl/fourth/wp.py): trees boosted from a logistic baseline that includes field position.
On synthetic plays with a known answer: the model is the baseline plus the trees, it reads through models.predict_wp
like 1.0.x did, and it keeps the field-position slope at the ends of the field, where plays are few and 1.0.1 went flat
(it rated teams inside their own 6 about 4 points too high on seasons it hadn't seen)."""
import numpy as np
import pandas as pd
import pytest

from nfl.fourth import decide as D
from nfl.fourth import models as M
from nfl.fourth import wp as W


def plays(n: int = 60000, seed: int = 0) -> pd.DataFrame:
    """Plays whose chance to win falls steadily from the opponent's goal line to the offense's own, with few plays
    inside either 6 (as in real play-by-play, where most snaps are between the 20s)."""
    rng = np.random.default_rng(seed)
    y = np.clip(rng.normal(55, 22, n).round(), 1, 99)
    sd = rng.normal(0, 9, n).round()
    sec = rng.uniform(0, 3600, n).round()
    home = rng.integers(0, 2, n).astype(float)
    logit = sd / np.sqrt(sec / 60 + 1) * 0.9 + 0.2 * (home - 0.5) + (50 - y) / 45
    d = pd.DataFrame({"season": 2024, "score_differential": sd, "game_seconds_remaining": sec, "half_seconds_remaining": sec % 1800,
                      "half2": (sec < 1800).astype(float), "ot": 0.0, "down": rng.integers(1, 5, n).astype(float),
                      "ydstogo": np.minimum(10.0, y), "yardline_100": y, "posteam_timeouts_remaining": 3.0,
                      "defteam_timeouts_remaining": 3.0, "home": home, "recv_2h": rng.integers(0, 2, n).astype(float),
                      "strength": 0.0, "truth": 1 / (1 + np.exp(-logit))})
    d["win"] = (rng.random(n) < d["truth"]).astype(float)
    return d


@pytest.fixture(scope="module")
def fitted():
    d = plays()
    return d, W.fit(d), M.fit_wp(d)


def test_model_is_the_baseline_plus_the_trees(fitted):
    d, new, _ = fitted
    X = M._prep(d)[M.WP_FEATURES].to_numpy(float)[:500]
    by_hand = 1 / (1 + np.exp(-(new.base.decision_function(W._base_x(X)) + new.trees.predict(X, raw_score=True))))
    assert np.allclose(new.predict(X), by_hand) and np.allclose(M.predict_wp(new, d.iloc[:500]), by_hand)
    assert ((new.predict(X) > 0) & (new.predict(X) < 1)).all()


def test_fit_is_deterministic(fitted):
    d, new, _ = fitted
    assert np.array_equal(W.fit(d).predict(M._prep(d)[M.WP_FEATURES].to_numpy(float)[:2000]),
                          new.predict(M._prep(d)[M.WP_FEATURES].to_numpy(float)[:2000]))


def test_field_position_still_counts_at_the_ends_of_the_field(fitted):
    """Deep in its own end the truth here is well below what it is at the own 25. The baseline carries that slope to
    the goal line; trees alone flatten it where plays are few."""
    d, new, old = fitted
    deep, mid = d[d["yardline_100"] >= 95], d[d["yardline_100"].between(70, 80)]
    assert 100 < len(deep) < 0.05 * len(d)
    err_new = abs(M.predict_wp(new, deep).mean() - deep["truth"].mean())
    err_old = abs(M.predict_wp(old, deep).mean() - deep["truth"].mean())
    assert err_new < 0.02 and err_new < err_old
    assert M.predict_wp(new, mid).mean() - M.predict_wp(new, deep).mean() > 0.6 * (mid["truth"].mean() - deep["truth"].mean())
    close = d[d["yardline_100"] <= 5]
    assert abs(M.predict_wp(new, close).mean() - close["truth"].mean()) < 0.03


def test_no_worse_overall_than_the_trees_alone():
    d, test = plays(seed=1), plays(seed=2)
    brier = lambda m: float(((M.predict_wp(m, test) - test["win"]) ** 2).mean())      # noqa: E731
    assert brier(W.fit(d)) <= brier(M.fit_wp(d)) + 1e-4


def test_shuffled_label_fit_drops_the_monotone_prior():
    d = plays(8000)
    free = W.fit(d, monotone=False)
    assert "monotone_constraints" not in free.trees.params and "monotone_constraints" in W.fit(d).trees.params


def test_the_engine_fits_this_model(monkeypatch):
    seen = {}
    monkeypatch.setattr(W, "fit", lambda d: seen.setdefault("rows", len(d)) and "wp-2")
    monkeypatch.setattr(M, "fit_wp", lambda *a, **k: pytest.fail("the engine fit the 1.0.x model"))
    d = plays(6000)
    d["game_id"] = (np.arange(len(d)) // 150).astype(str)
    d["play_id"] = np.arange(len(d)) % 150
    kind = np.random.default_rng(3).integers(0, 10, len(d))
    d["play_type"] = np.select([kind == 0, kind == 1], ["punt", "field_goal"], "run")
    d["converted"] = np.where((d["down"] == 4) & (kind > 1), (kind % 2).astype(float), np.nan)
    d["yards_gained"] = np.where(d["converted"] == 1, d["ydstogo"] + 2, 0.0)
    d["fg_made"] = np.where(kind == 1, 1.0 * (d["yardline_100"] < 30), np.nan)
    d.loc[kind == 1, "fg_made"] = np.where(np.arange((kind == 1).sum()) % 5 == 0, 0.0, 1.0)
    d["kick_distance"] = np.where(kind == 1, np.minimum(d["yardline_100"], 40) + 18, np.where(kind == 0, 45.0, np.nan))
    d["punt_net"] = np.where(kind == 0, 41.0 + (np.arange(len(d)) % 7), np.nan)
    d["punt_blocked"], d["kickoff_attempt"], d["indoors"], d["wind"] = 0.0, 0.0, 0.0, 5.0
    d["posteam"], d["two_point_attempt"] = "A", 0.0
    d.loc[d.groupby("game_id").head(1).index, "kickoff_attempt"] = 1.0
    eng = D.fit(d)
    assert eng.wp == "wp-2" and seen["rows"] > 0
