"""F4: team ratings entering a game use only earlier games (the one leakage path a play-level model has)."""
import pandas as pd

from nfl.fourth.data import K_PRIOR, PRIOR_SHRINK, ratings


def _tg(rows):
    return pd.DataFrame(rows, columns=["game_id", "season", "season_type", "game_date", "team", "margin"])


def test_rating_uses_only_earlier_games():
    tg = _tg([("g1", 2020, "REG", "2020-09-10", "A", 10), ("g2", 2020, "REG", "2020-09-17", "A", -4),
              ("g3", 2020, "REG", "2020-09-24", "A", 100)])
    r = ratings(tg).set_index("game_id")["rating"]
    assert r["g1"] == 0                                  # no prior season, no earlier games
    assert r["g2"] == 10 / (1 + K_PRIOR)
    assert r["g3"] == (10 - 4) / (2 + K_PRIOR)           # the 100-point game never rates itself or anything before it


def test_rating_carries_shrunk_last_season():
    tg = _tg([("a", 2019, "REG", "2019-09-10", "A", 8), ("b", 2019, "POST", "2020-01-10", "A", 50),
              ("c", 2020, "REG", "2020-09-10", "A", 0)])
    r = ratings(tg).set_index("game_id")["rating"]
    assert r["c"] == K_PRIOR * PRIOR_SHRINK * 8 / K_PRIOR   # last season's regular-season margin only, shrunk
