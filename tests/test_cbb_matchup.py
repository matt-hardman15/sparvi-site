"""The matchup tool's arithmetic (CBB-MATCHUP): cbb/ratings/matchup.py from the PUBLISHED ratings equals the model's own
forecast from the ratings it came from. tests/test_site.py then holds the page's numbers equal to matchup.py."""
import numpy as np
import pandas as pd

from cbb.ratings import model as M
from cbb.ratings.matchup import Z80, expected
from tests.test_cbb_model import _games


def _ratings():
    g = _games()
    gs = g[g["season"] == 2022]
    teams = M.teams_of(gs)
    r = M.fit_ratings(gs, gs["date"].max() + pd.Timedelta(days=1), teams,
                      M.season_prior(None, None, teams, 1.0, {"mu": 104.0, "hca": 0.0, "mu_t": 68.0}), M.PUBLISHED)
    return r, teams


def test_published_ratings_reproduce_the_models_forecast():
    r, teams = _ratings()
    # What the export publishes: tempo as possessions against an average opponent, and the league terms.
    pub = {int(t): {"off": r["off"][i], "def": r["def"][i], "tempo": r["mu_t"] + r["tempo"][i]} for i, t in enumerate(teams)}
    league = {"efficiency": r["mu"], "tempo": r["mu_t"], "home_court": r["hca"], "margin_sd": 11.0}
    for home, away in ((0, 1), (5, 2), (7, 11)):
        for neutral in (False, True):
            d = pd.DataFrame({"game_id": [1], "home": [home], "away": [away], "neutral": [neutral]})
            f = M.forecast(r, teams, d).iloc[0]
            x = expected(pub[home], pub[away], "n" if neutral else "a", league)
            assert np.isclose(x["margin"], f["pred_margin"]) and np.isclose(x["total"], f["pred_total"]) and np.isclose(x["poss"], f["pred_poss"])
            assert np.isclose(x["a_pts"] - x["b_pts"], x["margin"]) and np.isclose(x["a_pts"] + x["b_pts"], x["total"])
            assert np.isclose(x["p_a"], M.win_probability([f["pred_margin"]], 11.0)[0])


def test_site_and_symmetry():
    r, teams = _ratings()
    pub = {int(t): {"off": r["off"][i], "def": r["def"][i], "tempo": r["mu_t"] + r["tempo"][i]} for i, t in enumerate(teams)}
    league = {"efficiency": r["mu"], "tempo": r["mu_t"], "home_court": r["hca"], "margin_sd": 11.0}
    a, b = pub[3], pub[8]
    at_a, at_b, neutral = (expected(a, b, s, league) for s in ("a", "b", "n"))
    assert at_a["margin"] > neutral["margin"] > at_b["margin"] and np.isclose(at_a["total"], at_b["total"])       # home court moves the margin only
    flipped = expected(b, a, "b", league)                                                                           # the same game, named the other way
    assert np.isclose(flipped["margin"], -at_a["margin"]) and np.isclose(flipped["p_a"], 1 - at_a["p_a"])
    assert np.isclose(at_a["hi80"] - at_a["lo80"], 2 * Z80 * 11.0) and at_a["lo80"] < at_a["margin"] < at_a["hi80"]
    same = expected(a, a, "n", league)
    assert np.isclose(same["margin"], 0) and np.isclose(same["p_a"], 0.5)
