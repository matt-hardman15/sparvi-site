"""College basketball team model (CBB-0), on synthetic games only: a forecast never reads a later game, the ratings
recover known strengths, and a development run cannot load the held-out season."""
import numpy as np
import pandas as pd
import pytest

from cbb.ratings import develop as D
from cbb.ratings import features as F
from cbb.ratings import model as M

P = M.Params(5.0, 1.0, None)


def _games(seed=0, n_teams=12, days=30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 8, n_teams)
    rows, gid = [], 0
    for season in (2021, 2022):
        for d in range(days):
            order = rng.permutation(n_teams)
            for h, a in zip(order[::2], order[1::2], strict=True):
                gid += 1
                he = 104 + strength[h] - strength[a] + 3 + rng.normal(0, 10)
                ae = 104 + strength[a] - strength[h] - 3 + rng.normal(0, 10)
                rows.append({"game_id": gid, "season": season, "date": pd.Timestamp(f"{season - 1}-11-10") + pd.Timedelta(days=d),
                             "home": int(h), "away": int(a), "neutral": False, "poss": 68.0, "tempo": 68.0, "home_eff": he, "away_eff": ae,
                             "home_pts": he * 0.68, "away_pts": ae * 0.68, "margin": (he - ae) * 0.68, "total": (he + ae) * 0.68})
    return pd.DataFrame(rows)


class FakeLoader:
    """Serves synthetic box scores for the feature code (same call as backtest.asof.Loader.load)."""

    def __init__(self, g: pd.DataFrame, seed=3):
        rng = np.random.default_rng(seed)
        self.tables = {}
        for s, gs in g.groupby("season"):
            team, player = [], []
            for r in gs.itertuples():
                for t in (r.home, r.away):
                    team.append({"gameId": r.game_id, "teamId": t,
                                 "teamStats_threePointFieldGoals_made": rng.integers(3, 12), "teamStats_threePointFieldGoals_attempted": rng.integers(15, 30),
                                 "opponentStats_threePointFieldGoals_made": rng.integers(3, 12), "opponentStats_threePointFieldGoals_attempted": rng.integers(15, 30),
                                 **{f"{side}Stats_fourFactors_{v}": rng.uniform(10, 35) for v in F.FOUR.values() for side in ("team", "opponent")}})
                    for k in range(7):          # seven players a team; one sits out about a fifth of the time
                        if rng.random() > 0.2:
                            player.append({"gameId": r.game_id, "season": s, "teamId": t, "player_athleteId": t * 100 + k,
                                           "player_name": f"p{t}_{k}", "player_minutes": float(rng.integers(5, 35))})
            self.tables[f"cbb_team_games_{s}"] = pd.DataFrame(team)
            self.tables[f"cbb_player_games_{s}"] = pd.DataFrame(player)

    def load(self, table, as_of, columns=None):
        d = self.tables[table]
        return (d[columns] if columns else d).copy()


def test_forecasts_do_not_change_when_later_games_are_scrambled():
    g = _games()
    base = M.walk(g, [2022], P)
    cut = pd.Timestamp("2021-11-25")
    g2 = g.copy()
    later = (g2["season"] == 2022) & (g2["date"] >= cut)
    rng = np.random.default_rng(1)
    for c in ("home_eff", "away_eff", "tempo", "margin", "total"):
        g2.loc[later, c] = rng.permutation(g2.loc[later, c].to_numpy()) + 50
    scr = M.walk(g2, [2022], P)
    a, b = base[base["date"] <= cut].set_index("game_id"), scr[scr["date"] <= cut].set_index("game_id")
    assert len(a) > 20
    for c in ("pred_margin", "pred_total", "prior_margin"):
        assert np.allclose(a[c], b.loc[a.index, c])
    after = base[base["date"] > cut + pd.Timedelta(days=3)].set_index("game_id")
    assert not np.allclose(after["pred_margin"], scr.set_index("game_id").loc[after.index, "pred_margin"])     # the test can fail


def test_recency_weighted_forecasts_do_not_read_later_games_either():
    g = _games(seed=2)
    cut = pd.Timestamp("2021-11-20")
    g2 = g.copy()
    later = (g2["season"] == 2022) & (g2["date"] >= cut)
    g2.loc[later, ["home_eff", "away_eff"]] += 40
    a, b = (M.walk(x, [2022], M.Params(2.5, 0.8, 45.0)).set_index("game_id") for x in (g, g2))
    early = a.index[a["date"] <= cut]
    assert np.allclose(a.loc[early, "pred_margin"], b.loc[early, "pred_margin"])


def test_ratings_recover_strength_and_beat_the_prior_and_home_court():
    f = M.walk(_games(), [2022], P)
    late = f[f["date"] > f["date"].min() + pd.Timedelta(days=10)]
    err = (late["margin"] - late["pred_margin"]).abs().mean()
    assert err < (late["margin"] - late["margin"].mean()).abs().mean()            # better than one number for every game
    first = f[f["date"] == f["date"].min()]
    assert np.allclose(first["pred_margin"], first["prior_margin"], atol=1e-6)      # day one is the prior alone
    # last season's strengths carried over: day one already beats no information
    assert (first["margin"] - first["pred_margin"]).abs().mean() < (first["margin"] - first["margin"].mean()).abs().mean() + 2


def test_a_neutral_court_has_no_home_edge():
    g = _games()
    gs = g[g["season"] == 2022]
    teams = M.teams_of(gs)
    r = M.fit_ratings(gs, gs["date"].max() + pd.Timedelta(days=1), teams, M.season_prior(None, None, teams, 1.0, {"mu": 104.0, "hca": 0.0, "mu_t": 68.0}), P)
    assert r["hca"] > 1.0                                                           # the synthetic home edge is 3 points per 100
    d = gs.head(1)
    home, neutral = M.forecast(r, teams, d), M.forecast(r, teams, d.assign(neutral=True))
    assert home["pred_margin"].iloc[0] > neutral["pred_margin"].iloc[0] and np.isclose(home["pred_total"].iloc[0], neutral["pred_total"].iloc[0])


def test_win_probability_is_a_probability_and_follows_the_margin():
    p = M.win_probability(np.array([-30.0, -5.0, 0.0, 5.0, 30.0]), 11.0)
    assert np.all(np.diff(p) > 0) and np.isclose(p[2], 0.5) and np.isclose(p[1] + p[3], 1.0) and 0 < p[0] < 0.01


def test_features_for_a_game_do_not_read_that_game_or_later_ones():
    g = _games()
    f = M.walk(g, [2022], P)
    ld = FakeLoader(g)
    a = F.build(f, ld, [2022], "2022-05-01", 0.45)
    cut = pd.Timestamp("2021-11-25")
    f2, ld2 = f.copy(), FakeLoader(g)
    later = f2["date"] >= cut
    f2.loc[later, "margin"] += 25                                                   # the ratings' error from the cut on
    ids = set(f2.loc[later, "game_id"])
    for name in ("cbb_team_games_2022", "cbb_player_games_2022"):
        t = ld2.tables[name]
        hit = t["gameId"].isin(ids)
        for c in t.columns.difference(["gameId", "teamId", "season", "player_athleteId", "player_name"]):
            t.loc[hit, c] = t.loc[hit, c] * 3 + 7
    b = F.build(f2, ld2, [2022], "2022-05-01", 0.45)
    names = [c for cols in F.GROUPS.values() for c in cols]
    early = (a["date"] <= cut).to_numpy()                                           # includes the cut date's own games
    assert early.sum() > 20 and np.allclose(a.loc[early, names].to_numpy(), b.loc[early, names].to_numpy())
    late = (a["date"] > cut + pd.Timedelta(days=2)).to_numpy()
    for c in ("luck", "form", "missing", "net_to"):                                 # and each does change afterwards
        assert not np.allclose(a.loc[late, c].to_numpy(), b.loc[late, c].to_numpy()), c
    assert a[names].notna().all().all()


def test_a_development_run_cannot_load_the_held_out_season():
    ld = D.DevLoader()
    for table in (f"cbb_games_{D.HELD_OUT}", f"cbb_team_games_{D.HELD_OUT}", f"cbb_player_games_{D.HELD_OUT + 1}", f"cbb_lines_{D.HELD_OUT}"):
        with pytest.raises(PermissionError):
            ld.load(table, D.AS_OF)
    with pytest.raises(PermissionError):                                            # nor any season as of a later date
        ld.load(f"cbb_games_{D.HELD_OUT - 1}", f"{D.HELD_OUT}-04-30")
    assert max([D.SEED_SEASON, *D.TUNE, *D.DEV]) < D.HELD_OUT and pd.Timestamp(D.AS_OF) < pd.Timestamp(f"{D.HELD_OUT - 1}-11-01")


def test_interval_by_game_date_is_reproducible_and_brackets_the_estimate():
    g = _games()
    f = M.walk(g, [2022], P)
    a, b = (f["margin"] - f["pred_margin"]).abs(), (f["margin"] - f["margin"].mean()).abs()
    r = D.by_date_diff(f, a, b)
    assert r == D.by_date_diff(f, a, b) and r["ci90"][0] <= r["est"] <= r["ci90"][1] and r["est"] < 0 and r["dates"] == f["date"].nunique()
