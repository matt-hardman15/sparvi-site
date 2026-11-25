"""One day's college basketball forecasts (CBB-DAILY), on synthetic games: they equal the walk-forward's forecasts for
that day, they don't change when that day's own results or any later game changes, and a team with no game yet is
forecast from its prior."""
import numpy as np
import pandas as pd

from cbb.ratings import daily as D
from cbb.ratings import features as F
from cbb.ratings import model as M
from tests.test_cbb_model import FakeLoader, _games

SEASONS = [2021, 2022]


class DayLoader(FakeLoader):
    """Synthetic raw tables in the source's shape, so model.load_games and daily.scheduled read them as they read real ones."""

    def __init__(self, g: pd.DataFrame):
        super().__init__(g)
        for s, gs in g.groupby("season"):
            start = (gs["date"] + pd.Timedelta(hours=25)).dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")      # evening tip-off, next day in UTC
            self.tables[f"cbb_games_{s}"] = pd.DataFrame({
                "id": gs["game_id"], "season": s, "seasonType": "regular", "startDate": start, "status": "final", "neutralSite": False,
                "conferenceGame": True, "gameType": "STD", "homeTeamId": gs["home"], "homeTeam": "h" + gs["home"].astype(str),
                "homeConference": "A", "awayTeamId": gs["away"], "awayTeam": "a" + gs["away"].astype(str), "awayConference": "A",
                "homeTeamEloStart": 1500.0, "awayTeamEloStart": 1500.0})
            box = self.tables[f"cbb_team_games_{s}"].merge(gs.rename(columns={"game_id": "gameId"}), on="gameId")
            home = box["teamId"].eq(box["home"])
            self.tables[f"cbb_team_games_{s}"] = box.assign(
                season=s, seasonType="regular", startDate=box["gameId"].map(dict(zip(gs["game_id"], start, strict=True))),
                opponentId=np.where(home, box["away"], box["home"]), conference="A", opponentConference="A", neutralSite=False, isHome=home,
                conferenceGame=True, gameType="STD", gameMinutes=40.0, teamStats_possessions=68.0, opponentStats_possessions=68.0,
                teamStats_points_total=np.where(home, box["home_pts"], box["away_pts"]),
                opponentStats_points_total=np.where(home, box["away_pts"], box["home_pts"])).drop(columns=list(gs.columns.difference(["season"])), errors="ignore")


def _setup():
    g = _games()
    ld = DayLoader(g)
    # One earlier season, so nothing can be fit on forecast seasons before this one: weights and spread are given.
    fitted = {"final": D.final_ratings(M.load_games(ld, [2021], "2021-10-31")), "beta": np.array([0.3, -0.2, -3.0, 5.0, 4.0, 0.05]),
              "returning_mean": 0.5, "fixed": {"margin_sd": 11.0, "home_margin": 3.0, "elo_per_point": 0.03, "elo_home": 3.0}}
    return g, ld, fitted


def test_a_days_forecasts_equal_the_walk_forwards():
    g, ld, fitted = _setup()
    loaded = M.load_games(ld, SEASONS, "2022-05-01")
    f = M.walk(loaded, [2022], M.PUBLISHED)
    x = F.build(f, ld, [2022], "2022-05-01", fitted["returning_mean"])
    ref = x.assign(point=F.apply(x, D.COLS, fitted["beta"])).set_index("game_id")
    for day in ("2021-11-10", "2021-11-11", "2021-11-24", "2021-12-09"):
        out, ratings = D.day(ld, day, 2022, [2021], fitted=fitted)
        j = out.set_index("game_id")
        assert len(j) == 6 and set(j.index) == set(ref.index[ref["date"] == day])
        assert np.allclose(j["margin"], ref.loc[j.index, "point"]) and np.allclose(j["ratings_margin"], ref.loc[j.index, "pred_margin"])
        assert np.allclose(j["total"], ref.loc[j.index, "pred_total"])
        assert ((j["margin_lo80"] < j["margin"]) & (j["margin"] < j["margin_hi80"])).all() and j["p_home"].between(0, 1).all()
        assert (j["p_home"] > 0.5).eq(j["margin"] > 0).all() and len(ratings) == 12


def test_a_days_forecasts_do_not_read_that_days_results_or_later_ones():
    g, ld, fitted = _setup()
    day = pd.Timestamp("2021-11-24")
    base, _ = D.day(ld, day, 2022, [2021], fitted=fitted)
    g2 = g.copy()
    hit = (g2["season"] == 2022) & (g2["date"] >= day)
    g2.loc[hit, ["home_pts", "away_pts"]] += np.random.default_rng(5).uniform(20, 60, (int(hit.sum()), 2))
    ld2 = DayLoader(g2)
    for name in ("cbb_team_games_2022", "cbb_player_games_2022"):          # the day's own box scores and every later one
        t = ld2.tables[name]
        rows = t["gameId"].isin(set(g2.loc[hit, "game_id"]))
        for c in [c for c in t.columns if c.startswith(("teamStats_three", "opponentStats_three", "teamStats_four", "opponentStats_four", "player_minutes"))]:
            t.loc[rows, c] = t.loc[rows, c] * 3 + 7
    scr, _ = D.day(ld2, day, 2022, [2021], fitted=fitted)
    for c in ("margin", "ratings_margin", "total", "p_home"):
        assert np.allclose(base[c], scr[c]), c
    later, later2 = (D.day(x, day + pd.Timedelta(days=3), 2022, [2021], fitted=fitted)[0] for x in (ld, ld2))
    assert not np.allclose(later["margin"], later2["margin"])                # and the changed results do move later days


def test_opening_day_is_the_prior_and_a_new_team_is_league_average():
    g, ld, fitted = _setup()
    out, ratings = D.day(ld, "2021-11-10", 2022, [2021], fitted=fitted)
    assert (ratings["games"] == 0).all() and np.allclose(out["ratings_margin"], out["prior_margin"])
    prev, prev_teams = fitted["final"]
    assert np.allclose(ratings.set_index("team").loc[prev_teams, "offense"], prev["off"])       # carry-over 1.0
    ld.tables["cbb_games_2022"].loc[ld.tables["cbb_games_2022"].index[0], "awayTeamId"] = 99     # a team with no last season
    out2, ratings2 = D.day(ld, "2021-11-10", 2022, [2021], fitted=fitted)
    new = ratings2.set_index("team").loc[99]
    assert new["offense"] == 0 and new["defense"] == 0 and len(out2) == 6


def _day_frame():
    g, ld, fitted = _setup()
    f, _ = D.day(ld, "2021-11-24", 2022, [2021], fitted=fitted)
    return f.assign(status="scheduled")              # the synthetic schedule is final; the job only writes scheduled games


def test_ledger_records_are_valid_and_only_for_games_not_yet_started(tmp_path, monkeypatch):
    """CBB-DAILY: one line per game, before tip-off, never twice; the whole write path in a scratch ledger."""
    from cbb.ratings import freeze as Z
    from sparvi import ledger
    f = _day_frame()
    now = f["start"].min() - pd.Timedelta(hours=6)
    recs = Z.records(f, 2022, "1.0.0", set(), now)
    assert len(recs) == len(f) == 6 and len({r["forecast_id"] for r in recs}) == 6
    r = recs[0]
    assert r["prob"] is None and r["lo80"] < r["point"] < r["hi80"] and 0 < r["detail"]["p_home"] < 1
    assert set(r["baselines"]) == {"ratings", "prior", "home"} and "elo" not in str(r)       # the source's rating is not handed out
    assert Z.records(f, 2022, "1.0.0", {x["forecast_id"] for x in recs}, now) == []            # never forecast again
    assert Z.records(f, 2022, "1.0.0", set(), f["start"].max()) == []                         # none after tip-off
    assert Z.records(f.assign(status="final"), 2022, "1.0.0", set(), now) == []
    gate = lambda m: {"model": m, "version": "1.0.0", "state": "experimental", "reason": "test"}  # noqa: E731
    monkeypatch.setattr(ledger, "_now", lambda: now.to_pydatetime())            # the writer's clock, on the synthetic morning
    assert ledger.append_forecasts("cbb", 2022, recs, root=tmp_path, gate=gate) == 6
    fc, res = ledger.read("cbb", 2022, tmp_path)
    assert len(fc) == 6 and res == [] and fc[0]["model_state"] == "experimental" and fc[0]["resolves_after"] > fc[0]["created_at"][:10]


def test_resolutions_score_final_games_void_cancelled_ones_and_wait_for_the_rest():
    from cbb.ratings import freeze as Z
    f = _day_frame()
    recs = Z.records(f, 2022, "1.0.0", set(), f["start"].min() - pd.Timedelta(hours=6))
    ids = [r["entity_id"] for r in recs]
    games = pd.DataFrame({"id": ids, "status": ["final", "final", "cancelled", "postponed", "scheduled", "final"],
                          "homePoints": [80, 61, None, None, None, None], "awayPoints": [71, 70, None, None, None, None]})
    after = recs[0]["resolves_after"]
    out = {r["forecast_id"]: r for r in Z.resolutions(recs, set(), games, after)}
    assert out[recs[0]["forecast_id"]] == {"forecast_id": recs[0]["forecast_id"], "status": "scored", "actual": 9, "sample": "80-71"}
    assert out[recs[1]["forecast_id"]]["actual"] == -9 and out[recs[2]["forecast_id"]]["status"] == "void"
    assert len(out) == 3                                                    # postponed, scheduled, and final without a score: wait
    assert Z.resolutions(recs, set(out), games, after) == []                # each resolves once
    early = pd.Timestamp(after) - pd.Timedelta(days=1)
    assert [r["status"] for r in Z.resolutions(recs, set(), games, early.date())] == ["void"]   # not scored before resolves_after
