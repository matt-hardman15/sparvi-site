"""Forecasts for one day's college basketball games from the published team rating (CBB-DAILY).

    forecasts, ratings = day(loader, "2026-12-01")

For a game day D (the US game day), everything comes from what was known before D:
  - the ratings: this season's counted games before D, with last season's final ratings as the prior
  - what is fit on other seasons (the margin's spread, the second stage's weights, the typical returning share): the
    forecast seasons before the current one, exactly as the registered backtest fits them for its test season
  - the second stage's inputs for each team: its games before D

A game is forecast when it is scheduled for D between two teams in a Division I conference. A team with no counted game
yet this season is forecast from its prior alone. Nothing here writes anywhere: the caller decides what to do with the rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cbb.ratings import features as F
from cbb.ratings import model as M
from cbb.ratings import scoring as S

Z80 = 1.2815515655446004
COLS = [c for name in F.KEPT for c in F.GROUPS[name]]
SCHEDULE_COLS = ["id", "season", "seasonType", "startDate", "status", "neutralSite", "conferenceGame", "gameType",
                 "homeTeamId", "homeTeam", "homeConference", "awayTeamId", "awayTeam", "awayConference"]


def scheduled(loader, season: int, on: pd.Timestamp, as_of) -> pd.DataFrame:
    """The games of `season` scheduled for game day `on` between two teams in a Division I conference, as model rows
    (no result columns)."""
    g = loader.load(f"cbb_games_{season}", as_of, columns=SCHEDULE_COLS)
    g = g[M.game_day(g["startDate"]).eq(on) & g["status"].isin(["scheduled", "in_progress", "final"])
          & g["homeConference"].notna() & g["awayConference"].notna()].drop_duplicates("id")
    return pd.DataFrame({
        "game_id": g["id"].astype("int64"), "season": int(season), "date": on, "start": pd.to_datetime(g["startDate"], utc=True),
        "home": g["homeTeamId"].astype("int64"), "away": g["awayTeamId"].astype("int64"),
        "home_name": g["homeTeam"].astype(str), "away_name": g["awayTeam"].astype(str),
        "neutral": g["neutralSite"].fillna(False).astype(bool), "season_type": g["seasonType"].astype(str),
        "conference_game": g["conferenceGame"].fillna(False).astype(bool), "site_type": g["gameType"].astype(str),
    }).sort_values(["start", "game_id"]).reset_index(drop=True)


def final_ratings(g: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """The last loaded season's final ratings and their teams, reached the way model.walk reaches them: each season's
    prior is the season before's final ratings, and the first season starts from its own averages."""
    first = g[g["season"] == g["season"].min()]
    league = {"mu": float(np.r_[first["home_eff"], first["away_eff"]].mean()), "hca": 0.0, "mu_t": float(first["tempo"].mean())}
    prev, prev_teams = None, None
    for s in sorted(g["season"].unique()):
        gs = g[g["season"] == s]
        teams = M.teams_of(gs)
        prior = M.season_prior(prev, prev_teams, teams, M.PUBLISHED.carry, league)
        prev, prev_teams = M.fit_ratings(gs, gs["date"].max() + pd.Timedelta(days=1), teams, prior, M.PUBLISHED), teams
    return prev, prev_teams


def walk_season(played: pd.DataFrame, teams: np.ndarray, prior: dict) -> pd.DataFrame:
    """Each played game of the current season with the forecast it had that morning (as model.walk makes them)."""
    out = []
    for d, x in played.groupby("date"):
        f = M.forecast(M.fit_ratings(played, d, teams, prior, M.PUBLISHED), teams, x)
        f["prior_margin"] = M.forecast(prior, teams, x)["pred_margin"].to_numpy()
        out.append(f)
    return played.merge(pd.concat(out, ignore_index=True), on="game_id")


def fit_other_seasons(loader, seasons: list[int], as_of) -> dict:
    """What a season takes from the seasons before it (`seasons`: the first only seeds the next one's prior): the same
    quantities, fit the same way, as models/cbb/team_rating/backtest.py `fit`. Also the last season's final ratings."""
    g = M.load_games(loader, seasons, as_of)
    f = M.walk(g, seasons[1:], M.PUBLISHED)
    ret_mean = F.raw_returning(f, loader, seasons[1:], as_of)
    x = F.build(f, loader, seasons[1:], as_of, ret_mean)
    return {"fixed": S.fixed_from(f), "returning_mean": ret_mean, "beta": F.fit(x, COLS), "total_mean": float(f["total"].mean()),
            "final": final_ratings(g)}


def day(loader, on, season: int, earlier: list[int], as_of=None, fitted: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(forecasts for the games of game day `on`, every team's ratings that morning). `earlier`: the seasons before
    `season`, oldest first. `as_of`: what the loader may serve. The default is two days after `on`: the loader counts a
    finished game as known the day after its tip-off in UTC, which for an evening game is two mornings later, while
    the daily pull has last night's box scores, and today's schedule, that morning. Whatever is served, only games of
    game days before `on` are used (as in the registered backtest); tests/test_cbb_daily.py changes `on`'s own
    results and every later one and requires the same forecasts."""
    on = pd.Timestamp(on).normalize()
    as_of = on + pd.Timedelta(days=2) if as_of is None else pd.Timestamp(as_of)
    fitted = fitted or fit_other_seasons(loader, earlier, pd.Timestamp(f"{season - 1}-10-31"))
    played = M.load_games(loader, [season], as_of)
    played = played[played["date"] < on]
    today = scheduled(loader, season, on, as_of)
    prev, prev_teams = fitted["final"]
    teams = np.unique(np.r_[M.teams_of(played), today["home"].to_numpy(), today["away"].to_numpy()]).astype("int64")
    prior = M.season_prior(prev, prev_teams, teams, M.PUBLISHED.carry, {})
    r = M.fit_ratings(played, on, teams, prior, M.PUBLISHED)
    ratings = pd.DataFrame({"team": teams, "offense": r["off"], "defense": r["def"], "net": r["off"] + r["def"], "tempo": r["mu_t"] + r["tempo"],
                            "games": pd.Series(np.r_[played["home"], played["away"]]).value_counts().reindex(teams).fillna(0).astype(int).to_numpy()})
    if today.empty:
        return today, ratings
    # Each played game's own walk-forward forecast (the second stage's form input is the ratings' recent error), then today's.
    past = walk_season(played, teams, prior) if len(played) else played.assign(pred_margin=[], pred_total=[], prior_margin=[])
    now = today.merge(M.forecast(r, teams, today), on="game_id")
    now["prior_margin"] = M.forecast(prior, teams, today)["pred_margin"].to_numpy()
    both = pd.concat([past, now.assign(margin=np.nan, total=np.nan)], ignore_index=True)
    x = F.build(both, loader, [season], as_of, fitted["returning_mean"])
    x = x[x["game_id"].isin(set(today["game_id"])) & x["date"].eq(on)].set_index("game_id").reindex(now["game_id"])
    point = F.apply(x.reset_index(), COLS, fitted["beta"])
    sd = fitted["fixed"]["margin_sd"]
    out = now.assign(ratings_margin=now["pred_margin"].to_numpy(), margin=point, margin_lo80=point - Z80 * sd, margin_hi80=point + Z80 * sd,
                     p_home=M.win_probability(point, sd), total=now["pred_total"].to_numpy(), game_type=M.game_type(now).to_numpy())
    return out.drop(columns=["pred_margin", "pred_total"]), ratings
