"""College basketball team ratings: offense, defense and tempo, refit for every game date (CBB-0).

Per game: possessions are the mean of the two teams' recorded figures, scaled to 40 minutes; each side's efficiency is
its points per 100 possessions. Ratings are a ridge regression on one season's games before the forecast date:

    efficiency = league mean + offense - opposing defense + home court (none on a neutral court)
    tempo      = league mean + the two teams' tempos

The ridge pulls each team's rating toward a prior with the weight of `prior_weight` games. The prior is the team's
final rating last season times `carry`, and league average for a team with no last season. The three league terms
(mean efficiency, home court, mean tempo) are pulled toward last season's values with the weight of LEAGUE_WEIGHT games.

A forecast: expected possessions from the tempos, expected points per possession from offense against defense, and
from those the margin and the total. Home win probability is a normal distribution around the margin.

Every table is read through the point-in-time loader (backtest/asof.py). A forecast for a date uses games strictly
before that date (the date is the US game day: tip-off in UTC less six hours).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

LEAGUE_WEIGHT = 100.0        # games' worth of pull on the league mean, home court and mean tempo toward last season's
UTC_TO_GAME_DAY = pd.Timedelta(hours=6)
TEAM_COLS = ["gameId", "season", "seasonType", "startDate", "teamId", "opponentId", "conference", "opponentConference", "neutralSite",
             "isHome", "conferenceGame", "gameType", "gameMinutes", "teamStats_possessions", "opponentStats_possessions",
             "teamStats_points_total", "opponentStats_points_total"]
GAME_COLS = ["id", "status", "homeTeamEloStart", "awayTeamEloStart"]


@dataclass(frozen=True)
class Params:
    prior_weight: float = 5.0          # games' worth of pull toward the prior
    carry: float = 1.0                 # share of last season's final rating kept as this season's prior
    half_life: float | None = None     # recency weighting in days; None = every game counts the same


def game_day(start: pd.Series) -> pd.Series:
    return (pd.to_datetime(start, utc=True).dt.tz_localize(None) - UTC_TO_GAME_DAY).dt.normalize()


def load_games(loader, seasons: list[int], as_of) -> pd.DataFrame:
    """One row per counted game, from the home (or first-listed) side: both teams in a Division I conference, a final
    score, possessions recorded for both teams, at least 40 minutes."""
    t = pd.concat([loader.load(f"cbb_team_games_{s}", as_of, columns=TEAM_COLS) for s in seasons], ignore_index=True)
    g = pd.concat([loader.load(f"cbb_games_{s}", as_of, columns=GAME_COLS) for s in seasons], ignore_index=True).drop_duplicates("id")
    t = t[t["isHome"].fillna(False).astype(bool)].drop_duplicates("gameId").merge(g.rename(columns={"id": "gameId"}), on="gameId", how="left")
    t = t[t["status"].eq("final") & t["conference"].notna() & t["opponentConference"].notna()]
    t = t.dropna(subset=["teamStats_possessions", "opponentStats_possessions", "teamStats_points_total", "opponentStats_points_total", "gameMinutes"])
    poss = (t["teamStats_possessions"].astype(float) + t["opponentStats_possessions"].astype(float)) / 2
    out = pd.DataFrame({
        "game_id": t["gameId"].astype("int64"), "season": t["season"].astype(int), "date": game_day(t["startDate"]),
        "home": t["teamId"].astype("int64"), "away": t["opponentId"].astype("int64"),
        "neutral": t["neutralSite"].fillna(False).astype(bool), "season_type": t["seasonType"].astype(str),
        "conference_game": t["conferenceGame"].fillna(False).astype(bool), "site_type": t["gameType"].astype(str),
        "home_pts": t["teamStats_points_total"].astype(float), "away_pts": t["opponentStats_points_total"].astype(float),
        "minutes": t["gameMinutes"].astype(float), "poss": poss,
        "elo_diff": (t["homeTeamEloStart"] - t["awayTeamEloStart"]).astype(float)})
    out = out[(out["poss"] > 40) & (out["minutes"] >= 40)]
    out["tempo"] = out["poss"] * 40 / out["minutes"]
    out["home_eff"], out["away_eff"] = 100 * out["home_pts"] / out["poss"], 100 * out["away_pts"] / out["poss"]
    out["margin"], out["total"] = out["home_pts"] - out["away_pts"], out["home_pts"] + out["away_pts"]
    return out.sort_values(["date", "game_id"]).reset_index(drop=True)


def game_type(f: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(f["season_type"].eq("postseason"), "national postseason",
                              np.where(~f["conference_game"], "non-conference",
                                       np.where(f["site_type"].eq("TRNMNT"), "conference tournament", "conference"))), index=f.index)


def _normal(C: np.ndarray, V: np.ndarray, w: np.ndarray, y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """X'WX and X'Wy for a design whose row r has values V[r] in columns C[r] (never built as a matrix)."""
    A, b = np.zeros((p, p)), np.zeros(p)
    for i in range(C.shape[1]):
        np.add.at(b, C[:, i], w * V[:, i] * y)
        for j in range(C.shape[1]):
            np.add.at(A, (C[:, i], C[:, j]), w * V[:, i] * V[:, j])
    return A, b


def fit_ratings(g: pd.DataFrame, on: pd.Timestamp, teams: np.ndarray, prior: dict, p: Params) -> dict:
    """Ratings from the games of `g` (one season) strictly before `on`. `prior` holds 'off', 'def', 'tempo' arrays aligned
    to `teams` (already carried over) and the league terms 'mu', 'hca', 'mu_t'."""
    n = len(teams)
    g = g[g["date"] < on]
    m = len(g)
    idx = {t: i for i, t in enumerate(teams)}
    h, a = g["home"].map(idx).to_numpy(int), g["away"].map(idx).to_numpy(int)
    w = np.ones(m) if p.half_life is None else 0.5 ** ((on - g["date"]).dt.days.to_numpy() / p.half_life)
    hc, one, zero = np.where(g["neutral"].to_numpy(), 0.0, 1.0), np.ones(m), np.zeros(m, int)
    # efficiency: columns [mean, home court, offense (n), defense (n)]; two rows a game
    C = np.r_[np.c_[zero, zero + 1, 2 + h, 2 + n + a], np.c_[zero, zero + 1, 2 + a, 2 + n + h]]
    V = np.r_[np.c_[one, hc, one, -one], np.c_[one, -hc, one, -one]]
    A, b = _normal(C, V, np.r_[w, w], np.r_[g["home_eff"].to_numpy(), g["away_eff"].to_numpy()], 2 + 2 * n)
    pen = np.r_[LEAGUE_WEIGHT, LEAGUE_WEIGHT, np.full(2 * n, p.prior_weight)]
    beta = np.linalg.solve(A + np.diag(pen), b + pen * np.r_[prior["mu"], prior["hca"], prior["off"], prior["def"]])
    # tempo: columns [mean, team (n)]
    A, b = _normal(np.c_[zero, 1 + h, 1 + a], np.c_[one, one, one], w, g["tempo"].to_numpy(), 1 + n)
    pen = np.r_[LEAGUE_WEIGHT, np.full(n, p.prior_weight)]
    tb = np.linalg.solve(A + np.diag(pen), b + pen * np.r_[prior["mu_t"], prior["tempo"]])
    return {"mu": float(beta[0]), "hca": float(beta[1]), "off": beta[2:2 + n], "def": beta[2 + n:], "mu_t": float(tb[0]), "tempo": tb[1:]}


def forecast(r: dict, teams: np.ndarray, d: pd.DataFrame) -> pd.DataFrame:
    """Margin (home less away), total and possessions for the games in `d` from ratings `r`."""
    idx = {t: i for i, t in enumerate(teams)}
    h, a = d["home"].map(idx).to_numpy(int), d["away"].map(idx).to_numpy(int)
    hc = np.where(d["neutral"].to_numpy(), 0.0, 1.0)
    poss = r["mu_t"] + r["tempo"][h] + r["tempo"][a]
    eh = r["mu"] + r["off"][h] - r["def"][a] + r["hca"] * hc
    ea = r["mu"] + r["off"][a] - r["def"][h] - r["hca"] * hc
    return pd.DataFrame({"game_id": d["game_id"].to_numpy(), "pred_margin": poss * (eh - ea) / 100, "pred_total": poss * (eh + ea) / 100,
                         "pred_poss": poss})


def season_prior(prev: dict | None, prev_teams: np.ndarray | None, teams: np.ndarray, carry: float, league: dict) -> dict:
    """This season's prior: last season's final ratings times `carry`; zero (league average) for a team without one."""
    out = {"off": np.zeros(len(teams)), "def": np.zeros(len(teams)), "tempo": np.zeros(len(teams)), **league}
    if prev is not None:
        pos = {t: i for i, t in enumerate(prev_teams)}
        for i, t in enumerate(teams):
            if t in pos:
                j = pos[t]
                out["off"][i], out["def"][i], out["tempo"][i] = carry * prev["off"][j], carry * prev["def"][j], carry * prev["tempo"][j]
        out.update({"mu": prev["mu"], "hca": prev["hca"], "mu_t": prev["mu_t"]})
    return out


def teams_of(gs: pd.DataFrame) -> np.ndarray:
    return np.unique(np.r_[gs["home"].to_numpy(), gs["away"].to_numpy()])


def walk(g: pd.DataFrame, seasons: list[int], p: Params) -> pd.DataFrame:
    """Walk-forward forecasts for every game of `seasons`, each from that season's games before its date.
    `g` must hold the season before the first forecast season (it seeds the prior) and every season in between.
    The first season in `g` has no prior of its own: its league terms start from its own averages (it is never forecast)."""
    first = g[g["season"] == g["season"].min()]
    assert g["season"].min() not in seasons, "the first loaded season only seeds the next one's prior"
    league = {"mu": float(np.r_[first["home_eff"], first["away_eff"]].mean()), "hca": 0.0, "mu_t": float(first["tempo"].mean()) }
    out, prev, prev_teams = [], None, None
    for s in sorted(g["season"].unique()):
        gs = g[g["season"] == s]
        teams = teams_of(gs)
        prior = season_prior(prev, prev_teams, teams, p.carry, league)
        if s in seasons:
            for day, d in gs.groupby("date"):
                f = forecast(fit_ratings(gs, day, teams, prior, p), teams, d)
                f["prior_margin"] = forecast(prior, teams, d)["pred_margin"].to_numpy()
                out.append(f)
        prev, prev_teams = fit_ratings(gs, gs["date"].max() + pd.Timedelta(days=1), teams, prior, p), teams
    return g.merge(pd.concat(out, ignore_index=True), on="game_id")


def win_probability(margin, sd: float) -> np.ndarray:
    """P(home wins) from a normal distribution with spread `sd` around the forecast margin."""
    z = np.asarray(margin, float) / (sd * math.sqrt(2))
    return 0.5 * (1 + np.vectorize(math.erf)(z))
