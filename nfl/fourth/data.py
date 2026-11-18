"""Plays for the fourth-down models (F4), read through the point-in-time loader only.

    plays = load_plays(loader, seasons=range(2016, 2020), as_of="2020-09-01")

One row per nflverse play (pbp_{season} joined to state/pbp_state_{season}), from the offense's point of view, with:
- game state: score margin, time, half, down, distance, field position, timeouts, home, who receives the 2nd-half
  kickoff;
- team strength before the game (`off_rating`, `def_rating`, `strength`): season-to-date point margin per game from
  earlier games only, shrunk toward half of last season's margin per game (K_PRIOR games of weight);
- labels: `win` (offense won; ties are NaN), `converted` (4th-down attempts), `fg_made`, `punt_net`.

Inputs are raw game facts only. nflverse's own model columns (wp, vegas_wp, ep, epa, fg_prob, ...) are fitted on every
season, so nothing here reads them, and betting lines are never an input (market data is a benchmark only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FIRST_SEASON = 2016          # the NFL window in sparvi.toml; training starts here
K_PRIOR = 4                  # games of weight on last season's margin, shrunk by PRIOR_SHRINK
PRIOR_SHRINK = 0.5

PBP = ["game_id", "play_id", "season", "week", "season_type", "game_date", "home_team", "away_team", "posteam",
       "defteam", "down", "ydstogo", "yardline_100", "play_type", "two_point_attempt", "penalty", "home_score",
       "away_score", "qb_kneel", "qb_spike"]
STATE = ["game_id", "play_id", "game_seconds_remaining", "half_seconds_remaining", "game_half", "score_differential",
         "posteam_timeouts_remaining", "defteam_timeouts_remaining", "roof", "temp", "wind", "kick_distance",
         "field_goal_result", "punt_blocked", "return_yards", "touchback", "fourth_down_converted",
         "fourth_down_failed", "kickoff_attempt", "aborted_play", "first_down"]


def _season(loader, season: int, as_of) -> pd.DataFrame:
    p = loader.load(f"pbp_{season}", as_of, columns=PBP)
    s = loader.load(f"pbp_state_{season}", as_of, columns=STATE)
    return p.merge(s, on=["game_id", "play_id"], how="inner", validate="one_to_one")


def team_games(plays: pd.DataFrame) -> pd.DataFrame:
    """One row per team x game (regular season and playoffs): date and final margin."""
    g = plays.groupby("game_id", as_index=False).agg(season=("season", "first"), game_date=("game_date", "first"),
                                                     season_type=("season_type", "first"), home=("home_team", "first"),
                                                     away=("away_team", "first"), hs=("home_score", "max"),
                                                     as_=("away_score", "max"))
    home = g.assign(team=g["home"], margin=g["hs"] - g["as_"])
    away = g.assign(team=g["away"], margin=g["as_"] - g["hs"])
    return pd.concat([home, away])[["game_id", "season", "season_type", "game_date", "team", "margin"]]


def ratings(tg: pd.DataFrame) -> pd.DataFrame:
    """Each team's strength entering each game: (earlier margins this season + K_PRIOR x shrunk last-season margin per
    game) / (earlier games + K_PRIOR). Uses only games before the one being rated."""
    tg = tg.sort_values(["team", "season", "game_date", "game_id"]).copy()
    reg = tg[tg["season_type"] == "REG"]
    last = reg.groupby(["team", "season"])["margin"].mean().rename("prev").reset_index()
    last["season"] += 1
    tg = tg.merge(last, on=["team", "season"], how="left")
    tg["prev"] = tg["prev"].fillna(0.0) * PRIOR_SHRINK
    grp = tg.groupby(["team", "season"])["margin"]
    tg["sum_before"] = grp.cumsum() - tg["margin"]
    tg["n_before"] = grp.cumcount()
    tg["rating"] = (tg["sum_before"] + K_PRIOR * tg["prev"]) / (tg["n_before"] + K_PRIOR)
    return tg[["game_id", "team", "rating"]]


def load_plays(loader, seasons, as_of, prior_from=None) -> pd.DataFrame:
    """Plays of `seasons` as of `as_of`, with ratings (last season's margins come from season - 1 when it's loaded:
    pass prior_from to load it too)."""
    seasons = list(seasons)
    extra = [s - 1 for s in seasons if s - 1 >= FIRST_SEASON and s - 1 not in seasons]
    raw = pd.concat([_season(loader, s, as_of) for s in sorted(set(seasons) | set(extra))], ignore_index=True)
    rat = ratings(team_games(raw))
    d = raw[raw["season"].isin(seasons)].copy()
    d = d.merge(rat.rename(columns={"team": "posteam", "rating": "off_rating"}), on=["game_id", "posteam"], how="left")
    d = d.merge(rat.rename(columns={"team": "defteam", "rating": "def_rating"}), on=["game_id", "defteam"], how="left")
    d["strength"] = d["off_rating"] - d["def_rating"]

    # Who receives the second-half kickoff: the team that kicked the opening kickoff (posteam on a kickoff is the
    # receiving team).
    ko = d[d["kickoff_attempt"] == 1].sort_values(["game_id", "game_seconds_remaining"], ascending=[True, False])
    first = ko.drop_duplicates("game_id").set_index("game_id")
    kicker = first["defteam"]
    d["recv_2h"] = (d["posteam"] == d["game_id"].map(kicker)).astype(float)

    pos_final = np.where(d["posteam"] == d["home_team"], d["home_score"], d["away_score"])
    def_final = np.where(d["posteam"] == d["home_team"], d["away_score"], d["home_score"])
    d["win"] = np.where(pos_final > def_final, 1.0, np.where(pos_final < def_final, 0.0, np.nan))
    d["home"] = (d["posteam"] == d["home_team"]).astype(float)
    d["half2"] = (d["game_half"] == "Half2").astype(float)
    d["ot"] = (d["game_half"] == "Overtime").astype(float)
    d["indoors"] = d["roof"].isin(["dome", "closed"]).astype(float)
    d["wind"] = d["wind"].where(d["indoors"] == 0, 0.0)
    d["temp"] = d["temp"].where(d["indoors"] == 0, 70.0)

    go = (d["down"] == 4) & d["play_type"].isin(["pass", "run"]) & (d["qb_kneel"] != 1) & (d["qb_spike"] != 1)
    d["converted"] = np.where(go & (d["fourth_down_converted"] == 1), 1.0,
                              np.where(go & (d["fourth_down_failed"] == 1), 0.0, np.nan))
    # Third downs, the same kind of play, as extra training rows for the conversion model (is_4th tells them apart).
    third = (d["down"] == 3) & d["play_type"].isin(["pass", "run"]) & (d["qb_kneel"] != 1) & (d["qb_spike"] != 1)         & (d["penalty"] != 1)
    d["converted_3rd"] = np.where(third, (d["first_down"] == 1).astype(float), np.nan)
    fg = d["play_type"] == "field_goal"
    d["fg_made"] = np.where(fg & (d["field_goal_result"] == "made"), 1.0,
                            np.where(fg & d["field_goal_result"].isin(["missed", "blocked"]), 0.0, np.nan))
    punt = (d["play_type"] == "punt") & (d["punt_blocked"] != 1)
    net = np.where(d["touchback"] == 1, d["yardline_100"] - 20, d["kick_distance"] - d["return_yards"].fillna(0))
    d["punt_net"] = np.where(punt & d["kick_distance"].notna(), net, np.nan)
    return d.reset_index(drop=True)


def scrimmage(d: pd.DataFrame) -> pd.DataFrame:
    """Plays a win-probability model is fit and scored on: downs 1-4 from scrimmage, decided games (no ties), clock
    known, no two-point tries or aborted snaps."""
    keep = (d["down"].between(1, 4) & d["posteam"].notna() & d["win"].notna() & d["game_seconds_remaining"].notna()
            & (d["two_point_attempt"] != 1) & d["yardline_100"].notna() & d["strength"].notna())
    return d[keep].reset_index(drop=True)
