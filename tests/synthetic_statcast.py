"""Synthetic Triple-A Statcast for testing the dbt pipeline in CI.

Real pitch-level data can't be committed or published (CLAUDE.md rule 1), so CI builds the
pipeline against this instead. It has the same columns and plausible values, enough players to
fill the qualified percentile pools and the trailing window, and nothing real in it.

    write_fixture(out_dir, season)  ->  statcast_aaa_{season-1,season}.parquet,
                                        statcast_mlb_{season-2..season}.parquet,
                                        milb_players_{season}.parquet,
                                        pitch_type_aaa_{season}.csv
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

TEAMS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
PER_TEAM = 12  # hitters and pitchers per team
PA_PER_HALF = 36
PITCH_TYPES = np.array(["FF", "SI", "SL", "CH", "CU"])


def _players() -> pd.DataFrame:
    rows = []
    for t, team in enumerate(TEAMS):
        for i in range(PER_TEAM):
            rows.append({"player_id": 100000 + t * 100 + i, "name": f"Hitter {team}{i}", "team": team, "role": "H"})
            rows.append({"player_id": 200000 + t * 100 + i, "name": f"Pitcher {team}{i}", "team": team, "role": "P"})
    return pd.DataFrame(rows)


def _season(season: int, rng: np.random.Generator, pk_offset: int = 0) -> pd.DataFrame:
    start = date(season, 4, 1)
    days = [start + timedelta(d) for d in range(150)]
    # Four games a day: every team plays daily.
    games = []
    for d in days:
        order = rng.permutation(len(TEAMS))
        for g in range(len(TEAMS) // 2):
            games.append((d, TEAMS[order[2 * g]], TEAMS[order[2 * g + 1]]))
    g = pd.DataFrame(games, columns=["game_date", "home_team", "away_team"])
    g["game_pk"] = season * 10000 + pk_offset + np.arange(len(g))

    # Plate appearances: each half-inning side gets PA_PER_HALF PAs.
    pa = g.loc[g.index.repeat(2 * PA_PER_HALF)].reset_index(drop=True)
    pa["at_bat_number"] = pa.groupby("game_pk").cumcount() + 1
    pa["inning_topbot"] = np.where(pa["at_bat_number"] <= PA_PER_HALF, "Top", "Bot")
    bat_team = np.where(pa["inning_topbot"].eq("Top"), pa["away_team"], pa["home_team"])
    fld_team = np.where(pa["inning_topbot"].eq("Top"), pa["home_team"], pa["away_team"])
    tidx = {t: i for i, t in enumerate(TEAMS)}
    bt = np.vectorize(tidx.get)(bat_team)
    ft = np.vectorize(tidx.get)(fld_team)
    pa["batter"] = 100000 + bt * 100 + (pa["at_bat_number"] - 1) % PER_TEAM
    # Every team plays daily, so a five-man rotation runs by day: pitchers 0-4 start and face the
    # first 18 batters; relievers 5-11 take the rest. That gives SP and RP roles to test against.
    day = (pd.to_datetime(pa["game_date"]) - pd.Timestamp(start)).dt.days.to_numpy()
    k = (pa["at_bat_number"].to_numpy() - 1) % PA_PER_HALF
    slot = np.where(k < 18, day % 5, 5 + rng.integers(0, PER_TEAM - 5, len(pa)))
    pa["pitcher"] = 200000 + ft * 100 + slot
    pa["inning"] = (pa["at_bat_number"] - 1) % PA_PER_HALF // 4 + 1

    # Pitches: 1-6 per PA, the last one ends it.
    n = rng.integers(1, 7, len(pa))
    p = pa.loc[pa.index.repeat(n)].reset_index(drop=True)
    p["pitch_number"] = p.groupby(["game_pk", "at_bat_number"]).cumcount() + 1
    last = p["pitch_number"].to_numpy() == np.repeat(n, n)
    N = len(p)

    mid = rng.choice(["ball", "called_strike", "swinging_strike", "foul"], N, p=[0.38, 0.17, 0.12, 0.33])
    end_event = rng.choice(["strikeout", "walk", "field_out", "single", "double", "home_run"], N,
                           p=[0.22, 0.09, 0.45, 0.15, 0.06, 0.03])
    end_desc = np.select([end_event == "strikeout", end_event == "walk"], ["swinging_strike", "ball"], "hit_into_play")
    p["description"] = np.where(last, end_desc, mid)
    p["events"] = np.where(last, end_event, None)
    p["balls"] = rng.integers(0, 4, N)
    p["strikes"] = rng.integers(0, 3, N)
    p["zone"] = rng.integers(1, 15, N)

    bip = p["description"].eq("hit_into_play").to_numpy()
    p["launch_speed"] = np.where(bip, rng.normal(88, 12, N), np.nan)
    p["launch_angle"] = np.where(bip, rng.normal(12, 25, N), np.nan)
    p["estimated_woba_using_speedangle"] = np.where(bip, rng.uniform(0, 1.2, N), np.nan)
    woba = {"strikeout": 0.0, "walk": 0.69, "field_out": 0.0, "single": 0.88, "double": 1.25, "home_run": 2.0}
    p["woba_value"] = np.where(last, pd.Series(end_event).map(woba).to_numpy(), np.nan)
    p["woba_denom"] = np.where(last, 1.0, np.nan)

    # Each pitcher has a fixed base velocity; pitch type varies per pitch.
    base_velo = 88 + (p["pitcher"].to_numpy() % 97) / 97 * 10
    p["pitch_type"] = rng.choice(PITCH_TYPES, N, p=[0.28, 0.27, 0.20, 0.15, 0.10])  # FF and SI nearly tied on purpose
    offset = pd.Series(p["pitch_type"]).map({"FF": 0, "SI": -7, "SL": -8, "CH": -9, "CU": -13}).to_numpy()
    p["release_speed"] = base_velo + offset + rng.normal(0, 0.8, N)
    p["release_spin_rate"] = rng.normal(2300, 250, N)
    p["release_extension"] = rng.normal(6.3, 0.3, N)
    p["pfx_x"] = rng.normal(0, 0.7, N)
    p["pfx_z"] = rng.normal(0.8, 0.6, N)
    p["p_throws"] = np.where(p["pitcher"] % 3 == 0, "L", "R")
    p["stand"] = np.where(p["batter"] % 4 == 0, "L", "R")
    p["game_year"] = season
    p["game_type"] = "R"
    p["game_date"] = pd.to_datetime(p["game_date"])
    return p


def write_fixture(out: Path, season: int, seed: int = 0) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for s in (season - 1, season):
        _season(s, rng).to_parquet(out / f"statcast_aaa_{s}.parquet", index=False)
    for s in (season - 2, season - 1, season):
        _season(s, rng, pk_offset=5000).to_parquet(out / f"statcast_mlb_{s}.parquet", index=False)

    pl = _players()
    pl["birth_date"] = [f"{season - 22 - i % 10}-05-01" for i in range(len(pl))]
    pl["position"] = np.where(pl["role"].eq("P"), "P", "CF")
    pl["bats"], pl["throws"], pl["level"] = "R", "R", "AAA"
    pl["team_id"], pl["parent_org_id"] = 1, 1
    pl.drop(columns=["team", "role"]).to_parquet(out / f"milb_players_{season}.parquet", index=False)

    pitchers = pl.loc[pl["position"].eq("P"), "player_id"]
    sq = pd.DataFrame([(pid, pt) for pid in pitchers for pt in PITCH_TYPES], columns=["pitcher", "pitch_type"])
    sq["stuff_plus"] = rng.normal(100, 10, len(sq)).round(1)
    sq.to_csv(out / f"pitch_type_aaa_{season}.csv", index=False)
