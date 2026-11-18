"""Shared fold logic for the four F4 backtest modules (models/nfl/*/backtest.py).

Fold (season t): every model trains on seasons 2016..t-1 as of fold.train_until (Sept 1 of t, after the previous
Super Bowl), and is scored on every qualifying play of season t. Plays are predicted from their own pre-snap state;
team ratings use earlier games only (data.ratings), so a play's row holds nothing that happened at or after its snap
except its own outcome label. Rows are plays; the bootstrap resamples games.
"""
from __future__ import annotations

import pandas as pd

from nfl.fourth.data import FIRST_SEASON, load_plays


def train(loader, fold) -> pd.DataFrame:
    return load_plays(loader, range(FIRST_SEASON, fold.test_season), fold.train_until)


def test(loader, fold) -> pd.DataFrame:
    return load_plays(loader, [fold.test_season], fold.decision_date)


def ids(d: pd.DataFrame) -> pd.Series:
    return d["game_id"] + ":" + d["play_id"].astype(int).astype(str)


def rows(d: pd.DataFrame, point, **extra) -> pd.DataFrame:
    return pd.DataFrame({"entity_id": ids(d), "point": point, "game_id": d["game_id"].to_numpy(), **extra})


def outcome(d: pd.DataFrame, label: str) -> pd.DataFrame:
    return pd.DataFrame({"entity_id": ids(d), "y": d[label].to_numpy(float)})


def load_plays_resolved(loader, fold) -> pd.DataFrame:
    """Season t's plays as of resolve_after, for the outcome labels."""
    return load_plays(loader, [fold.test_season], fold.resolve_after)
