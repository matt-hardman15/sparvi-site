"""Walk-forward splits (PLAN-backtest §3): train on the past, predict the next period, roll forward.
Never shuffled. The harness, not the model, sets every date a model may see.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    test_season: int
    train_seasons: tuple[int, ...]
    train_until: pd.Timestamp          # last moment of training data
    decision_date: pd.Timestamp        # when the prediction is made (everything before it is fair game)
    resolve_after: pd.Timestamp        # when the outcome is known


def _on(season: int, month_day: str) -> pd.Timestamp:
    """A date in season `season`'s calendar year, or with a "+1:" prefix the next one (seasons that end in February),
    or with "-1:" the one before (a season named for the year it ends, as college basketball's is, starts in it)."""
    if month_day.startswith(("+1:", "-1:")):
        return pd.Timestamp(f"{season + int(month_day[:2])}-{month_day[3:]}")
    return pd.Timestamp(f"{season}-{month_day}")


def season_folds(first_test: int, last_test: int, train_window: int | None, decision_month_day: str = "03-01",
                 predictor_lag: int = 1, train_until_month_day: str | None = None,
                 resolve_month_day: str | None = None, first_train: int | None = None) -> list[Fold]:
    """One fold per test season. Training uses seasons up to test - predictor_lag - 1 (the predictor
    season itself is graded, not trained on), either all of them from `first_train` on (train_window None, expanding)
    or the last `train_window`. Decisions happen on decision_month_day of the test season (before it starts).
    train_until_month_day moves the training cutoff from Dec 31 of the last training season to that day of the test
    season, for models whose training labels finish with the bowls (e.g. a transfer's second season).
    resolve_month_day moves the outcome date from Dec 31 of the test season. Any of the three may carry a "+1:" prefix
    for the next calendar year (an NFL season's last games are in February)."""
    folds = []
    for t in range(first_test, last_test + 1):
        last_train = t - predictor_lag - 1
        if train_window:
            start = last_train - train_window + 1
        elif first_train is not None:
            start = first_train
        else:   # used to fall back silently to one season, contrary to this docstring (no card relied on it)
            raise ValueError("train_window None (expanding) needs first_train")
        train = tuple(range(start, last_train + 1))
        until = _on(t, train_until_month_day) if train_until_month_day else pd.Timestamp(f"{last_train}-12-31 23:59:59")
        resolve = _on(t, resolve_month_day) if resolve_month_day else pd.Timestamp(f"{t}-12-31 23:59:59")
        folds.append(Fold(t, train, until, _on(t, decision_month_day), resolve))
    return folds
