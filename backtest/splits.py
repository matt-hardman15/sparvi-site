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


def season_folds(first_test: int, last_test: int, train_window: int | None, decision_month_day: str = "03-01",
                 predictor_lag: int = 1) -> list[Fold]:
    """One fold per test season. Training uses seasons up to test - predictor_lag - 1 (the predictor
    season itself is graded, not trained on), either all of them (train_window None, expanding) or the
    last `train_window`. Decisions happen on decision_month_day of the test season (before it starts)."""
    folds = []
    for t in range(first_test, last_test + 1):
        last_train = t - predictor_lag - 1
        first_train = last_train - train_window + 1 if train_window else None
        train = tuple(range(first_train, last_train + 1)) if first_train else (last_train,)
        folds.append(Fold(t, train, pd.Timestamp(f"{last_train}-12-31 23:59:59"), pd.Timestamp(f"{t}-{decision_month_day}"),
                          pd.Timestamp(f"{t}-12-31 23:59:59")))
    return folds
