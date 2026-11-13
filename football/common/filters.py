"""Play filters shared by every football view.

The dbt macro `nfl_garbage_time()` (football/dbt/macros/football_metrics.sql) applies the same rule
in SQL; tests/test_football.py checks the two agree play for play, and that the thresholds here
match the dbt vars.

NFL garbage time: a 4th-quarter play where the offense's win probability is below 10% or above 90%.
Plays with no win probability are kept.

College garbage time (`cfb_garbage_time()`): the score margin before the play is more than 38 points
in the 2nd quarter, 28 in the 3rd, or 22 in the 4th (either team ahead). The 1st quarter and
overtime never count.
"""
from __future__ import annotations

import numpy as np

NFL_GARBAGE_QTR = 4
NFL_GARBAGE_WP_LO = 0.10
NFL_GARBAGE_WP_HI = 0.90


def nfl_garbage_time(qtr, wp) -> np.ndarray:
    """Boolean array: True where a play is in NFL garbage time."""
    qtr = np.asarray(qtr, dtype=float)
    wp = np.asarray(wp, dtype=float)
    with np.errstate(invalid="ignore"):
        lopsided = (wp < NFL_GARBAGE_WP_LO) | (wp > NFL_GARBAGE_WP_HI)
    return (qtr == NFL_GARBAGE_QTR) & lopsided & ~np.isnan(wp)


CFB_GARBAGE_MARGIN = {2: 38, 3: 28, 4: 22}     # quarter -> margin above which a play is garbage time


def cfb_garbage_time(period, margin) -> np.ndarray:
    """Boolean array: True where a play is in college garbage time."""
    period = np.asarray(period, dtype=float)
    margin = np.abs(np.asarray(margin, dtype=float))
    out = np.zeros(period.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        for q, m in CFB_GARBAGE_MARGIN.items():
            out |= (period == q) & (margin > m)
    return out
