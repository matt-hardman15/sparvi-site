"""Play filters shared by every football view.

The dbt macro `nfl_garbage_time()` (football/dbt/macros/football_metrics.sql) applies the same rule
in SQL; tests/test_football.py checks the two agree play for play, and that the thresholds here
match the dbt vars.

NFL garbage time: a 4th-quarter play where the offense's win probability is below 10% or above 90%.
Plays with no win probability are kept.
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
