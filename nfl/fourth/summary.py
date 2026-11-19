"""Aggregates of the fourth-down decisions for the site (F4): team-season totals, the league by season, and the
costliest clear calls. Reads output/decisions_{season}.parquet (decisions.py); nfl/export_site_data.py writes what
this returns.

Definitions (the page states them):
    clear go      the 90% interval of go-minus-best-kick is entirely above zero
    clear kick    entirely below zero
    toss-up       the interval includes zero; no choice there counts for or against a team
    cost          win probability given up on clear calls (best choice minus the actual one); it sums over a team's
                  decisions, so 1.0 is one full win's worth of chances
Every team total must equal the sum of its decision rows (checked here; PLAN-football F4 "done when").

Not judged: decisions in the last LATE_SECONDS of either half and in overtime. There the clock decides what a
choice is worth, and the win-probability model, well calibrated over the game as a whole, is not reliable: on
held-out seasons a team with the ball and a lead in the final 40 seconds won 99.8% of the time and was rated 95.9%,
and calibration error in the last minute and in overtime is well above the rest of the game
(nfl/fourth/late_check.py -> output/late_calibration.json). Errors that size are larger than the gaps being judged.
`load()` drops those rows; `late_share()` reports how many there were.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "output"
TOP = 15                    # costliest calls listed per season
LATE_SECONDS = 120          # the last two minutes of each half are not judged (see above)


def _all(seasons) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(OUT / f"decisions_{s}.parquet") for s in seasons], ignore_index=True)


def _late(d: pd.DataFrame) -> pd.Series:
    return (d["game_half"] == "Overtime") | (d["half_seconds_remaining"] <= LATE_SECONDS)


def load(seasons) -> pd.DataFrame:
    """The judged decisions: everything except the end of each half and overtime."""
    d = _all(seasons)
    return d[~_late(d)].reset_index(drop=True)


def late_share(seasons) -> tuple[int, int]:
    """(decisions not judged, all decisions)."""
    d = _all(seasons)
    return int(_late(d).sum()), int(len(d))


def _agg(d: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    go, kick = d["call"] == "go", d["call"] == "kick"
    went = d["choice"] == "go"
    g = d.assign(clear_go=go, clear_kick=kick, toss_up=d["call"] == "toss-up", went=went, went_on_clear_go=go & went,
                 went_on_clear_kick=kick & went, cost_passive=np.where(go & ~went, d["cost"], 0.0),
                 cost_aggressive=np.where(kick & went, d["cost"], 0.0))
    t = g.groupby(by).agg(decisions=("cost", "size"), went=("went", "sum"), clear_go=("clear_go", "sum"),
                          went_on_clear_go=("went_on_clear_go", "sum"), clear_kick=("clear_kick", "sum"),
                          went_on_clear_kick=("went_on_clear_kick", "sum"), toss_up=("toss_up", "sum"),
                          cost=("cost", "sum"), cost_passive=("cost_passive", "sum"),
                          cost_aggressive=("cost_aggressive", "sum")).reset_index()
    t["go_rate_clear"] = t["went_on_clear_go"] / t["clear_go"].where(t["clear_go"] > 0)
    return t


def team_seasons(d: pd.DataFrame) -> pd.DataFrame:
    t = _agg(d, ["season", "posteam"]).rename(columns={"posteam": "team"})
    # Reconcile: totals equal the decision rows they come from.
    chk = d.groupby(["season", "posteam"])["cost"].sum()
    got = t.set_index(["season", "team"])["cost"]
    if not np.allclose(got.to_numpy(), chk.reindex(got.index).to_numpy()) or int(t["decisions"].sum()) != len(d):
        raise ValueError("fourth-down team totals don't reconcile to the decision rows")
    if not np.allclose(t["cost"], t["cost_passive"] + t["cost_aggressive"]):
        raise ValueError("fourth-down cost doesn't split into passive + aggressive")
    return t.sort_values(["season", "team"]).reset_index(drop=True)


def league(d: pd.DataFrame) -> pd.DataFrame:
    return _agg(d, ["season"]).sort_values("season").reset_index(drop=True)


def costliest(d: pd.DataFrame, top: int = TOP) -> pd.DataFrame:
    """The clear calls that gave up the most win probability, per season. A handful of described plays, not data."""
    c = d[d["cost"] > 0].sort_values(["season", "cost", "game_id", "play_id"], ascending=[True, False, True, True])
    c = c.groupby("season").head(top)
    cols = ["season", "week", "season_type", "posteam", "defteam", "ydstogo", "yardline_100", "score_differential",
            "game_seconds_remaining", "game_half", "choice", "call", "wp_go", "wp_kick", "wp_actual", "margin", "margin_lo", "margin_hi", "cost"]
    return c[cols].rename(columns={"posteam": "team", "defteam": "opponent"}).reset_index(drop=True)
