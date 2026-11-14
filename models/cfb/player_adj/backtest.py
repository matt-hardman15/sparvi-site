"""PLAYER-ADJ in the backtest harness: does a transfer's opponent- and teammate-adjusted production at his old
school predict his production at the new one better than his raw production did?

Fold (outcome season t, the season a mover plays for his new school):
  fit       choose the ridge penalties by held-out-game cross-validation on the training seasons (the
            `train_window` seasons ending at t-2), as of their last day.
  predict   as of the decision date (February 1 of season t, after the bowls and the January portal window):
            fit season t-1 with those penalties; every passer, rusher, and target with the role's minimum plays
            gets point = adjusted value and last_season = raw EPA/play, both standardized within role.
  outcome   season t: raw EPA/play per player and role, standardized within role over every player with the
            minimum plays; kept only for movers, players whose team in season t differs from season t-1.
entity_id = "<athlete id>:<role>". FBS games only (games in the season's games file).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfb.player_adj.model import LAMBDAS, choose_lambda, fit_kind, play_rows, player_values  # noqa: F401

MIN_PLAYS = {"passer": 100, "rusher": 40, "target": 25}


def _rows(loader, season: int, as_of, permute_seed: int | None = None) -> pd.DataFrame:
    games = loader.load(f"games_{season}", as_of, columns=["id"])
    plays = loader.load(f"plays_{season}", as_of)
    plays = plays[plays["gameId"].isin(set(games["id"]))]
    rows = play_rows(plays, loader.load(f"play_players_{season}", as_of))
    if permute_seed is not None:      # label-permutation leakage test: the EPA shuffled across plays
        rows["ppa"] = np.random.default_rng(permute_seed).permutation(rows["ppa"].to_numpy())
    return rows


def _z(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out.groupby("role")[c].transform(lambda v: (v - v.mean()) / v.std())
    return out


def _eligible(v: pd.DataFrame) -> pd.DataFrame:
    return v[v["plays"] >= v["role"].map(MIN_PLAYS)]


def fit(loader, fold, permute_seed: int | None = None) -> dict:
    train = [_rows(loader, s, fold.train_until) for s in fold.train_seasons]
    return {"lambda": {k: choose_lambda(train, k)[0] for k in ("pass", "run")}, "permute_seed": permute_seed}


def predict(fitted, loader, fold) -> pd.DataFrame:
    p = fold.test_season - 1
    rows = _rows(loader, p, fold.decision_date, fitted["permute_seed"])
    fits = {k: fit_kind(rows, k, fitted["lambda"][k]) for k in ("pass", "run")}
    v = _z(_eligible(player_values(rows, fits)), ["adj", "raw"])
    return pd.DataFrame({"entity_id": v["player_id"] + ":" + v["role"], "point": v["adj"], "last_season": v["raw"],
                         "role": v["role"], "team_prev": v["team"], "plays_prev": v["plays"]})


def outcome(loader, fold) -> pd.DataFrame:
    t = fold.test_season
    cur = _eligible(player_values(*_raw_only(_rows(loader, t, fold.resolve_after))))
    prev = player_values(*_raw_only(_rows(loader, t - 1, fold.resolve_after)))
    team_prev = dict(zip(prev["player_id"] + ":" + prev["role"], prev["team"], strict=True))
    cur = _z(cur, ["raw"]).assign(entity_id=lambda d: d["player_id"] + ":" + d["role"])
    moved = cur["entity_id"].map(team_prev).pipe(lambda tp: tp.notna() & (tp != cur["team"]))
    return pd.DataFrame({"entity_id": cur["entity_id"], "y": cur["raw"]})[moved.to_numpy()]


def _raw_only(rows: pd.DataFrame):
    """Raw values need no fit: a fit with no players (coefficients all zero) gives player_values' raw column."""
    means = {k: {"mean": float(rows.loc[rows["kind"] == k, "ppa"].mean()), "coef": {}} for k in ("pass", "run")}
    return rows, means
