"""College basketball team rating in the backtest harness (CBB-0).

Fold (season t, the year it ends): everything that is fit on other seasons is fit on the forecast seasons before t,
read as of the day before t starts: the margin's spread, the baselines' fits and the second stage's weights. The ratings'
settings and the second stage's feature groups are fixed (cbb/ratings/model.py PUBLISHED, features.py KEPT; chosen on
2022 and 2023). Season t's games are then forecast one date at a time, each from that season's games before its date.

The fold's decision date is the end of season t, because the forecasts are made throughout it; what keeps a forecast
from reading its own or a later game is the walk-forward itself, tested in tests/test_cbb_model.py (forecasts and
features for a date do not change when every later game is changed).

Rows: one per counted game of season t. `point` is the forecast home margin (ratings plus second stage).
Baselines: home court, last season's ratings never updated, the source's pre-game Elo; `ratings` is the first stage alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cbb.ratings import features as F
from cbb.ratings import model as M
from cbb.ratings import scoring as S

COLS = [c for name in F.KEPT for c in F.GROUPS[name]]
EXTRA = {"second_stage": "point", "ratings": "ratings", "home": "home", "prior": "prior", "elo": "elo"}


def _seasons(fold) -> list[int]:
    return [int(s) for s in fold.train_seasons]


def fit(loader, fold) -> dict:
    seasons = _seasons(fold)                       # the first only seeds the next one's prior
    g = M.load_games(loader, seasons, fold.train_until)
    f = M.walk(g, seasons[1:], M.PUBLISHED)
    ret_mean = F.raw_returning(f, loader, seasons[1:], fold.train_until)
    x = F.build(f, loader, seasons[1:], fold.train_until, ret_mean)
    return {"fixed": S.fixed_from(f), "returning_mean": ret_mean, "beta": F.fit(x, COLS), "total_mean": float(f["total"].mean()),
            "fit_games": int(len(f))}


def predict(fitted, loader, fold) -> pd.DataFrame:
    t = int(fold.test_season)
    g = M.load_games(loader, [*_seasons(fold), t], fold.decision_date)
    f = M.walk(g, [t], M.PUBLISHED)
    x = S.add_baselines(F.build(f, loader, [t], fold.decision_date, fitted["returning_mean"]), fitted["fixed"])
    point = F.apply(x, COLS, fitted["beta"])
    out = pd.DataFrame({
        "entity_id": x["game_id"].to_numpy(), "point": point, "ratings": x["pred_margin"].to_numpy(),
        "home": x["home_margin"].to_numpy(), "prior": x["prior_margin"].to_numpy(), "elo": x["elo_margin"].to_numpy(),
        "pred_total": x["pred_total"].to_numpy(), "total_baseline": fitted["total_mean"],
        "p_home": M.win_probability(point, fitted["fixed"]["margin_sd"]),
        "date": x["date"].to_numpy(), "month": x["date"].dt.strftime("%b").to_numpy(), "game_type": M.game_type(x).to_numpy()})
    # The closing line, for the aggregate benchmark in `secondary` only. It is read after every forecast above is made.
    return out.merge(S.closing(loader, [t], fold.decision_date).rename(columns={"game_id": "entity_id"}), on="entity_id", how="left")


def outcome(loader, fold) -> pd.DataFrame:
    g = M.load_games(loader, [int(fold.test_season)], fold.resolve_after)
    return pd.DataFrame({"entity_id": g["game_id"].to_numpy(), "y": g["margin"].to_numpy(), "y_total": g["total"].to_numpy()})


def secondary(df: pd.DataFrame) -> dict:
    """Reported with the run, not part of the gate: the total, the win probability, months, and the closing-line
    benchmark (aggregate)."""
    d = df.rename(columns={"y": "margin"})
    y = (d["margin"] > 0).to_numpy(float)
    tot, tot0 = (d["y_total"] - d["pred_total"]).abs().to_numpy(), (d["y_total"] - d["total_baseline"]).abs().to_numpy()
    out = {"total_mae": {"model": float(tot.mean()), "earlier_seasons_mean": float(tot0.mean()), "edge": S.by_date_diff(d, tot, tot0)},
           "win_probability": {"log_loss": S.log_loss(d["p_home"].to_numpy(), y), "log_loss_home_rate": S.log_loss(np.full(len(d), y.mean()), y),
                               "accuracy": float(((d["p_home"] > 0.5) == (y == 1)).mean()), **S.calibration(d["p_home"].to_numpy(), y)},
           "by_month": {str(k): {"games": int(len(x)), "margin_mae": {m: float((x["margin"] - x[c]).abs().mean()) for m, c in EXTRA.items()},
                                 "edge_over_elo": S.by_date_diff(x, (x["margin"] - x["point"]).abs(), (x["margin"] - x["elo"]).abs()),
                                 "total_mae": float((x["y_total"] - x["pred_total"]).abs().mean())}
                        for k, x in d.groupby("month")}}
    cl, ct = d.dropna(subset=["close_margin"]), d.dropna(subset=["close_total"])
    out["closing_line_benchmark"] = {
        "note": "aggregate accuracy benchmark only; never an input, never per game",
        "games": int(len(cl)), "share_of_games": float(len(cl) / len(d)),
        "margin_mae": {"line": S.mae(cl, "close_margin"), "second_stage": S.mae(cl, "point"), "ratings": S.mae(cl, "ratings")},
        "total_mae": {"line": S.mae(ct, "close_total", "y_total"), "model": S.mae(ct, "pred_total", "y_total")}}
    return out
