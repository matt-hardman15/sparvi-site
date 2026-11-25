"""Scoring shared by the development script and the registered backtest of the college basketball team rating:
errors, intervals by game date, the win probability's log loss and calibration, and the baselines' fits."""
from __future__ import annotations

import numpy as np
import pandas as pd

BOOT, SEED = 2000, 7
BASELINES = {"home": "home_margin", "prior": "prior_margin", "elo": "elo_margin"}


def mae(d: pd.DataFrame, col: str, y: str = "margin") -> float:
    return float((d[y] - d[col]).abs().mean())


def by_date_diff(d: pd.DataFrame, a: np.ndarray, b: np.ndarray, boot: int = BOOT, seed: int = SEED) -> dict:
    """Mean of a - b over games, with a 90% interval from resampling game dates."""
    x = pd.DataFrame({"date": d["date"].to_numpy(), "v": np.asarray(a, float) - np.asarray(b, float)}).groupby("date")["v"].agg(["sum", "size"])
    s, n = x["sum"].to_numpy(), x["size"].to_numpy()
    ix = np.random.default_rng(seed).integers(0, len(s), size=(boot, len(s)))
    draws = s[ix].sum(axis=1) / n[ix].sum(axis=1)
    return {"est": float(s.sum() / n.sum()), "ci90": [float(np.percentile(draws, 5)), float(np.percentile(draws, 95))],
            "games": int(n.sum()), "dates": int(len(s))}


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    d = pd.DataFrame({"p": p, "y": y})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False)
    t = d.groupby("bin").agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    return {"ece": float((t["n"] * (t["predicted"] - t["observed"]).abs()).sum() / t["n"].sum()),
            "table": [{"n": int(r.n), "predicted": round(float(r.predicted), 4), "observed": round(float(r.observed), 4)} for r in t.itertuples()]}


def fixed_from(train: pd.DataFrame) -> dict:
    """What a forecast season takes from the forecast seasons before it: the margin's spread and the baselines' fits."""
    site = (~train["neutral"]).to_numpy(float)
    e = train.dropna(subset=["elo_diff"])
    X = np.c_[e["elo_diff"].to_numpy(float), (~e["neutral"]).to_numpy(float)]
    b = np.linalg.lstsq(X, e["margin"].to_numpy(float), rcond=None)[0]
    return {"margin_sd": float((train["margin"] - train["pred_margin"]).std()),
            "home_margin": float(train["margin"].to_numpy()[site == 1].mean()),
            "elo_per_point": float(b[0]), "elo_home": float(b[1])}


def add_baselines(d: pd.DataFrame, fx: dict) -> pd.DataFrame:
    site = (~d["neutral"]).to_numpy(float)
    return d.assign(home_margin=fx["home_margin"] * site,
                    elo_margin=fx["elo_per_point"] * d["elo_diff"].fillna(0.0).to_numpy() + fx["elo_home"] * site)


def closing(loader, seasons: list[int], as_of) -> pd.DataFrame:
    """The closing spread and total per game, averaged over providers, from the home side. For aggregate scoring only:
    never a model input, never shown per game."""
    ln = pd.concat([loader.load(f"cbb_lines_{s}", as_of, columns=["gameId", "spread", "overUnder"]) for s in seasons])
    ln = ln.groupby("gameId", as_index=False)[["spread", "overUnder"]].mean()
    return pd.DataFrame({"game_id": ln["gameId"].astype("int64"), "close_margin": -ln["spread"], "close_total": ln["overUnder"]})
