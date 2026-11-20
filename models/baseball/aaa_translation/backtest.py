"""The proposed small-sample Triple-A translation (2.0.0) in the backtest harness (M5-EB).

The published translation (1.0.0) is one line per metric fit on players with 150+ PA/BF in Triple-A, and the site shows
nothing for a player below that. The candidate covers the players with 50-149: the Triple-A rate is shrunk toward the
Triple-A mean by its reliability (n / (n + k), k from split halves of earlier Triple-A seasons), and the line is fit on
every pair down to 50 PA/BF. Chosen on development folds that never read the test season
(baseball/aaa_translation/output/development.json). Above 150 the published line stays: the candidate was no better
there.

Fold (MLB outcome season t). Pairs, as the published model defines them: a player's Triple-A line in season t-1 and his
MLB line in season t ("across"), and his Triple-A line before his first MLB game of season t and his MLB line that
season ("within"). Fit on pairs with outcome seasons before t; predict the pairs of season t with 50-149 PA/BF in
Triple-A; score against MLB lines with 100+ PA/BF. Each metric's own sample is taken as PA or BF times its Triple-A-wide
count per PA (amended Nov 20 2026: the site's files carry PA and BF only; see the card and small_sample.py). One row per player x pair type x metric, in units of that
metric's MLB spread (its SD in the training pairs), so one RMSE covers every metric. Baseline on the same rows: the
published line stretched down to these players.

What the harness's date checks cover here. An across prediction reads Triple-A season t-1 as of March 1 of t. A
within prediction is made on the day of the call-up, so there is no one decision date for the fold: the fold's date
is the end of season t, and this module keeps the rule itself. From MLB season t, predict() loads only what defines a
first game (ids, dates, game type, pitch description); every outcome column is absent, and Triple-A pitches on or after
a player's first MLB game are dropped before any line is built. tests/test_backtest.py checks both on corrupted data.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from sparvi.config import ROOT

sys.path.insert(0, str(ROOT / "baseball" / "aaa_translation"))
import small_sample  # noqa: E402
from develop import COLS, METRICS, MIN_AAA, MIN_AAA_WIDE, MIN_MLB, _wls, lines_from, pairs  # noqa: E402

DEBUT_COLS = ["game_pk", "game_date", "game_year", "game_type", "pitcher", "batter", "at_bat_number", "pitch_number",
              "description"]                      # all predict() may read from the outcome season's MLB pitches
NO_AGES = pd.Series({-1: pd.Timestamp("2000-01-01")})   # pairs() wants birth dates; the candidate has no age term
_fits: dict = {}                                 # (loader, test season) -> the unpermuted fit, for outcome()'s units
_built: dict = {}                                # (what, loader, test season) -> lines already built from that loader


def _once(what: str, loader, fold, build):
    """Lines are the same for every call with one loader and fold (the label-permutation test refits many times);
    build them once. The loader is kept in the entry so a reused id can't return another loader's lines."""
    key = (what, id(loader), fold.test_season)
    if key not in _built or _built[key][0] is not loader:
        _built[key] = (loader, build())
    return _built[key][1]


def _load(loader, level: str, season: int, as_of, columns: list[str] = COLS) -> pd.DataFrame:
    d = loader.load(f"statcast_{level}_{season}", as_of, columns)
    d = d.assign(level=level.upper(), game_date=pd.to_datetime(d["game_date"]).dt.date)
    for c in COLS:
        if c not in d:
            d[c] = None if c in ("events",) else np.nan
    return d


def _train_lines(loader, fold) -> pd.DataFrame:
    def build():
        raw = pd.concat([_load(loader, lvl, s, fold.train_until) for lvl in ("aaa", "mlb") for s in fold.train_seasons], ignore_index=True)
        return lines_from(raw, loader.load("batter_heights", fold.train_until))
    return _once("train", loader, fold, build)


def _fit(lines: pd.DataFrame, before: int, permute_seed: int | None = None) -> dict:
    """{side: {metric: the published line, the candidate's constants, MLB spread and mean}} from pairs with outcome
    seasons before `before`. The candidate is fit by the code the site's constants come from (small_sample.fit)."""
    cand = small_sample.fit(lines, before, permute_seed)
    out = {}
    for side, ms in METRICS.items():
        P = pairs(lines, side, NO_AGES)
        P = P[P["outcome_season"] < before]
        out[side] = {}
        for m in ms:
            a, y = f"{m}_aaa", f"{m}_mlb"
            pub = P.dropna(subset=[a, y])
            pub = pub[pub["pa_aaa"].ge(MIN_AAA)]
            b_pub = _wls(np.column_stack([np.ones(len(pub)), pub[a]]), pub[y].to_numpy(float),
                         np.minimum(pub["pa_aaa"], pub["pa_mlb"]).to_numpy(float))
            c = cand[side][m]
            out[side][m] = {"published": b_pub, "candidate": c, "scale": c["mlb_sd"], "mean": c["mlb_mean"], "n_train": c["pairs"]}
    return out


def fit(loader, fold, permute_seed: int | None = None) -> dict:
    fitted = _fit(_train_lines(loader, fold), fold.test_season, permute_seed)
    if permute_seed is None:
        _fits[(id(loader), fold.test_season)] = (loader, fitted)
    return fitted


def _candidates(loader, fold) -> pd.DataFrame:
    """Triple-A lines to translate for season t: last season's ("across") and this season's before the first MLB game
    ("within"), 50-149 PA/BF. Nothing here reads an MLB outcome of season t."""
    t = fold.test_season
    raw = pd.concat([_load(loader, "aaa", t - 1, min(pd.Timestamp(f"{t}-03-01"), fold.decision_date)),
                     _load(loader, "aaa", t, fold.decision_date),
                     _load(loader, "mlb", t, fold.decision_date, DEBUT_COLS)], ignore_index=True)
    L = lines_from(raw, loader.load("batter_heights", fold.decision_date))
    L = L[L["level"].eq("AAA") & L["pa"].between(MIN_AAA_WIDE, MIN_AAA - 1)]
    across = L[L["segment"].eq("full") & L["season"].eq(t - 1)].assign(cohort="across")
    within = L[L["segment"].eq("pre_debut") & L["season"].eq(t)].assign(cohort="within")
    return pd.concat([across, within], ignore_index=True)


def predict(fitted, loader, fold) -> pd.DataFrame:
    C = _once("candidates", loader, fold, lambda: _candidates(loader, fold))
    rows = []
    for side, ms in METRICS.items():
        c = C[C["side"].eq(side)]
        for m in ms:
            f = fitted[side][m]
            d = c.dropna(subset=[m])
            x = d[m].to_numpy(float)
            point, _ = small_sample.apply(f["candidate"], x, d["pa"].to_numpy(float))
            ids = side + "|" + d["cohort"] + "|" + d["player_id"].astype(str)
            rows.append(pd.DataFrame({
                "entity_id": ids + "|" + m, "player": side + "|" + d["player_id"].astype(str), "side": side,
                "cohort": d["cohort"].to_numpy(), "metric": m, "pa_aaa": d["pa"].to_numpy(),
                "point": point / f["scale"],
                "published_line": (f["published"][0] + f["published"][1] * x) / f["scale"],
                "train_mean": f["mean"] / f["scale"]}))
    return pd.concat(rows, ignore_index=True)


def outcome(loader, fold) -> pd.DataFrame:
    key = (id(loader), fold.test_season)
    fitted = _fits[key][1] if key in _fits and _fits[key][0] is loader else fit(loader, fold)
    t = fold.test_season
    L = _once("outcome", loader, fold, lambda: lines_from(_load(loader, "mlb", t, fold.resolve_after),
                                                          loader.load("batter_heights", fold.resolve_after)))
    L = L[L["level"].eq("MLB") & L["segment"].eq("full") & L["pa"].ge(MIN_MLB)]
    rows = []
    for side, ms in METRICS.items():
        d0 = L[L["side"].eq(side)]
        for m in ms:
            d = d0.dropna(subset=[m])
            for cohort in ("across", "within"):
                rows.append(pd.DataFrame({"entity_id": side + "|" + cohort + "|" + d["player_id"].astype(str) + "|" + m,
                                          "y": d[m].to_numpy(float) / fitted[side][m]["scale"]}))
    return pd.concat(rows, ignore_index=True)
