"""Development of the college basketball team model (CBB-0): settings, feature groups and the gate statistics.

DEVELOPMENT ONLY. Seasons (a season is the year it ends):
    2021         seeds 2022's prior; never forecast
    2022, 2023   tuning: the ratings' settings and the second stage's feature groups are chosen here
    2024, 2025   development: scored with what the tuning seasons chose (an earlier prototype had already scored
                 these two, so they are not a test)
    2026         held out for the registered backtest. Nothing here can load it: `DevLoader` refuses any table of a
                 season from HELD_OUT on.

Fixed before the first run:
    grid           prior weight (games) x carry-over x recency half-life (days), GRID below; chosen by the lowest mean
                   absolute error of margin over 2022 and 2023
    feature groups features.GROUPS. Cross-season on the tuning seasons (fit on 2022, score 2023, and the reverse); a
                   group is kept if removing it from the full set raises that error. The interval of each removal's
                   effect is reported beside it.
    development    2024 is scored with everything fit on 2022-2023, 2025 with everything fit on 2022-2024: the margin's
                   spread, the Elo baseline's line, the returning-share mean and the second stage's weights. The
                   registered backtest fits the same things on every forecast season before its test season.
    baselines      home team by the average home margin (zero on a neutral court); last season's ratings never updated;
                   the source's pre-game Elo with a line fit of margin on the Elo difference and home court
    intervals      90%, 2,000 resamples of game dates, seed 7

The closing line is reported as an aggregate benchmark only (mean absolute error over the games that have one). It is
never an input and never shown per game.

    python -m cbb.ratings.develop            # about 20 minutes (the grid)
    python -m cbb.ratings.develop --quick    # reuse the grid result in the output file

Writes cbb/ratings/output/development.json (aggregates only).
"""
from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.asof import Loader
from cbb.ratings import features as F
from cbb.ratings import model as M
from cbb.ratings.scoring import (
    BASELINES,
    BOOT,
    SEED,
    add_baselines,
    by_date_diff,
    calibration,
    closing,
    fixed_from,
    log_loss,
    mae,
)

OUT = Path(__file__).resolve().parent / "output"
SEED_SEASON, TUNE, DEV, HELD_OUT = 2021, [2022, 2023], [2024, 2025], 2026
AS_OF = f"{HELD_OUT - 1}-10-31"                   # before the held-out season's first game
GRID = {"prior_weight": [1.0, 2.5, 5.0, 10.0, 20.0], "carry": [0.6, 0.8, 1.0], "half_life": [None, 90.0, 45.0]}


class DevLoader(Loader):
    """The point-in-time loader, refusing every table of the held-out season or later."""

    def load(self, table: str, as_of, columns: list[str] | None = None) -> pd.DataFrame:
        m = re.search(r"_(\d{4})$", table)
        if m and int(m.group(1)) >= HELD_OUT:
            raise PermissionError(f"{table}: season {m.group(1)} is held out of development (the registered backtest reads it)")
        if pd.Timestamp(as_of) > pd.Timestamp(AS_OF):
            raise PermissionError(f"as_of {as_of} is after the development cutoff {AS_OF}")
        return super().load(table, as_of, columns)


def tune(g: pd.DataFrame) -> dict:
    rows = []
    for pw, ca, hl in itertools.product(GRID["prior_weight"], GRID["carry"], GRID["half_life"]):
        f = M.walk(g[g["season"] <= max(TUNE)], TUNE, M.Params(pw, ca, hl))
        rows.append({"prior_weight": pw, "carry": ca, "half_life": hl, "margin_mae": mae(f, "pred_margin"),
                     "total_mae": mae(f, "pred_total", "total"), "games": int(len(f))})
        print(rows[-1], flush=True)
    best = min(rows, key=lambda r: r["margin_mae"])
    return {"grid": rows, "chosen": {k: best[k] for k in ("prior_weight", "carry", "half_life")}, "chosen_margin_mae": best["margin_mae"],
            "on_grid_edge": {k: best[k] in (GRID[k][0], GRID[k][-1]) for k in GRID}}


def select(tune_f: pd.DataFrame) -> dict:
    """Leave-one-group-out on the tuning seasons, cross-season."""
    allc = [c for cols in F.GROUPS.values() for c in cols]

    def errors(cols: list[str]) -> np.ndarray:
        out = pd.Series(np.nan, index=tune_f.index)
        for s in TUNE:
            tr, te = tune_f[tune_f["season"] != s], tune_f[tune_f["season"] == s]
            pred = F.apply(te, cols, F.fit(tr, cols)) if cols else te["pred_margin"].to_numpy(float)
            out[te.index] = np.abs(te["margin"].to_numpy(float) - pred)
        return out.to_numpy()

    base, full = errors([]), errors(allc)
    res = {"ratings_only": float(base.mean()), "all_groups": float(full.mean()), "all_minus_ratings": by_date_diff(tune_f, full, base),
           "groups": {}, "kept": []}
    for name, cols in F.GROUPS.items():
        without, alone = errors([c for c in allc if c not in cols]), errors(cols)
        res["groups"][name] = {"without": float(without.mean()), "removal_raises_error_by": by_date_diff(tune_f, without, full),
                               "alone": float(alone.mean()), "alone_minus_ratings": by_date_diff(tune_f, alone, base)}
        if without.mean() > full.mean():
            res["kept"].append(name)
    kept = [c for n in res["kept"] for c in F.GROUPS[n]]
    k = errors(kept)
    res["kept_set"] = {"mae": float(k.mean()), "minus_ratings": by_date_diff(tune_f, k, base)}
    return res


def score(d: pd.DataFrame) -> dict:
    """Everything reported for a set of scored games. `d` has pred_margin (ratings), pred2 (with the second stage),
    pred_total, the baselines, p_home / p_home2, and the closing line where there is one."""
    y = (d["margin"] > 0).to_numpy(float)
    err = {k: (d["margin"] - d[c]).abs().to_numpy() for k, c in {"ratings": "pred_margin", "second_stage": "pred2", **BASELINES}.items()}
    out = {"games": int(len(d)), "dates": int(d["date"].nunique()),
           "margin_mae": {k: float(v.mean()) for k, v in err.items()},
           "total_mae": {"model": mae(d, "pred_total", "total"), "season_mean": float((d["total"] - d["total_mean"]).abs().mean())},
           "win": {k: {"log_loss": log_loss(d[c].to_numpy(), y), "accuracy": float(((d[c] > 0.5) == (y == 1)).mean()),
                       **calibration(d[c].to_numpy(), y)} for k, c in (("ratings", "p_home"), ("second_stage", "p_home2"))},
           "edges": {m: {b: by_date_diff(d, err[m], err[b]) for b in BASELINES} for m in ("ratings", "second_stage")}}
    out["edges"]["second_stage"]["ratings"] = by_date_diff(d, err["second_stage"], err["ratings"])
    cl = d.dropna(subset=["close_margin"])
    ct = d.dropna(subset=["close_total"])
    out["closing_line_benchmark"] = {"games": int(len(cl)), "share_of_games": float(len(cl) / len(d)),
                                     "margin_mae": {"line": mae(cl, "close_margin"), "ratings": mae(cl, "pred_margin"), "second_stage": mae(cl, "pred2")},
                                     "total_mae": {"line": mae(ct, "close_total", "total"), "model": mae(ct, "pred_total", "total")}}
    return out


def develop(loader, g: pd.DataFrame, params: M.Params, kept: list[str]) -> tuple[dict, pd.DataFrame]:
    """Walk-forward development: season t scored with everything fit on the forecast seasons before it."""
    f = M.walk(g, [*TUNE, *DEV], params)
    frames, fits = [], {}
    for t in DEV:
        train, test = f[f["season"] < t], f[f["season"] == t]
        fx = fixed_from(train)
        ret_mean = F.raw_returning(train, loader, sorted(train["season"].unique()), AS_OF)
        x = F.build(f[f["season"] <= t], loader, sorted(f.loc[f["season"] <= t, "season"].unique()), AS_OF, ret_mean)
        xtr, xte = x[x["season"] < t], x[x["season"] == t]
        cols = [c for n in kept for c in F.GROUPS[n]]
        beta = F.fit(xtr, cols)
        d = add_baselines(xte, fx).assign(pred2=F.apply(xte, cols, beta), total_mean=float(train["total"].mean()))
        d["p_home"], d["p_home2"] = M.win_probability(d["pred_margin"], fx["margin_sd"]), M.win_probability(d["pred2"], fx["margin_sd"])
        frames.append(d)
        fits[str(t)] = {**fx, "returning_mean": ret_mean, "second_stage": dict(zip(["intercept", *cols], map(float, beta), strict=True)),
                        "fit_on": sorted(int(s) for s in train["season"].unique()), "games_fit_on": int(len(train))}
        assert test["game_id"].tolist() == xte["game_id"].tolist()
    d = pd.concat(frames, ignore_index=True).merge(closing(loader, DEV, AS_OF), on="game_id", how="left")
    d["month"], d["game_type"] = d["date"].dt.strftime("%b"), M.game_type(d)
    rep = {"fits": fits, "pooled": score(d), "by_season": {str(s): score(x) for s, x in d.groupby("season")}}
    for seg in ("month", "game_type"):
        rep[f"by_{seg}"] = {str(k): {"games": int(len(x)), "margin_mae": {m: mae(x, c) for m, c in (("ratings", "pred_margin"), ("second_stage", "pred2"), ("elo", "elo_margin"))},
                                     "second_stage_minus_elo": by_date_diff(x, (x["margin"] - x["pred2"]).abs(), (x["margin"] - x["elo_margin"]).abs()),
                                     "total_mae": mae(x, "pred_total", "total"),
                                     "log_loss": log_loss(x["p_home2"].to_numpy(), (x["margin"] > 0).to_numpy(float))}
                            for k, x in d.groupby(seg)}
    return rep, d


def main() -> None:
    loader = DevLoader()
    g = M.load_games(loader, [SEED_SEASON, *TUNE, *DEV], AS_OF)
    out = OUT / "development.json"
    res = {"seasons": {"seed": SEED_SEASON, "tune": TUNE, "develop": DEV, "held_out": HELD_OUT}, "as_of": AS_OF, "grid_searched": GRID,
           "games_by_season": {str(s): int(n) for s, n in g["season"].value_counts().sort_index().items()},
           "bootstrap": {"samples": BOOT, "seed": SEED, "blocks": "game date"}, "feature_groups": F.GROUPS}
    if "--quick" in sys.argv and out.exists():
        res["tuning"] = json.loads(out.read_text(encoding="utf-8"))["tuning"]
    else:
        res["tuning"] = tune(g)
    params = M.Params(**res["tuning"]["chosen"])
    tune_f = M.walk(g[g["season"] <= max(TUNE)], TUNE, params)
    ret_mean = F.raw_returning(tune_f, loader, TUNE, AS_OF)
    tune_x = F.build(tune_f, loader, TUNE, AS_OF, ret_mean)
    res["selection"] = select(tune_x)
    res["development"], _ = develop(loader, g, params, res["selection"]["kept"])
    OUT.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    p = res["development"]["pooled"]
    print(json.dumps({"chosen": res["tuning"]["chosen"], "kept": res["selection"]["kept"], "margin_mae": p["margin_mae"],
                      "edges": p["edges"], "closing": p["closing_line_benchmark"]}, indent=1))


if __name__ == "__main__":
    main()
