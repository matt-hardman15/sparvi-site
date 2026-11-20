"""The small-sample Triple-A translation (M5-EB, aaa_translation 2.0.0): a line for players with 50-149 PA or batters
faced in Triple-A, whom the published line (150+) doesn't cover.

For each metric: shrink the Triple-A rate toward the Triple-A mean by its reliability, then put it through a line fit
on every pair down to 50 PA/BF, weighted by the MLB sample.

    r      = pa / (pa + k)                       how much of the rate is signal at this sample
    shrunk = mean + r * (x - mean)
    mlb    = intercept + slope * shrunk
    band   = mlb +/- z80 * sqrt(base + slope^2 * talent * (1 - r))      wider as the Triple-A sample shrinks

`k` is in plate appearances (batters faced for pitchers). Each metric's own sample (pitches outside the zone for chase
rate, swings for whiff rate, balls in play for hard-hit rate) is taken as PA or BF times that metric's Triple-A-wide
count per PA: the site's files carry PA and BF only, and on the development folds the result is the same as with exact
counts (50-149 pool, both folds: hitters 0.8629 either way, pitchers 0.9596 and 0.9598; develop.py). The noise and
talent variances behind `k` are estimated from odd-game and even-game halves with the exact counts.

The fit needs the Triple-A history seasons (`python ingest/baseball/pull.py --history`), which the refresh never has,
so the fitted constants are committed in small_sample.json and the build only reads them. Refit by hand each offseason
(a new minor version of the model):

    python baseball/aaa_translation/small_sample.py        # about two minutes; rewrites small_sample.json

The backtest (models/baseball/aaa_translation/backtest.py) calls `fit` with the pairs before its test season and
`apply` on the test pairs, so what is tested is this code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from develop import METRICS, MIN_AAA, MIN_AAA_WIDE, MIN_MLB, Z80, _wls, pairs, reliability, shrink  # noqa: E402

PARAMS = HERE / "small_sample.json"
NO_AGES = pd.Series({-1: pd.Timestamp("2000-01-01")})     # pairs() wants birth dates; this model has no age term


def count_per_pa(lines: pd.DataFrame, side: str, before: int | None) -> dict[str, float]:
    """Each metric's sample count per PA (BF), over Triple-A season lines with 50+ (seasons before `before`)."""
    A = lines[lines["side"].eq(side) & lines["level"].eq("AAA") & lines["segment"].eq("full") & lines["pa"].ge(MIN_AAA_WIDE)]
    if before is not None:
        A = A[A["season"].lt(before)]
    return {m: float(A[n].sum() / A["pa"].sum()) for m, n in METRICS[side].items()}


def fit(lines: pd.DataFrame, before: int | None = None, permute_seed: int | None = None) -> dict:
    """{side: {metric: constants}} from pairs with outcome seasons before `before` (every pair when None)."""
    last = int(lines["season"].max()) + 1 if before is None else before
    rng = np.random.default_rng(permute_seed) if permute_seed is not None else None
    out = {}
    for side, ms in METRICS.items():
        P = pairs(lines, side, NO_AGES)
        P = P[P["outcome_season"] < last]
        rel, per_pa = reliability(lines, side, last), count_per_pa(lines, side, before)
        out[side] = {}
        for m in ms:
            a, y = f"{m}_aaa", f"{m}_mlb"
            tr = P.dropna(subset=[a, y]).copy()
            if rng is not None:      # label-permutation leakage test: MLB outcomes shuffled across the training pairs
                tr[y] = rng.permutation(tr[y].to_numpy())
            rm = rel[m]
            n = tr["pa_aaa"].to_numpy(float) * per_pa[m]
            xs, left = shrink(tr[a].to_numpy(float), n, rm)
            X = np.column_stack([np.ones(len(tr)), xs])
            yv, w = tr[y].to_numpy(float), tr["pa_mlb"].to_numpy(float)
            b = _wls(X, yv, w)
            base = max(float(np.average((yv - X @ b) ** 2 - b[1] ** 2 * left, weights=w)), 0.0)
            out[side][m] = {"mean": rm["mean"], "k": rm["k"] / per_pa[m], "talent": rm["talent"], "count_per_pa": per_pa[m],
                            "intercept": float(b[0]), "slope": float(b[1]), "base": base,
                            "pairs": int(len(tr)), "pairs_under_150": int((tr["pa_aaa"] < MIN_AAA).sum()),
                            "mlb_sd": float(P[y].std()), "mlb_mean": float(tr[y].mean())}
    return out


def apply(c: dict, x, pa) -> tuple[np.ndarray, np.ndarray]:
    """The MLB translation of Triple-A rate `x` from `pa` PA/BF, and the half-width of its 80% band."""
    x, pa = np.asarray(x, float), np.asarray(pa, float)
    r = pa / (pa + c["k"])
    t = c["intercept"] + c["slope"] * (c["mean"] + r * (x - c["mean"]))
    return t, Z80 * np.sqrt(c["base"] + c["slope"] ** 2 * c["talent"] * (1 - r))


def main() -> None:
    from develop import CFG, COLS, Loader, lines_from
    seasons = {"aaa": CFG.aaa_history_seasons + CFG.aaa_seasons, "mlb": [s for s in CFG.history_seasons + CFG.mlb_seasons
                                                                         if s >= min(CFG.aaa_history_seasons + CFG.aaa_seasons)]}
    if not CFG.aaa_history_seasons:
        raise SystemExit("no Triple-A history seasons configured (sparvi.toml aaa_history_first)")
    loader, frames = Loader(), []
    for lvl, ss in seasons.items():
        for s in ss:
            d = loader.load(f"statcast_{lvl}_{s}", f"{s + 1}-03-01", COLS)
            frames.append(d.assign(level=lvl.upper(), game_date=pd.to_datetime(d["game_date"]).dt.date))
    lines = lines_from(pd.concat(frames, ignore_index=True), loader.load("batter_heights", f"{CFG.season + 1}-03-01"))
    fitted = fit(lines)
    out = {"fitted_through": CFG.season, "seasons": seasons, "min_pa": MIN_AAA_WIDE, "max_pa": MIN_AAA - 1, "min_mlb": MIN_MLB,
           "z80": Z80, "band_pct": [10, 90],
           "note": "Fitted by hand (the history seasons are not in the refresh). The build reads this file; it does not refit.",
           **fitted}
    PARAMS.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    for side, ms in fitted.items():
        print(f"\n{side}")
        for m, c in ms.items():
            t, half = apply(c, np.array([c["mean"], c["mean"]]), np.array([50.0, 149.0]))
            print(f"  {m:17s} pairs {c['pairs']:4d} ({c['pairs_under_150']} under 150)  k {c['k']:7.1f} PA  slope {c['slope']:.3f}  "
                  f"band half-width at 50 / 149 PA: {half[0]:.4f} / {half[1]:.4f}")


if __name__ == "__main__":
    main()
