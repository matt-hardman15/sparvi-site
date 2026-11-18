"""Development folds for the pitch-quality models (PQ-WINDOW): does training on more seasons help?

DEVELOPMENT ONLY. The pre-registered test is models/baseball/stuff_plus/backtest.py (train 2024, grade 2025, score
against 2026); nothing here reads an outcome from the current season, so choices made on these folds stay clear of
it. Needs the MLB history seasons in data/raw (`python ingest/baseball/pull.py --history`, by hand, once).

Fold (outcome season t < the current season), every read through the point-in-time loader:
    train on a window of seasons ending at t-2  ->  grade season t-1 pitchers  ->  correlate with season t RV/100
Windows compared on the same pitchers (500+ pitches in t-1 and t):
    1    season t-2 alone (what the published test does)
    2    seasons t-3 and t-2 (what the published model does)
    all  every history season through t-2
The edge of each longer window over one season is a paired bootstrap over pitchers, per fold and pooled.

    python baseball/pitch_quality/develop.py            # Stuff+ (about 90 minutes on 4 cores)
    python baseball/pitch_quality/develop.py --kinds stuff loc pitching

Writes output/development.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from features import build_features
from train import MIN_PITCHES, SEED, fit_kinds, pitcher_season, predict_all, xrv100  # noqa: F401  (MIN_PITCHES: query strings)

from backtest.asof import Loader
from sparvi.config import baseball

CFG = baseball()
OUT = Path(__file__).resolve().parent / "output"
BOOT = 2000


def folds() -> dict[int, dict[str, list[int]]]:
    """Outcome season -> {window name: training seasons}. Outcome seasons stop before the current season (the test's
    outcome); a fold needs at least two training seasons to compare anything."""
    first = min(CFG.history_seasons + CFG.mlb_seasons)
    out = {}
    for t in range(first + 3, CFG.season):
        last, w = t - 2, {}
        for name, seasons in (("1", [last]), ("2", [last - 1, last]), ("all", list(range(first, last + 1)))):
            if seasons not in w.values():
                w[name] = seasons
        out[t] = w
    return out


def _pitches(loader: Loader, season: int, as_of: str, heights: pd.Series) -> pd.DataFrame:
    return build_features(loader.load(f"statcast_mlb_{season}", as_of), heights)


def run(kinds: list[str]) -> dict:
    loader = Loader()
    assert all(t < CFG.season for t in folds()), "a development fold may not use the current season's outcomes"
    h = loader.load("batter_heights", f"{CFG.season}-03-01")
    heights = (h.set_index("batter")["height_in"] / 12).astype(float)
    rng = np.random.default_rng(SEED)
    res = {"kinds": kinds, "min_pitches": MIN_PITCHES, "bootstrap_samples": BOOT, "folds": {}}
    pooled = []
    for t, windows in folds().items():
        graded = _pitches(loader, t - 1, f"{t}-03-01", heights)
        outcome = pitcher_season(_pitches(loader, t, f"{t + 1}-03-01", heights)).query("pitches >= @MIN_PITCHES")
        fold = {"grade_season": t - 1, "outcome_season": t, "windows": {}}
        cols = {}
        for name, seasons in windows.items():
            tr = pd.concat([_pitches(loader, s, f"{t - 1}-03-01", heights) for s in seasons], ignore_index=True)
            models = fit_kinds(tr, kinds=kinds)
            d = graded.copy()
            predict_all(models, d)
            ps = pitcher_season(d).query("pitches >= @MIN_PITCHES").merge(
                outcome[["pitcher", "rv100"]].rename(columns={"rv100": "y"}), on="pitcher")
            for k in kinds:
                cols[f"{name}|{k}"] = ps.set_index("pitcher")[xrv100(k)]
            fold["windows"][name] = {"train_seasons": seasons, "train_pitches": int(len(tr)), "n": int(len(ps)),
                                     "r": {k: float(ps[xrv100(k)].corr(ps["y"])) for k in kinds},
                                     "r_last_season_results": float(ps["rv100"].corr(ps["y"])),
                                     "trees": {f"{k}/{g}": [m.best_iteration for m in bag]
                                               for k in kinds for g, bag in models[k].items()}}
            print(f"  outcome {t}, window {name} {seasons}: " + ", ".join(f"{k} r {fold['windows'][name]['r'][k]:.4f}" for k in kinds),
                  flush=True)
        wide = (pd.DataFrame(cols).join(outcome.set_index("pitcher")["rv100"].rename("y"), how="inner").dropna()
                .rename_axis("pitcher").reset_index())
        pooled.append(wide.assign(fold=t))
        fold["edges"] = _edges(wide, [n for n in windows if n != "1"], kinds, rng)
        res["folds"][str(t)] = fold
    allf = pd.concat(pooled, ignore_index=True)
    common = [n for n in ("2", "all") if all(n in f["windows"] for f in res["folds"].values())]
    res["pooled"] = {"n": int(len(allf)), "edges": _edges(allf, common, kinds, rng, by_fold=True)}
    return res


def _r(d: pd.DataFrame, col: str, by_fold: bool) -> float:
    """Correlation with the outcome; pooled over folds as the mean of the per-fold correlations."""
    if not by_fold:
        return float(d[col].corr(d["y"]))
    return float(np.mean([g[col].corr(g["y"]) for _, g in d.groupby("fold")]))


def _edges(d: pd.DataFrame, names: list[str], kinds: list[str], rng, by_fold: bool = False) -> dict:
    out = {}
    for n in names:
        for k in kinds:
            a, b = f"{n}|{k}", f"1|{k}"
            if a not in d or d[a].isna().all():
                continue
            est = _r(d, a, by_fold) - _r(d, b, by_fold)
            groups = list(d.groupby("pitcher").indices.values())      # a pitcher's rows (one per fold) travel together
            draws = []
            for _ in range(BOOT):
                s = d.iloc[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])]
                draws.append(_r(s, a, by_fold) - _r(s, b, by_fold))
            out[f"{n}_minus_1|{k}"] = {"est": est, "ci90": [float(np.percentile(draws, 5)), float(np.percentile(draws, 95))]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["stuff"], choices=["stuff", "loc", "pitching"])
    a = ap.parse_args()
    r = run(a.kinds)
    (OUT / "development.json").write_text(json.dumps(r, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"pooled": r["pooled"], **{t: f["edges"] for t, f in r["folds"].items()}}, indent=2))
