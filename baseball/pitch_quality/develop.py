"""Development folds for the pitch-quality models: does training on more seasons help (PQ-WINDOW), and does a
target with the game context taken out help (PQ-TARGET)?

DEVELOPMENT ONLY. The pre-registered test is models/baseball/stuff_plus/backtest.py (train 2024, grade 2025, score
against 2026); nothing here reads an outcome from the current season, so choices made on these folds stay clear of
it. Needs the MLB history seasons in data/raw (`python ingest/baseball/pull.py --history`, by hand, once).

Fold (outcome season t < the current season), every read through the point-in-time loader:
    train on a window of seasons ending at t-2  ->  grade season t-1 pitchers  ->  correlate with season t RV/100
The outcome is always the pitcher's actual run value per 100 pitches, whatever the model was trained on.

Windows, compared on the same pitchers (500+ pitches in t-1 and t):
    1    season t-2 alone (what the published test does)
    2    seasons t-3 and t-2 (what the published model does)
    all  every history season through t-2
Targets (features.py's `rv_x` is the published one; the others are built here, from the training seasons only):
    rv_x      run value of the pitch as Statcast records it (count and base-out state included), with balls in play
              swapped for their expected value
    cn        context-neutral: every pitch gets the training seasons' average run value of its outcome (ball, called
              strike, swinging strike, foul, hit by pitch, in play), whatever the count or the runners; balls in play
              move from that average by their expected wOBA
    cn_count  the same averages by outcome and count: the base-out state is removed, the count is kept
Each variant's edge over the reference (window 1, or target rv_x) is a paired bootstrap over pitchers, per fold and
pooled (the mean of the folds' correlations).

    python baseball/pitch_quality/develop.py windows                     # about 90 minutes on 4 cores
    python baseball/pitch_quality/develop.py targets --window 2

Writes output/development_windows.json / output/development_targets.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from features import FEATURE_SETS, WOBA_SCALE, build_features
from train import MIN_PITCHES, SEED, fit_kinds, pitcher_season, predict_all, xrv100  # noqa: F401  (MIN_PITCHES: query strings)

from backtest.asof import Loader
from sparvi.config import baseball

CFG = baseball()
OUT = Path(__file__).resolve().parent / "output"
BOOT = 2000
KEY = ["game_pk", "at_bat_number", "pitch_number"]
TARGETS = ("rv_x", "cn", "cn_count")
OUTCOME_CLASS = {
    "ball": "ball", "blocked_ball": "ball", "pitchout": "ball", "automatic_ball": "ball",
    "called_strike": "called_strike", "automatic_strike": "called_strike",
    "swinging_strike": "swinging_strike", "swinging_strike_blocked": "swinging_strike", "missed_bunt": "swinging_strike",
    "foul_tip": "swinging_strike",
    "foul": "foul", "foul_bunt": "foul", "bunt_foul_tip": "foul", "foul_pitchout": "foul",
    "hit_by_pitch": "hit_by_pitch", "hit_into_play": "in_play",
}


CACHE = OUT / "dev_cache"       # pitcher-level grades per fit (gitignored)


def _code_key() -> str:
    """Changes when the training or feature code does, so cached fits never outlive the code that made them."""
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for f in ("train.py", "features.py", "develop.py"):
        h.update((here / f).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:10]


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
    """Model features plus the raw columns the context-neutral targets need (joined on the pitch key)."""
    raw = loader.load(f"statcast_mlb_{season}", as_of)
    f = build_features(raw, heights)
    extra = raw[[*KEY, "description", "estimated_woba_using_speedangle", "woba_value"]].drop_duplicates(KEY)
    return f.merge(extra, on=KEY, how="left", validate="one_to_one")


def context_neutral(tr: pd.DataFrame, by_count: bool) -> pd.Series:
    """The target with game context removed, from the training pitches alone: the average run value (pitcher's view)
    of the pitch's outcome class, optionally within its count; a ball in play moves from its class average by its
    expected wOBA relative to the average ball in play (actual wOBA where Statcast has no expected value)."""
    cls = tr["description"].map(OUTCOME_CLASS)
    keys = [cls, tr["balls"], tr["strikes"]] if by_count else [cls]
    base = tr["rv"].groupby(keys).transform("mean")
    xw = tr["estimated_woba_using_speedangle"].fillna(tr["woba_value"])
    bip = cls.eq("in_play") & xw.notna()
    mean_xw = xw[bip].groupby([k[bip] for k in keys]).transform("mean")
    out = base.copy()
    out[bip] = base[bip] - (xw[bip] - mean_xw) / WOBA_SCALE
    return out.fillna(tr["rv_x"])          # an outcome description outside the table: the published target


FIT_COLS = sorted({c for cols in FEATURE_SETS.values() for c in cols}) + ["rv_x", "pitcher", "group"]
TARGET_COLS = ["rv", "description", "estimated_woba_using_speedangle", "woba_value"]      # balls, strikes are features


def _train(loader, seasons, as_of, heights, target: str) -> pd.DataFrame:
    """Training rows with only the columns a fit reads (a season's full frame is several times larger, and three of
    them at once ran the machine out of memory). Dropping unused columns doesn't change a fit."""
    tr = pd.concat([_pitches(loader, s, as_of, heights)[FIT_COLS + TARGET_COLS] for s in seasons], ignore_index=True)
    if target != "rv_x":
        tr["rv_x"] = context_neutral(tr, by_count=target == "cn_count")      # fit_kinds trains on the rv_x column
    return tr[FIT_COLS]


def run(variants: dict[int, dict[str, tuple[list[int], str]]], ref: str, kinds: list[str]) -> dict:
    """variants: outcome season -> {name: (training seasons, target)}; `ref` is the name the others are compared to."""
    loader = Loader()
    assert all(t < CFG.season for t in variants), "a development fold may not use the current season's outcomes"
    h = loader.load("batter_heights", f"{CFG.season}-03-01")
    heights = (h.set_index("batter")["height_in"] / 12).astype(float)
    rng = np.random.default_rng(SEED)
    res = {"kinds": kinds, "reference": ref, "min_pitches": MIN_PITCHES, "bootstrap_samples": BOOT, "folds": {}}
    pooled = []
    for t, vs in variants.items():
        graded = _pitches(loader, t - 1, f"{t}-03-01", heights)
        outcome = pitcher_season(_pitches(loader, t, f"{t + 1}-03-01", heights)).query("pitches >= @MIN_PITCHES")
        fold = {"grade_season": t - 1, "outcome_season": t, "variants": {}}
        cols = {}
        for name, (seasons, target) in vs.items():
            # Each fit takes 10-30 minutes: keep its pitcher-level grades, so a stopped run resumes where it was.
            tag = f"{t}_{'-'.join(map(str, seasons))}_{target}_{'-'.join(kinds)}_{_code_key()}"
            f_ps, f_info = CACHE / f"{tag}.parquet", CACHE / f"{tag}.json"
            if f_ps.exists() and f_info.exists():
                ps, info = pd.read_parquet(f_ps), json.loads(f_info.read_text(encoding="utf-8"))
            else:
                tr = _train(loader, seasons, f"{t - 1}-03-01", heights, target)
                models = fit_kinds(tr, kinds=kinds)
                d = graded.copy()
                predict_all(models, d)
                ps = pitcher_season(d).query("pitches >= @MIN_PITCHES").merge(
                    outcome[["pitcher", "rv100"]].rename(columns={"rv100": "y"}), on="pitcher")
                info = {"train_pitches": int(len(tr)),
                        "trees": {f"{k}/{g}": [m.best_iteration for m in bag] for k in kinds for g, bag in models[k].items()}}
                del tr, models, d
                CACHE.mkdir(exist_ok=True)
                ps.to_parquet(f_ps, index=False)
                f_info.write_text(json.dumps(info), encoding="utf-8")
            for k in kinds:
                cols[f"{name}|{k}"] = ps.set_index("pitcher")[xrv100(k)]
            fold["variants"][name] = {"train_seasons": seasons, "target": target, "n": int(len(ps)),
                                      "r": {k: float(ps[xrv100(k)].corr(ps["y"])) for k in kinds},
                                      "r_last_season_results": float(ps["rv100"].corr(ps["y"])), **info}
            print(f"  outcome {t}, {name} (train {seasons}, target {target}): "
                  + ", ".join(f"{k} r {fold['variants'][name]['r'][k]:.4f}" for k in kinds)
                  + f"; last season's results {fold['variants'][name]['r_last_season_results']:.4f}", flush=True)
        wide = (pd.DataFrame(cols).join(outcome.set_index("pitcher")["rv100"].rename("y"), how="inner").dropna()
                .rename_axis("pitcher").reset_index())
        pooled.append(wide.assign(fold=t))
        fold["edges"] = _edges(wide, [n for n in vs if n != ref], ref, kinds, rng)
        res["folds"][str(t)] = fold
    allf = pd.concat(pooled, ignore_index=True)
    common = [n for n in next(iter(variants.values())) if n != ref and all(n in f["variants"] for f in res["folds"].values())]
    res["pooled"] = {"n": int(len(allf)), "edges": _edges(allf, common, ref, kinds, rng, by_fold=True)}
    return res


def _r(d: pd.DataFrame, col: str, by_fold: bool) -> float:
    """Correlation with the outcome; pooled over folds as the mean of the per-fold correlations."""
    if not by_fold:
        return float(d[col].corr(d["y"]))
    return float(np.mean([g[col].corr(g["y"]) for _, g in d.groupby("fold")]))


def _edges(d: pd.DataFrame, names: list[str], ref: str, kinds: list[str], rng, by_fold: bool = False) -> dict:
    out = {}
    groups = list(d.groupby("pitcher").indices.values())      # a pitcher's rows (one per fold) travel together
    for n in names:
        for k in kinds:
            a, b = f"{n}|{k}", f"{ref}|{k}"
            est = _r(d, a, by_fold) - _r(d, b, by_fold)
            draws = []
            for _ in range(BOOT):
                s = d.iloc[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])]
                draws.append(_r(s, a, by_fold) - _r(s, b, by_fold))
            out[f"{n}_minus_{ref}|{k}"] = {"est": est, "ci90": [float(np.percentile(draws, 5)), float(np.percentile(draws, 95))]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["windows", "targets"])
    ap.add_argument("--window", default="2", help="targets: the training window every target is fit on")
    ap.add_argument("--kinds", nargs="+", default=["stuff"], choices=["stuff", "loc", "pitching"])
    a = ap.parse_args()
    F = folds()
    if a.what == "windows":
        variants, ref = {t: {n: (s, "rv_x") for n, s in w.items()} for t, w in F.items()}, "1"
    else:
        variants, ref = {t: {tg: (w[a.window], tg) for tg in TARGETS} for t, w in F.items() if a.window in w}, "rv_x"
    r = run(variants, ref, a.kinds)
    (OUT / f"development_{a.what}.json").write_text(json.dumps(r, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"pooled": r["pooled"], **{t: f["edges"] for t, f in r["folds"].items()}}, indent=2))
