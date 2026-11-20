"""Development folds for Location+ (LOC-TYPE): does a location grade that knows the pitch predict, where the published
one (location and count only) doesn't?

DEVELOPMENT ONLY, on the folds of develop.py: train on the two seasons ending at t-2, grade season t-1 pitchers,
compare with season t run value per 100 pitches, for outcome seasons before the current one. Nothing here reads an
outcome from the current season, so a variant chosen here can still be tested on it. Needs the MLB history seasons
(`python ingest/baseball/pull.py --history`) and develop.py's cached Stuff+ fits for the same folds
(`python baseball/pitch_quality/develop.py windows`).

Variants (each a grade per pitcher-season, higher = better for the pitcher):
    loc              the published Location+ features: where the pitch crossed the plate, the count, same-handed batter
    loc_type         the same plus the pitch type, so a four-seamer up and a sinker up are different pitches
    loc_given_shape  the value of the location given the pitch: a model on shape, count, hand and location minus a
                     model on everything but location (both trained the same way), so the stuff is held fixed
    command          no model: how tightly a pitcher's pitches cluster around his own average spot for that pitch
                     type, batter side and count situation (minus the mean distance, against the league's for the
                     pitch type). The seasons have no record of the intended target; his own average stands in.

For each: r with next-season RV/100; r with the same season's Stuff+ grade; and the partial correlation with
next-season RV/100 given Stuff+ (does it add anything to Stuff+?). 90% intervals from a bootstrap over pitchers,
per fold and pooled (the mean of the folds).

    python baseball/pitch_quality/develop_loc.py            # about 30 minutes on 4 cores

Writes output/development_location.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from develop import BOOT, CACHE, CFG, OUT, _code_key, _pitches, folds
from features import FEATURES, LOC_FEATURES, PITCHING_FEATURES
from train import GROUPS, MIN_PITCHES, N_BAGS, SEED, bag_folds, fit_group, pitcher_season, predict  # noqa: F401  (MIN_PITCHES: query strings)

from backtest.asof import Loader

LR = 0.03                                   # Location+'s published learning rate
WINDOW = "2"                                # PQ-WINDOW: two training seasons
PITCH_TYPES = ["FF", "SI", "FC", "SL", "ST", "CU", "KC", "SV", "CS", "CH", "FS", "FO", "SC"]
NO_LOC = FEATURES + ["balls", "strikes", "same_hand"]
MODELS = {"loc": LOC_FEATURES, "loc_type": LOC_FEATURES + ["ptype"], "full": PITCHING_FEATURES, "no_loc": NO_LOC}
VARIANTS = ["loc", "loc_type", "loc_given_shape", "command"]
ZONE_FT = 1.9                               # loc_z is in zone heights; about 1.9 ft, so both axes are in feet
CELL_MIN = 20                               # pitches a pitcher needs in a cell for its spread to count


def _ptype(d: pd.DataFrame) -> pd.DataFrame:
    """Pitch type as a category with one fixed list, so the codes mean the same pitch in training and grading."""
    d["ptype"] = pd.Categorical(d["pitch_type"], categories=PITCH_TYPES)
    return d


def _fit(tr: pd.DataFrame, feats: list[str]) -> dict:
    """{group: bag}, trained as train.fit_kinds trains Location+ (pitcher folds, early stopping on the held-out fold)."""
    fold = bag_folds(tr["pitcher"], SEED, N_BAGS)
    return {g: [fit_group(tr[tr["group"].eq(g) & fold.ne(f)], tr[tr["group"].eq(g) & fold.eq(f)], feats,
                          seed=SEED * 100 + f, learning_rate=LR) for f in range(N_BAGS)] for g in GROUPS}


def command(d: pd.DataFrame) -> pd.Series:
    """Per pitch: minus the distance (feet) from the pitcher's own average location for that pitch type, batter side
    and count situation, relative to the league's mean distance for the pitch type. NaN in thin cells."""
    sit = np.select([d["strikes"].eq(2), d["balls"] > d["strikes"]], ["two_strikes", "behind"], default="other")
    cell = [d["pitcher"], d["pitch_type"], d["same_hand"], pd.Series(sit, index=d.index)]
    z = d["loc_z"] * ZONE_FT
    dx, dz = d["loc_x"] - d["loc_x"].groupby(cell).transform("mean"), z - z.groupby(cell).transform("mean")
    dist = np.sqrt(dx ** 2 + dz ** 2).where(d["loc_x"].groupby(cell).transform("size") >= CELL_MIN)
    return -(dist - dist.groupby(d["pitch_type"]).transform("mean"))


def _fold(loader: Loader, t: int, seasons: list[int], heights: pd.Series) -> pd.DataFrame:
    """Pitcher-level grades of season t-1 for every variant, with Stuff+'s grade and the season-t outcome."""
    f_ps = CACHE / f"loc_{t}_{'-'.join(map(str, seasons))}_{_loc_key()}.parquet"
    if f_ps.exists():
        return pd.read_parquet(f_ps)
    keep = sorted(set(PITCHING_FEATURES)) + ["pitch_type", "pitcher", "group", "rv_x"]
    tr = _ptype(pd.concat([_pitches(loader, s, f"{t - 1}-03-01", heights)[keep] for s in seasons], ignore_index=True))
    d = _ptype(_pitches(loader, t - 1, f"{t}-03-01", heights))
    for name, feats in MODELS.items():
        # Each model's predictions are kept as it finishes (private, gitignored), so a stopped run resumes there.
        f_p = CACHE / f"loc_{t}_{'-'.join(map(str, seasons))}_{name}_{_loc_key()}.parquet"
        if not f_p.exists():
            CACHE.mkdir(exist_ok=True)
            pd.DataFrame({"p": predict(_fit(tr, feats), d, feats)}).to_parquet(f_p, index=False)
            print(f"  outcome {t}: fit {name}", flush=True)
        d[f"p_{name}"] = pd.read_parquet(f_p)["p"].to_numpy()
    del tr
    d["v_loc"], d["v_loc_type"], d["v_loc_given_shape"], d["v_command"] = d["p_loc"], d["p_loc_type"], d["p_full"] - d["p_no_loc"], command(d)
    ps = pitcher_season(d).query("pitches >= @MIN_PITCHES")
    grades = (100 * d.groupby("pitcher")[[f"v_{v}" for v in VARIANTS]].mean()).rename(columns=lambda c: c[2:]).reset_index()
    outcome = pitcher_season(_pitches(loader, t, f"{t + 1}-03-01", heights)).query("pitches >= @MIN_PITCHES")
    stuff = pd.read_parquet(_stuff_file(t, seasons))[["pitcher", "xrv100"]].rename(columns={"xrv100": "stuff"})
    ps = (ps[["pitcher", "pitches", "rv100"]].merge(grades, on="pitcher").merge(stuff, on="pitcher")
          .merge(outcome[["pitcher", "rv100"]].rename(columns={"rv100": "y"}), on="pitcher"))
    CACHE.mkdir(exist_ok=True)
    ps.to_parquet(f_ps, index=False)
    return ps


def _stuff_file(t: int, seasons: list[int]):
    return CACHE / f"{t}_{'-'.join(map(str, seasons))}_rv_x_stuff_{_code_key()}.parquet"


def _loc_key() -> str:
    import hashlib
    from pathlib import Path
    return hashlib.sha256((_code_key() + Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n")).encode()).hexdigest()[:10]


def _partial(d: pd.DataFrame, col: str) -> float:
    """Correlation of the grade with next-season RV/100 once Stuff+ is taken out of both."""
    rx, ry, rxy = d[col].corr(d["stuff"]), d["y"].corr(d["stuff"]), d[col].corr(d["y"])
    return float((rxy - rx * ry) / np.sqrt((1 - rx ** 2) * (1 - ry ** 2)))


STATS = {"r_next_rv": lambda d, c: float(d[c].corr(d["y"])), "r_with_stuff": lambda d, c: float(d[c].corr(d["stuff"])),
         "partial_r_given_stuff": _partial}


def _summ(d: pd.DataFrame, rng, by_fold: bool) -> dict:
    def stat(x, fn, c):
        return float(np.mean([fn(g, c) for _, g in x.groupby("fold")])) if by_fold else fn(x, c)
    groups = list(d.groupby("pitcher").indices.values())          # a pitcher's rows (one per fold) travel together
    draws = [d.iloc[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])] for _ in range(BOOT)]
    out = {}
    for v in VARIANTS:
        out[v] = {}
        for name, fn in STATS.items():
            b = [stat(s, fn, v) for s in draws]
            out[v][name] = {"est": stat(d, fn, v), "ci90": [float(np.percentile(b, 5)), float(np.percentile(b, 95))]}
    return out


def main() -> None:
    loader = Loader()
    h = loader.load("batter_heights", f"{CFG.season}-03-01")
    heights = (h.set_index("batter")["height_in"] / 12).astype(float)
    rng = np.random.default_rng(SEED)
    F = {t: w[WINDOW] for t, w in folds().items() if WINDOW in w}
    assert all(t < CFG.season for t in F), "a development fold may not use the current season's outcomes"
    missing = [str(_stuff_file(t, s).name) for t, s in F.items() if not _stuff_file(t, s).exists()]
    if missing:
        raise SystemExit(f"run develop.py windows first: no cached Stuff+ fit {missing}")
    res = {"window": WINDOW, "min_pitches": MIN_PITCHES, "bootstrap_samples": BOOT, "variants": VARIANTS, "folds": {}}
    frames = []
    for t, seasons in F.items():
        ps = _fold(loader, t, seasons, heights).dropna(subset=VARIANTS)
        frames.append(ps.assign(fold=t))
        res["folds"][str(t)] = {"train_seasons": seasons, "grade_season": t - 1, "n": int(len(ps)),
                                "r_stuff_next_rv": float(ps["stuff"].corr(ps["y"])),
                                "r_last_season_results": float(ps["rv100"].corr(ps["y"])), **_summ(ps, rng, False)}
        print(f"outcome {t} (n {len(ps)}): " + "; ".join(
            f"{v} r {res['folds'][str(t)][v]['r_next_rv']['est']:+.3f}, given Stuff+ {res['folds'][str(t)][v]['partial_r_given_stuff']['est']:+.3f}"
            for v in VARIANTS), flush=True)
    allf = pd.concat(frames, ignore_index=True)
    res["pooled"] = {"n": int(len(allf)), **_summ(allf, rng, True)}
    (OUT / "development_location.json").write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(res["pooled"], indent=2))


if __name__ == "__main__":
    main()
