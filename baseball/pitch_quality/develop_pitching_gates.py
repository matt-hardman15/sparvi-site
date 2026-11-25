"""How each candidate pass gate for Pitching+ would have come out on the development folds (A5-port-PITCHING).

DEVELOPMENT ONLY: reads the pitch-level predictions develop_loc.py cached for outcome seasons before the current
one (its `full` model is Pitching+ as train.py trains it: shape, count, batter side and location, rate 0.03 in bags
of 10). No training here. The pre-registered test has one fold of about 320 pitchers, so each gate is judged per
fold, with a paired bootstrap over pitchers (90% interval).

Candidates:
    vs_last_season   its r with next-season RV/100 minus the r of last season's results (Stuff+'s gate)
    vs_stuff         its r minus Stuff+'s r
    adds_to_stuff    r of Stuff+ and Pitching+ combined, minus Stuff+'s r. The combination's weights are fit on the
                     other development fold, never the one being scored
    partial_r_given_stuff   its correlation with next-season RV/100 once Stuff+ is taken out of both

    python baseball/pitch_quality/develop_pitching_gates.py [CACHE_KEY]     # after develop_loc.py

CACHE_KEY names the cached grades to read when they were made by an earlier revision of the training code (the key
is a hash of the code; a change that doesn't alter the fits still changes it). Default: the current code's key.
Writes output/development_pitching_gates.json (aggregates only).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from develop import BOOT, CACHE, CFG, OUT, _pitches, folds
from develop_loc import WINDOW, _loc_key
from develop_loc_gates import _edge, _z
from train import SEED

from backtest.asof import Loader
from backtest.metrics import partial_r


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else _loc_key()
    loader = Loader()
    h = loader.load("batter_heights", f"{CFG.season}-03-01")
    heights = (h.set_index("batter")["height_in"] / 12).astype(float)
    F = {t: w[WINDOW] for t, w in folds().items() if WINDOW in w}
    assert all(t < CFG.season for t in F), "a development fold may not use the current season's outcomes"
    frames = {}
    for t, s in F.items():
        tag = f"loc_{t}_{'-'.join(map(str, s))}"
        ps = pd.read_parquet(CACHE / f"{tag}_{key}.parquet")           # pitcher level: Stuff+, last season, the outcome
        d = _pitches(loader, t - 1, f"{t}-03-01", heights)
        p = pd.read_parquet(CACHE / f"{tag}_full_{key}.parquet")["p"].to_numpy()
        assert len(p) == len(d), "the cached predictions are not for these pitches"
        grade = (100 * pd.Series(p).groupby(d["pitcher"].to_numpy()).mean()).rename("pitching").rename_axis("pitcher").reset_index()
        frames[t] = ps.merge(grade, on="pitcher").dropna(subset=["stuff", "pitching", "rv100", "y"]).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    res = {"model": "pitching", "window": WINDOW, "seed": SEED, "bootstrap_samples": BOOT, "grades_cache_key": key, "folds": {}}
    for t, d in frames.items():
        o = pd.concat([x for u, x in frames.items() if u != t], ignore_index=True)
        w = np.linalg.lstsq(np.column_stack([_z(o["stuff"]), _z(o["pitching"])]), o["y"] - o["y"].mean(), rcond=None)[0]
        d = d.assign(combined=w[0] * _z(d["stuff"]) + w[1] * _z(d["pitching"]))
        draws = [partial_r(s["pitching"], s["y"], s["stuff"]) for s in (d.iloc[rng.integers(0, len(d), len(d))] for _ in range(BOOT))]
        lo, hi = float(np.percentile(draws, 5)), float(np.percentile(draws, 95))
        res["folds"][str(t)] = {
            "train_seasons": F[t], "grade_season": t - 1, "n": int(len(d)),
            "r": {"pitching": float(d["pitching"].corr(d["y"])), "stuff": float(d["stuff"].corr(d["y"])),
                  "last_season": float(d["rv100"].corr(d["y"]))},
            "r_pitching_with_stuff": float(d["pitching"].corr(d["stuff"])),
            "combination_weights": {"stuff": float(w[0]), "pitching": float(w[1])},
            "vs_last_season": _edge(d, "pitching", "rv100", rng),
            "vs_stuff": _edge(d, "pitching", "stuff", rng),
            "adds_to_stuff": _edge(d, "combined", "stuff", rng),
            "partial_r_given_stuff": {"est": partial_r(d["pitching"], d["y"], d["stuff"]), "ci90": [lo, hi], "strict_gate": bool(lo > 0)}}
    (OUT / "development_pitching_gates.json").write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")
    for t, f in res["folds"].items():
        print(f"outcome {t} (n {f['n']}): " + "; ".join(
            f"{k} {v.get('edge', v.get('est')):+.3f} [{v['ci90'][0]:+.3f}, {v['ci90'][1]:+.3f}] {'pass' if v['strict_gate'] else 'fail'}"
            for k, v in f.items() if isinstance(v, dict) and "ci90" in v))


if __name__ == "__main__":
    main()
