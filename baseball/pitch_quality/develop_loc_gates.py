"""How each candidate pass gate for a new Location+ would have come out on the development folds (LOC-TYPE).

DEVELOPMENT ONLY: reads the pitcher-level grades develop_loc.py cached for outcome seasons before the current one
(no training here, a few seconds). The pre-registered test has one fold of about 320 pitchers, so each gate is
judged per fold, the size it will be run at, with the harness's paired bootstrap over pitchers (90% interval of
the difference in r with next-season RV/100).

Candidates, for the variant chosen on these folds (CHOSEN):
    vs_published    its r minus the published Location+ features' r (does it beat what it would replace?)
    adds_to_stuff   r of Stuff+ and the variant combined, minus Stuff+'s r. The combination's weights are fit on the
                    other development fold, never the one being scored
    vs_last_season  its r minus the r of last season's results (the Stuff+ gate's baseline)
and, without a baseline, its own r and its partial r given Stuff+ (from development_location.json).

    python baseball/pitch_quality/develop_loc_gates.py      # after develop_loc.py

Writes output/development_location_gates.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from develop import BOOT, CACHE, OUT, folds
from develop_loc import VARIANTS, WINDOW, _loc_key
from train import SEED

CHOSEN = "loc_given_shape"


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def _weights(d: pd.DataFrame) -> np.ndarray:
    """Least-squares weights of next-season RV/100 on standardized Stuff+ and the chosen variant."""
    return np.linalg.lstsq(np.column_stack([_z(d["stuff"]), _z(d[CHOSEN])]), d["y"] - d["y"].mean(), rcond=None)[0]


def _edge(d: pd.DataFrame, a: str, b: str, rng) -> dict:
    def gap(s):
        return float(s[a].corr(s["y"]) - s[b].corr(s["y"]))
    draws = [gap(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(BOOT)]
    lo, hi = float(np.percentile(draws, 5)), float(np.percentile(draws, 95))
    return {"r": float(d[a].corr(d["y"])), "r_baseline": float(d[b].corr(d["y"])), "edge": gap(d), "ci90": [lo, hi],
            "strict_gate": bool(lo > 0)}


def main() -> None:
    F = {t: w[WINDOW] for t, w in folds().items() if WINDOW in w}
    frames = {t: pd.read_parquet(CACHE / f"loc_{t}_{'-'.join(map(str, s))}_{_loc_key()}.parquet")
              .dropna(subset=VARIANTS).reset_index(drop=True) for t, s in F.items()}
    dev = json.loads((OUT / "development_location.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    res = {"variant": CHOSEN, "bootstrap_samples": BOOT, "folds": {}}
    for t, d in frames.items():
        w = _weights(pd.concat([o for u, o in frames.items() if u != t], ignore_index=True))
        d = d.assign(combined=w[0] * _z(d["stuff"]) + w[1] * _z(d[CHOSEN]))
        own = dev["folds"][str(t)][CHOSEN]
        res["folds"][str(t)] = {
            "n": int(len(d)), "combination_weights": {"stuff": float(w[0]), CHOSEN: float(w[1])},
            "vs_published": _edge(d, CHOSEN, "loc", rng),
            "adds_to_stuff": _edge(d, "combined", "stuff", rng),
            "vs_last_season": _edge(d, CHOSEN, "rv100", rng),
            "r_alone": {**own["r_next_rv"], "strict_gate": own["r_next_rv"]["ci90"][0] > 0},
            "partial_r_given_stuff": {**own["partial_r_given_stuff"], "strict_gate": own["partial_r_given_stuff"]["ci90"][0] > 0}}
    (OUT / "development_location_gates.json").write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")
    for t, f in res["folds"].items():
        print(f"outcome {t} (n {f['n']}): " + "; ".join(
            f"{k} {v.get('edge', v.get('est')):+.3f} [{v['ci90'][0]:+.3f}, {v['ci90'][1]:+.3f}] {'pass' if v['strict_gate'] else 'fail'}"
            for k, v in f.items() if isinstance(v, dict) and "ci90" in v))


if __name__ == "__main__":
    main()
