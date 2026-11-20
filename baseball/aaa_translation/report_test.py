"""What the pre-registered test of the small-sample translation (M5-EB, candidate 2.0.0) promised to report beside
its gate, on the test season's pairs. REPORTED, NOT GATED, and run only after the official backtest: this reads the
current season's MLB lines, which develop.py never does.

    python baseball/aaa_translation/report_test.py

Writes output/test_report.json: the published line against the candidate by Triple-A sample (the gate's 50-149 pool,
and 150+ where the published line stays), by pair type, 80% band coverage by sample (the published band against one
sized by the Triple-A sample), and the step between the two lines for players near 150.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from develop import (
    BOOT,
    CFG,
    COLS,
    METRICS,
    MIN_AAA,
    OUT,
    SEED,
    Loader,
    _coverage,
    _gate_candidates,
    ages,
    lines_from,
    pairs,
    predict_metric,
    reliability,
)

T = CFG.season


def main() -> None:
    loader = Loader()
    S = {"aaa": [s for s in CFG.aaa_history_seasons + CFG.aaa_seasons], "mlb": [s for s in CFG.mlb_seasons if s >= min(CFG.aaa_history_seasons)]}
    frames = []
    for lvl, ss in S.items():
        for s in ss:
            d = loader.load(f"statcast_{lvl}_{s}", f"{s + 1}-03-01", COLS)
            frames.append(d.assign(level=lvl.upper(), game_date=pd.to_datetime(d["game_date"]).dt.date))
    lines = lines_from(pd.concat(frames, ignore_index=True), loader.load("batter_heights", f"{T + 1}-03-01"))
    born = ages()
    res = {"test_season": T, "seasons": S, "bootstrap_samples": BOOT,
           "note": "reported, not gated; the gate's own numbers are in models/baseball/aaa_translation/backtest/"}
    for side, ms in METRICS.items():
        P = pairs(lines, side, born)
        tr, te = P[P["outcome_season"] < T], P[P["outcome_season"] == T]
        rel = reliability(lines, side, T)
        per = {m: predict_metric(tr, te, m, n, rel[m]).assign(fold=T, metric=m, scale=float(tr[f"{m}_mlb"].std())) for m, n in ms.items()}
        long = pd.concat(per.values(), ignore_index=True)
        R = {"n_train": int(len(tr)), "n_test": int(len(te)), "missing_age": int(te["age"].isna().sum()),
             "by_pool": _gate_candidates(long, np.random.default_rng(SEED + 2))[str(T)],
             "by_pair_type_50_149": {}, "bands": {}}
        for c in ("across", "within"):
            q = long[(long["cohort"] == c) & (long["pa_aaa"] < MIN_AAA)]
            r = {v: float(np.sqrt((((q[v] - q["y"]) / q["scale"]) ** 2).mean())) for v in ("raw", "eb_wide")}
            R["by_pair_type_50_149"][c] = {"players": int(q["player_id"].nunique()), "published": r["raw"], "candidate": r["eb_wide"],
                                           "gain_pct": float(100 * (r["raw"] - r["eb_wide"]) / r["raw"])}
        cov = {m: _coverage(p) for m, p in per.items()}
        for bucket in ("50-149", "150-299", "300+"):
            rows = [c[bucket] for c in cov.values() if bucket in c]
            if rows:
                R["bands"][bucket] = {"pairs": rows[0]["n"], "published": float(np.mean([r["raw"]["coverage"] for r in rows])),
                                      "sized": float(np.mean([r["sized"]["coverage"] for r in rows])),
                                      "by_metric_range_sized": [float(min(r["sized"]["coverage"] for r in rows)), float(max(r["sized"]["coverage"] for r in rows))]}
        near = long[long["pa_aaa"].between(MIN_AAA - 25, MIN_AAA + 25)]
        step = ((near["eb_wide"] - near["raw"]) / near["scale"]).abs()
        R["step_at_150"] = {"players": int(near["player_id"].nunique()), "rows": int(len(near)), "mean_abs_sd_units": float(step.mean()),
                            "p90_abs_sd_units": float(step.quantile(0.9)),
                            "what": "for pairs with 125-175 PA/BF in Triple-A, |candidate - published line| in units of the metric's MLB spread"}
        res[side] = R
        print(side, json.dumps({k: R[k] for k in ("by_pool", "by_pair_type_50_149", "bands", "step_at_150")}, indent=1))
    (OUT / "test_report.json").write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
