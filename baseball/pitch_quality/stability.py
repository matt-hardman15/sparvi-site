"""How much do Stuff+, Location+, and Pitching+ depend on the training seed? (model noise, v2.0.0)

Retrains every model exactly as train.py does but with a different seed (SEED + 1: other bag folds,
other LightGBM seeds), grades the current MLB season, and compares each pitcher's grades with the
published run (pitch_quality/output/pitch_predictions_mlb_{season}.parquet, written by train.py), on
the frozen scale. Pitchers with MIN_STABLE+ pitches.

Writes output/stability.json: per grade, the mean and largest per-pitcher difference and the share of
pitchers that moved more than 1 and 2 points. Run it after train.py whenever a pitch-quality model gets a
new version (as expensive as training, so not on every refresh). Expected for v2.0.0 (Nov 14, 2026):
Stuff+ ~0.4 mean, Pitching+ ~0.4 mean; before bagging the same test gave 3.0 and 2.1 (max 64 and 42).

    python baseball/pitch_quality/stability.py
"""
import json

import numpy as np
import pandas as pd
from train import CFG, GRADE, KINDS, OUT, SEED, XRV, fit_kinds, load, predict_all

MIN_STABLE = 300


def grades(df: pd.DataFrame, ref: dict, cols: dict) -> pd.DataFrame:
    g = df.groupby("pitcher")
    out = pd.DataFrame({"pitches": g.size()})
    for k, c in cols.items():
        out[k] = 100 + 10 * (100 * g[c].mean() - ref[k]["mean"]) / ref[k]["sd"]
    return out


def main() -> None:
    Y = CFG.season
    ref = json.loads((OUT / "scale_reference.json").read_text(encoding="utf-8"))
    published = pd.read_parquet(OUT / f"pitch_predictions_mlb_{Y}.parquet")
    a = grades(published, ref, {k: ("xrv_stuff" if k == "stuff" else XRV[k]) for k in KINDS})

    tr = pd.concat([load("mlb", y) for y in CFG.train_seasons], ignore_index=True)
    ho = load("mlb", Y)
    predict_all(fit_kinds(tr, seed=SEED + 1), ho)
    b = grades(ho, ref, {k: XRV[k] for k in KINDS})

    q = a.index[a["pitches"] >= MIN_STABLE]
    res = {"season": Y, "seeds": [SEED, SEED + 1], "min_pitches": MIN_STABLE, "n": int(len(q))}
    for k in KINDS:
        d = (a.loc[q, k] - b.reindex(q)[k]).abs()
        res[GRADE[k]] = {"mean_abs": round(float(d.mean()), 3), "p95": round(float(np.percentile(d, 95)), 3),
                         "max": round(float(d.max()), 3), "share_over_1": round(float((d > 1).mean()), 4),
                         "share_over_2": round(float((d > 2).mean()), 4)}
    (OUT / "stability.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
