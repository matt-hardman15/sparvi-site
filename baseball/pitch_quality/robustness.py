"""Fully out-of-sample version of the year-over-year test.

train.py fits on 2024-2025, so its 2025 Stuff+ is in-sample. Here the model
sees only 2024; 2025 Stuff+ is then used to predict 2026 run value, so neither
side of the comparison touched training.
"""
import json

import numpy as np

from train import MIN_PITCHES, OUT, fit_group, load, pitcher_season, predict

tr = load("mlb", 2024)
rng = np.random.default_rng(7)
val_p = set(rng.choice(tr["pitcher"].unique(), size=int(0.15 * tr["pitcher"].nunique()), replace=False))
is_val = tr["pitcher"].isin(val_p)
models = {g: fit_group(tr[tr.group.eq(g) & ~is_val], tr[tr.group.eq(g) & is_val])
          for g in ["fastball", "breaking", "offspeed"]}

d25, d26 = load("mlb", 2025), load("mlb", 2026)
d25["xrv"], d26["xrv"] = predict(models, d25), predict(models, d26)
a = pitcher_season(d25).query("pitches >= @MIN_PITCHES")
b = pitcher_season(d26).query("pitches >= @MIN_PITCHES")
yy = a.merge(b, on="pitcher", suffixes=("_25", "_26"))

# Bootstrap the gap between the two correlations.
gaps = []
for _ in range(2000):
    s = yy.sample(len(yy), replace=True, random_state=rng.integers(1e9))
    gaps.append(s["xrv100_25"].corr(s["rv100_26"]) - s["rv100_25"].corr(s["rv100_26"]))
res = {
    "n": int(len(yy)),
    "r_stuff25_to_rv26": float(yy["xrv100_25"].corr(yy["rv100_26"])),
    "r_rv25_to_rv26": float(yy["rv100_25"].corr(yy["rv100_26"])),
    "gap_ci90": [float(np.percentile(gaps, 5)), float(np.percentile(gaps, 95))],
}
(OUT / "robustness.json").write_text(json.dumps(res, indent=2))
yy[["pitcher", "name_25", "team_26", "xrv100_25", "rv100_25", "rv100_26", "pitches_25", "pitches_26"]].to_csv(OUT / "yoy_pairs.csv", index=False)
print(json.dumps(res, indent=2))
