"""Fully out-of-sample version of the year-over-year test.

train.py fits on every MLB season before the current one, so its last-season Stuff+ is
in-sample. Here the model sees only the season before that; last season's Stuff+ is
then used to predict this season's run value, so neither side touched training.
Seasons come from sparvi.toml (with season = 2026: train 2024, predict 2025 -> 2026).
"""
import json

import numpy as np
from train import MIN_PITCHES, OUT, SEED, fit_group, load, pitcher_season, predict  # noqa: F401  (MIN_PITCHES is used inside query strings)

from sparvi.config import RAW, baseball
from sparvi.provenance import provenance

CFG = baseball()
Y, P = CFG.season, CFG.prev
TRAIN = P - 1
BOOT, BOOT_SEED = 2000, SEED

tr = load("mlb", TRAIN)
rng = np.random.default_rng(BOOT_SEED)
val_p = set(rng.choice(tr["pitcher"].unique(), size=int(0.15 * tr["pitcher"].nunique()), replace=False))
is_val = tr["pitcher"].isin(val_p)
models = {g: fit_group(tr[tr.group.eq(g) & ~is_val], tr[tr.group.eq(g) & is_val])
          for g in ["fastball", "breaking", "offspeed"]}

d_prev, d_cur = load("mlb", P), load("mlb", Y)
d_prev["xrv"], d_cur["xrv"] = predict(models, d_prev), predict(models, d_cur)
a = pitcher_season(d_prev).query("pitches >= @MIN_PITCHES")
b = pitcher_season(d_cur).query("pitches >= @MIN_PITCHES")
yy = a.merge(b, on="pitcher", suffixes=("_prev", "_cur"))

# Bootstrap the gap between the two correlations.
gaps = []
for _ in range(BOOT):
    s = yy.sample(len(yy), replace=True, random_state=rng.integers(1e9))
    gaps.append(s["xrv100_prev"].corr(s["rv100_cur"]) - s["rv100_prev"].corr(s["rv100_cur"]))
res = {
    "train_season": TRAIN, "predictor_season": P, "outcome_season": Y,
    "n": int(len(yy)),
    "r_stuff_prev_to_rv_cur": float(yy["xrv100_prev"].corr(yy["rv100_cur"])),
    "r_rv_prev_to_rv_cur": float(yy["rv100_prev"].corr(yy["rv100_cur"])),
    "gap_ci90": [float(np.percentile(gaps, 5)), float(np.percentile(gaps, 95))],
    "bootstrap_samples": BOOT,
    "provenance": provenance([RAW / f"statcast_mlb_{y}.parquet" for y in (TRAIN, P, Y)],
                             seeds={"lightgbm": SEED, "split_and_bootstrap": BOOT_SEED}),
}
(OUT / "robustness.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
yy[["pitcher", "name_prev", "team_cur", "xrv100_prev", "rv100_prev", "rv100_cur", "pitches_prev", "pitches_cur"]].to_csv(
    OUT / "yoy_pairs.csv", index=False)
print(json.dumps({k: v for k, v in res.items() if k != "provenance"}, indent=2))
