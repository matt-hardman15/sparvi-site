"""Fully out-of-sample version of the year-over-year test, for Stuff+, Location+, and Pitching+.

train.py fits on every MLB season before the current one, so its last-season grades are in-sample.
Here the models see only the season before that; last season's grades are then used to predict this
season's run value, so neither side touched training. Seasons come from sparvi.toml (with
season = 2026: train 2024, grade 2025, predict 2026).

Writes:
  robustness.json         the Stuff+ test (the Stuff+ article reads this; keys unchanged)
  yoy_pairs.csv           pitcher pairs behind it (the Stuff+ article's explorer)
  metrics_location.json   the side-by-side validation for the stuff-or-command article: split-half
                          reliability, out-of-sample year-over-year with bootstrap intervals, the
                          Pitching+-vs-Stuff+ comparison, and the Triple-A -> MLB check
"""
import json

import numpy as np
from features import ZONE_BOT, ZONE_TOP
from train import KINDS, MIN_PITCHES, OUT, SEED, fit_kinds, load, pitcher_season, predict_all, xrv100  # noqa: F401  (MIN_PITCHES: query strings)

from sparvi.config import RAW, baseball
from sparvi.provenance import provenance

CFG = baseball()
Y, P = CFG.season, CFG.prev
TRAIN = P - 1
BOOT, BOOT_SEED = 2000, SEED

tr = load("mlb", TRAIN)
rng = np.random.default_rng(BOOT_SEED)
val_p = set(rng.choice(tr["pitcher"].unique(), size=int(0.15 * tr["pitcher"].nunique()), replace=False))
models = fit_kinds(tr, tr["pitcher"].isin(val_p))

d_prev, d_cur = load("mlb", P), load("mlb", Y)
predict_all(models, d_prev)
predict_all(models, d_cur)
a = pitcher_season(d_prev).query("pitches >= @MIN_PITCHES")
b = pitcher_season(d_cur).query("pitches >= @MIN_PITCHES")
yy = a.merge(b, on="pitcher", suffixes=("_prev", "_cur"))

# Predictors of this season's run value: each grade last season, and last season's results.
PRED = {k: f"{xrv100(k)}_prev" for k in KINDS} | {"results": "rv100_prev"}
OUTCOME = "rv100_cur"
GAPS = {"stuff_minus_results": ("stuff", "results"), "pitching_minus_stuff": ("pitching", "stuff"),
        "loc_minus_stuff": ("loc", "stuff"), "pitching_minus_results": ("pitching", "results")}

# One bootstrap for everything: same seed and draws as the single-grade version, so the Stuff+
# numbers don't change.
draws = {k: [] for k in PRED} | {g: [] for g in GAPS}
for _ in range(BOOT):
    s = yy.sample(len(yy), replace=True, random_state=rng.integers(1e9))
    r = {k: s[c].corr(s[OUTCOME]) for k, c in PRED.items()}
    for k in PRED:
        draws[k].append(r[k])
    for g, (x, z) in GAPS.items():
        draws[g].append(r[x] - r[z])


def ci(v):
    return [float(np.percentile(v, 5)), float(np.percentile(v, 95))]


full_r = {k: float(yy[c].corr(yy[OUTCOME])) for k, c in PRED.items()}

res = {
    "train_season": TRAIN, "predictor_season": P, "outcome_season": Y,
    "n": int(len(yy)),
    "r_stuff_prev_to_rv_cur": full_r["stuff"],
    "r_rv_prev_to_rv_cur": full_r["results"],
    "gap_ci90": ci(draws["stuff_minus_results"]),
    "bootstrap_samples": BOOT,
    "provenance": provenance([RAW / f"statcast_mlb_{y}.parquet" for y in (TRAIN, P, Y)],
                             seeds={"lightgbm": SEED, "split_and_bootstrap": BOOT_SEED}),
}
(OUT / "robustness.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
yy[["pitcher", "name_prev", "team_cur", "xrv100_prev", "rv100_prev", "rv100_cur", "pitches_prev", "pitches_cur"]].to_csv(
    OUT / "yoy_pairs.csv", index=False)

m = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
loc = {
    "seasons": {"train": TRAIN, "predictor": P, "outcome": Y},
    "min_pitches": MIN_PITCHES,
    "zone_share_of_height": {"top": ZONE_TOP, "bottom": ZONE_BOT},
    "split_half": {"stuff": m["split_half_r_stuff"], "loc": m["split_half_r_loc"],
                   "pitching": m["split_half_r_pitching"], "results": m["split_half_r_actual_rv"]},
    "yoy": {"n": int(len(yy)), "bootstrap_samples": BOOT,
            "r_next_rv": {k: {"r": full_r[k], "ci90": ci(draws[k])} for k in PRED},
            "gaps": {g: {"est": full_r[x] - full_r[z], "ci90": ci(draws[g])} for g, (x, z) in GAPS.items()},
            # How stable each grade is from one season to the next (out of sample)
            "r_self": {k: float(yy[f"{xrv100(k)}_prev"].corr(yy[f"{xrv100(k)}_cur"])) for k in KINDS}
                      | {"results": float(yy["rv100_prev"].corr(yy["rv100_cur"]))}},
    "aaa_to_mlb": {"n": m["aaa_to_mlb_n"], "stuff": m["aaa_to_mlb_r_stuff"], "loc": m["aaa_to_mlb_r_loc"],
                   "pitching": m["aaa_to_mlb_r_pitching"], "results": m["aaa_to_mlb_r_actual"]},
    "holdout_pitch_r2": {"stuff": m["holdout_pitch_r2"], "loc": m["holdout_pitch_r2_loc"],
                         "pitching": m["holdout_pitch_r2_pitching"]},
}
(OUT / "metrics_location.json").write_text(json.dumps(loc, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "provenance"}, indent=2))
print(json.dumps(loc, indent=2))
