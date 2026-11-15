# Model changes

Newest first. Each entry: date, model and version, what changed, why, and before/after metrics.

## 2026-11-15 · CFB player profile 1.0.0 → 1.0.1 (patch)
Share features (target share, carry share, dropback share, runs per dropback) now divide by the team's pass or run plays counted by play id instead of matched dropbacks, capped at 1. Found in SYSTEM-FIT F2: teams whose passers weren't matched to rosters (mostly FCS) produced shares up to 74. The opponent-adjusted features the pre-registered gate tests are unaffected.

## 2026-11-14 · Stuff+, Location+, Pitching+ 2.0.0: bagged training (PQ-STABLE)
**Why.** Checking free-agent grades (MLB-FA) against the published ones found per-pitcher Stuff+ up to 15 points apart between a local build and CI. Experiments (same data and code): Windows and Linux gave byte-identical models, so neither the OS nor floating point was the cause; changing only the seed (7 → 8) moved Stuff+ by 3.0 points on average and up to 64 (Pitching+ 2.1, up to 42). Each regressor was one LightGBM model early-stopped on a single 15% pitcher slice; the Stuff+ models stopped after 12–100 trees, and which pitchers fell in the slice (or a tiny data revision, which is how CI's cached raw data differed from the local copy) moved the stopping point and with it the grades. This corrects the 2026-11-13 entry below, which put cross-machine differences down to CPU models: the instability, not the hardware, made them large.

**What changed.** Each pitch-group regressor is now a bag of 10: pitchers split into 10 folds by a seeded shuffle, model f trains on the other nine folds and early-stops on fold f (member seed 700 + f), and predictions are the bag average. Stuff+ learning rate 0.03 → 0.005 (its models now run 130–650 trees). Model files are no longer committed (90 files; the version records keep their hashes). New `baseball/pitch_quality/stability.py` retrains with another seed and reports the per-pitcher differences.

**Results.** Seed sensitivity (562 MLB pitchers with 300+ pitches, seeds 7 vs 8): Stuff+ mean 0.35, max 3.5 (was 3.0 / 64); Pitching+ 0.38 / 2.7 (was 2.1 / 42); Location+ 0.12 / 0.6. For scale, a typical pitcher's 90% sampling interval is about ±2 points for Stuff+ and ±6 for Pitching+. Frozen eval set (r with next-season run value): Stuff+ 0.297 → 0.305, Pitching+ 0.318 → 0.318, Location+ 0.012 → 0.008. Split-half reliability unchanged (Stuff+ 0.994). Training takes about 64 minutes on 4 cores (was about 4); robustness.py about 25.

## 2026-11-14 · Pitching+ 1.0.1 → 2.0.0 (promoted by exception, Matt)
Frozen eval set: r_next_rv 0.3183 → 0.3179, lower by less than the ±0.01 reproduction tolerance, so `registry promote` declined. Matt approved promotion anyway (2026-11-14): 1.0.1's value was itself seed noise, and the shared training code (bagging) is what makes the grades stable. Live: not yet available (no ledger forecasts).

## 2026-11-14 · Location+ 1.0.1 → 2.0.0 (promoted by exception, Matt)
Frozen eval set: r_next_rv 0.01156 → 0.007667, lower by less than the ±0.01 reproduction tolerance, so `registry promote` declined. Matt approved promotion anyway (2026-11-14): 1.0.1's value was itself seed noise, and the shared training code (bagging) is what makes the grades stable. Live: not yet available (no ledger forecasts).

## 2026-11-14 · Stuff+ 1.0.1 → 2.0.0 (promoted)
Frozen eval set: Out-of-sample year over year: models refit on the earliest season only, grade the next season, and correlate with the season after's run value (MLB pitchers with enough pitches both years). Same split as the published Stuff+ validation.. r_next_rv: 0.297 → 0.3046. Live: not yet available (no ledger forecasts).

## 2026-11-13 · Stuff+ through the backtest harness (A5): reproduction
The harness reproduces `robustness.py` exactly on the same machine (n 324; r 0.3014 vs last-season RV 0.1926). The published 0.2970 was trained on a different CPU model: LightGBM trees differ slightly across hardware (also the cause of the refresh drift fixed in OPS-0). Not a leak or a harness bug. Reproduction standard (Matt): exact on one machine, ±0.01 in r across machines; official backtests run in CI. Pass gate pre-registered as strict (Matt): the edge over last season's run value must have a 90% interval excluding zero.

## 2026-11-13 · Stuff+, Location+, Pitching+ 1.0.0 → 1.0.1 (patch)
`build_features()` now takes batter heights as an argument instead of always reading the raw file, so the backtest harness (A5) can supply them through the point-in-time loader. With no argument it reads the same file as before: identical features and models; validation metrics unchanged (r_next_rv 0.297, 0.012, 0.318).

## 2026-11-13 · Registry started (A2): six models at v1.0.0
Registered as published: Stuff+, Location+, Pitching+ (out-of-sample r with next-season run value
0.297, 0.012, 0.318), Triple-A → MLB translation (hitter CV R² 0.046), NFL opponent adjustment
(next-week margin MAE 10.39), CFB power rating (MAE 12.55 vs. Elo 12.68). Records:
`models/<sport>/<model>/versions/1.0.0.json`. No model changed.

## 2026-11-13 · Stage layout (A0)
No model changed. Ingest moved to `ingest/`, data checks and contracts added; published outputs
verified byte-identical before and after on the same machine and data.
