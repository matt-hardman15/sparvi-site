# Model changes

Newest first. Each entry: date, model and version, what changed, why, and before/after metrics.

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
