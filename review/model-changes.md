# Model changes

Newest first. Each entry: date, model and version, what changed, why, and before/after metrics.

## 2026-11-13 · Registry started (A2): six models at v1.0.0
Registered as published: Stuff+, Location+, Pitching+ (out-of-sample r with next-season run value
0.297, 0.012, 0.318), Triple-A → MLB translation (hitter CV R² 0.046), NFL opponent adjustment
(next-week margin MAE 10.39), CFB power rating (MAE 12.55 vs. Elo 12.68). Records:
`models/<sport>/<model>/versions/1.0.0.json`. No model changed.

## 2026-11-13 · Stage layout (A0)
No model changed. Ingest moved to `ingest/`, data checks and contracts added; published outputs
verified byte-identical before and after on the same machine and data.
