# models/ (stage 3: model registry)

One folder per model, `models/<sport>/<model>/`, with a `card.yml` (target, features, training window,
validation, version, owner, status) and one subfolder per version holding the serialized model, its
`metrics.json`, and the training-data hash. Champion/challenger and versioning rules: PLAN-scout A2.

Empty until A2. Today's models still live with their analyses (`baseball/pitch_quality/`,
`baseball/aaa_translation/`, `nfl/team_efficiency/`); they move here with a card each under A2.
