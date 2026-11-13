# forecasts/ (stage 4: the forecast ledger)

Append-only JSONL, one file per sport and season (`forecasts/<sport>/<season>.jsonl`), written only
by the scheduled GitHub Actions job; resolutions go in `<season>.resolved.jsonl`. Lines are never
edited or deleted; mistakes are voided with a reason. Format and integrity rules: PLAN-scout §4.

Empty until L1 (2027 MLB preseason forecasts, locked before spring training).
