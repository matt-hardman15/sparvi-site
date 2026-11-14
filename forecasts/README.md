# forecasts/ (stage 4: the forecast ledger)

Append-only JSONL, one file per sport and season, written only by GitHub Actions on main
(`sparvi/ledger.py`; format and integrity rules: PLAN-scout §4).

- `<sport>/<season>.jsonl`: one line per forecast. The writer stamps `created_at` (UTC), `code_sha`,
  and `model_state` (`validated` or `experimental`, from the model's publish gate). A forecast is a
  probability (`prob`) or a point with an 80% interval (`point`, `lo80`, `hi80`), names its model's
  current registered version, and resolves after `resolves_after`. Type-specific fields go in `detail`.
- `<sport>/<season>.resolved.jsonl`: one line per resolution, `scored` with an `actual`, or `void`
  with a `void_reason`. Each forecast resolves once.

Lines are never edited, reordered, or deleted; a mistake is voided with a reason. CI enforces it:
every pull request runs `python -m sparvi.ledger check --pr` (no PR may change or add a ledger line),
and the `ledger` workflow checks every Actions push to main against the commit before it.

A job that writes forecasts calls `ledger.append_forecasts(...)` / `ledger.append_resolutions(...)`,
runs `python -m sparvi.ledger check --base HEAD`, commits the ledger files on their own, and pushes
with the deploy key. Locally the writer refuses the real ledger; `python -m sparvi.ledger smoke`
exercises the whole path in a scratch copy.

First entries: the January 2027 portal board (SC-3a). Then 2027 MLB preseason forecasts (L1b).
