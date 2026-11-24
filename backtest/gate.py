"""The publish gate (PLAN-backtest §6): no model output reaches the site, the ledger, or a client report
without a passing, current backtest for that exact model version, unless the model's card flags it
`experimental` (with a reason), in which case the pages say so next to the version.

    from backtest.gate import require_passing
    require_passing("stuff_plus")        # in every exporter, before writing model output

Current (A5-next; Matt, Nov 24 2026): the model's backtest ran within MAX_AGE_DAYS, in every month. The backtest
workflow reruns every model each Monday, so a report is normally up to a week old; one failed Monday run doesn't trip
the limit and two in a row do. An older pass doesn't count: the export stops (the refresh keeps the last good data and
ends red) until the backtest runs again. While a card's block tests a `candidate`, the Monday run rewrites the
candidate's report and not the published version's, so the newer of the two dates is the one that counts: the published
version still needs its own passing report. A model flagged `experimental` publishes with its label as before.
The age isn't enforced on a pull request's checks (nothing is published from one, and the pull request that repairs
a broken backtest has to be able to pass).
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime

from sparvi import registry

MAX_AGE_DAYS = 15


class BacktestRequired(RuntimeError):
    pass


def _ran_at(rep: dict | None) -> datetime | None:
    try:
        return datetime.fromisoformat(str(rep["ran_at"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return None


def age_days(m, now: datetime | None = None) -> float | None:
    """Days since the model's backtest last ran: the newer of the published version's report and, when the block tests
    a candidate, the candidate's. None when no report carries a readable date."""
    versions = {m.version, str(m.card.get("backtest", {}).get("candidate", m.version))}
    paths = [m.dir / "backtest" / f"{v}.json" for v in sorted(versions)]
    ran = [t for t in (_ran_at(json.loads(p.read_text(encoding="utf-8"))) for p in paths if p.exists()) if t is not None]
    return ((now or datetime.now(UTC)) - max(ran)).total_seconds() / 86400 if ran else None


def status(name: str, root=None, now: datetime | None = None) -> dict:
    m = registry.get(name, root) if root else registry.get(name)
    path = m.dir / "backtest" / f"{m.version}.json"
    rep = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    passing = rep is not None and rep.get("version") == m.version and bool(rep.get("passed"))
    stale = None
    if passing and os.environ.get("GITHUB_EVENT_NAME") != "pull_request":
        age = age_days(m, now)
        if age is None or age > MAX_AGE_DAYS:
            stale = ("its backtest report carries no run date" if age is None else
                     f"its backtest last ran {int(age)} days ago (limit {MAX_AGE_DAYS}): the weekly backtest run has not been landing")
    # The validated state carries no date or age: exporters write it into site data, which must not change on a rerun.
    if passing and stale is None:
        return {"model": name, "version": m.version, "state": "validated", "report": path.name}
    exp = m.card.get("experimental")
    if exp:
        return {"model": name, "version": m.version, "state": "experimental", "reason": exp,
                "report": path.name if rep else None}
    if stale:
        return {"model": name, "version": m.version, "state": "blocked", "stale": True, "reason": stale}
    return {"model": name, "version": m.version, "state": "blocked",
            "reason": "no passing backtest for this version" + ("" if rep is None else " (latest run failed its gate)")}


def require_passing(name: str, root=None, now: datetime | None = None) -> dict:
    s = status(name, root, now)
    if s["state"] == "blocked" and s.get("stale"):
        raise BacktestRequired(f"{name} v{s['version']}: {s['reason']}. Run the backtest workflow on main "
                               "(`gh workflow run backtest.yml`) and pull its reports.")
    if s["state"] == "blocked":
        raise BacktestRequired(f"{name} v{s['version']}: {s['reason']}. Run `python -m backtest.runner {name}`, "
                               "or flag the card `experimental:` with a reason (the site then labels it).")
    return s
