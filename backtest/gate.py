"""The publish gate (PLAN-backtest §6): no model output reaches the site, the ledger, or a client report
without a passing, current backtest for that exact model version, unless the model's card flags it
`experimental` (with a reason), in which case the pages say so next to the version.

    from backtest.gate import require_passing
    require_passing("stuff_plus")        # in every exporter, before writing model output
"""
from __future__ import annotations

import json

from sparvi import registry


class BacktestRequired(RuntimeError):
    pass


def status(name: str, root=None) -> dict:
    m = registry.get(name, root) if root else registry.get(name)
    path = m.dir / "backtest" / f"{m.version}.json"
    rep = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if rep is not None and rep.get("version") == m.version and rep.get("passed"):
        return {"model": name, "version": m.version, "state": "validated", "report": path.name}
    exp = m.card.get("experimental")
    if exp:
        return {"model": name, "version": m.version, "state": "experimental", "reason": exp,
                "report": path.name if rep else None}
    return {"model": name, "version": m.version, "state": "blocked",
            "reason": "no passing backtest for this version" + ("" if rep is None else " (latest run failed its gate)")}


def require_passing(name: str, root=None) -> dict:
    s = status(name, root)
    if s["state"] == "blocked":
        raise BacktestRequired(f"{name} v{s['version']}: {s['reason']}. Run `python -m backtest.runner {name}`, "
                               "or flag the card `experimental:` with a reason (the site then labels it).")
    return s
