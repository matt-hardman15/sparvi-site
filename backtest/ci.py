"""Backtests in CI (PLAN-backtest §5): run every model that has a pre-registered `backtest:` block,
compare with the last recorded report, and decide.

    python -m backtest.ci --fast --summary summary.md      # pull requests: blocks on leaks or regressions
    python -m backtest.ci --write --summary summary.md     # push to main / weekly: full runs, reports + run log

A run fails CI if any leakage test fails, if the primary metric got worse than the last recorded report by
more than the model's tolerance (`regression_tolerance`, default 0.01: the cross-machine reproduction
tolerance), or if the gate fails for a model that isn't flagged experimental. A model already flagged
experimental can fail its gate without failing CI (the site labels it); it can't get worse or leak.
"""
from __future__ import annotations

import argparse
import json
import sys

from backtest import metrics, runner
from backtest.gate import status
from sparvi import registry


def last_report(m) -> dict | None:
    p = m.dir / "backtest" / f"{m.version}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def judge(m, rep: dict, prev: dict | None) -> tuple[bool, list[str]]:
    spec = m.card["backtest"]
    notes, ok = [], True
    leaks = [k for k, v in rep["leakage"].items() if not v["passed"]]
    if leaks:
        ok = False
        notes.append(f"leakage test failed: {', '.join(leaks)}")
    if prev is not None:
        tol = spec.get("regression_tolerance", 0.01)
        sign = 1 if metrics.higher_is_better(rep["evaluation"]["metric"]) else -1
        delta = sign * (rep["evaluation"]["point"] - prev["evaluation"]["point"])
        if delta < -tol:
            ok = False
            notes.append(f"{rep['evaluation']['metric']} got worse: {prev['evaluation']['point']:.4f} → {rep['evaluation']['point']:.4f} (tolerance {tol})")
    if not rep["gate"]["passed"] and not m.card.get("experimental"):
        ok = False
        notes.append("failed its pre-registered gate and isn't flagged experimental")
    return ok, notes


def table(rows: list[dict]) -> str:
    out = ["| Model | Version | Run | Metric | Now | Last recorded | vs. baseline (90% CI) | Gate | Leakage | Status | CI |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append("| " + " | ".join(str(r[k]) for k in ("model", "version", "kind", "metric", "now", "prev", "edge", "gate", "leakage", "status", "ci")) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--write", action="store_true", help="write reports and append to the run log")
    ap.add_argument("--summary", default=None)
    a = ap.parse_args()
    rows, all_ok, notes_all = [], True, []
    for m in registry.models():
        if "backtest" not in m.card:
            continue
        prev = last_report(m)
        rep = runner.run(m.name, fast=a.fast, write=a.write)
        ok, notes = judge(m, rep, prev)
        all_ok &= ok
        notes_all += [f"**{m.name}**: {n}" for n in notes]
        b = next(iter(rep["evaluation"]["baselines"].items()))
        rows.append({"model": m.name, "version": m.version, "kind": rep["kind"], "metric": rep["evaluation"]["metric"],
                     "now": f"{rep['evaluation']['point']:.4f}", "prev": f"{prev['evaluation']['point']:.4f}" if prev else "none",
                     "edge": f"{b[0]}: {b[1]['paired']['est']:+.4f} ({b[1]['paired']['ci90'][0]:+.3f} to {b[1]['paired']['ci90'][1]:+.3f})",
                     "gate": "pass" if rep["gate"]["passed"] else "fail",
                     "leakage": "pass" if all(v["passed"] for v in rep["leakage"].values()) else "FAIL",
                     "status": status(m.name)["state"], "ci": "ok" if ok else "BLOCK"})
    not_ported = [m.name for m in registry.models() if "backtest" not in m.card]
    md = "## Backtests\n\n" + table(rows) + "\n\n" + ("\n".join(f"- {n}" for n in notes_all) + "\n\n" if notes_all else "") \
         + (f"Not yet in the harness (A5-port): {', '.join(not_ported)}.\n" if not_ported else "")
    print(md)
    if a.summary:
        with open(a.summary, "w", encoding="utf-8") as f:
            f.write(md)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
