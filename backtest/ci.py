"""Backtests in CI (PLAN-backtest §5): run every model that has a pre-registered `backtest:` block,
compare with the last recorded report, and decide.

    python -m backtest.ci --fast --summary summary.md      # pull requests: blocks on leaks or regressions
    python -m backtest.ci --write --summary summary.md     # push to main / weekly: full runs, reports + run log
    python -m backtest.ci --changed files.txt --select     # which models a change touches (CI picks what to run)

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

# Files every backtest depends on: a change here runs every model.
HARNESS = ("backtest/", ".github/workflows/backtest.yml", "pyproject.toml", "uv.lock", "sparvi.toml", "sparvi/registry.py",
           "sparvi/config.py", "ingest/snapshot.py", "ingest/common/asof.py")


def selected(changed: list[str], models=None) -> list | None:
    """The models a change touches: a file in a model's card `code` list or in its folder (not its committed reports).
    None means every model (a harness file changed). Paths are repo-relative with forward slashes."""
    changed = [c.strip().replace(chr(92), "/") for c in changed if c.strip()]
    if any(c.startswith(HARNESS) and not c.startswith("backtest/runs/") for c in changed):
        return None
    out = []
    for m in models or [m for m in registry.models() if "backtest" in m.card]:
        own = {p.relative_to(m.root).as_posix() for p in m.files("code")}
        folder = m.dir.relative_to(m.root).as_posix() + "/"
        reports = folder + "backtest/"
        if any(c in own or (c.startswith(folder) and not c.startswith(reports)) for c in changed):
            out.append(m)
    return out


def last_report(m) -> dict | None:
    from backtest.runner import tested_version
    p = m.dir / "backtest" / f"{tested_version(m)}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def comparable_point(rep: dict, prev: dict | None) -> float | None:
    """The recorded score on the same folds as this run: the pooled score when the folds match, else (a fast run
    against a full report) that fold's score from the report's by_fold. None when there is nothing comparable."""
    if prev is None:
        return None
    if list(prev.get("folds", [])) == list(rep["folds"]):
        return prev["evaluation"]["point"]
    by = prev["evaluation"].get("by_fold", {})
    if len(rep["folds"]) == 1 and str(rep["folds"][0]) in {str(k) for k in by}:
        return {str(k): v for k, v in by.items()}[str(rep["folds"][0])]
    return None


def judge(m, rep: dict, prev: dict | None) -> tuple[bool, list[str]]:
    spec = m.card["backtest"]
    notes, ok = [], True
    leaks = [k for k, v in rep["leakage"].items() if not v["passed"]]
    if leaks:
        ok = False
        notes.append(f"leakage test failed: {', '.join(leaks)}")
    before = comparable_point(rep, prev)
    if prev is not None and before is None:
        notes.append(f"no recorded score for fold(s) {rep['folds']} to compare this {rep['kind']} run with; regression check skipped")
    if before is not None:
        tol = spec.get("regression_tolerance", 0.01)
        sign = 1 if metrics.higher_is_better(rep["evaluation"]["metric"]) else -1
        delta = sign * (rep["evaluation"]["point"] - before)
        if delta < -tol:
            ok = False
            notes.append(f"{rep['evaluation']['metric']} got worse: {before:.4f} → {rep['evaluation']['point']:.4f} (tolerance {tol})")
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
    ap.add_argument("--changed", help="file listing changed paths: only the models they touch run (default: all)")
    ap.add_argument("--select", action="store_true", help="print models=... and sports=... for CI, run nothing")
    a = ap.parse_args()
    ported = [m for m in registry.models() if "backtest" in m.card]
    pick = ported
    if a.changed:
        with open(a.changed, encoding="utf-8") as f:
            sel = selected(f.read().splitlines(), ported)
        pick = ported if sel is None else sel
    if a.select:
        print(f"models={','.join(m.name for m in pick)}")
        print(f"sports={','.join(sorted({m.card['sport'] for m in pick}))}")
        return 0
    rows, all_ok, notes_all = [], True, []
    for m in pick:
        prev = last_report(m)
        rep = runner.run(m.name, fast=a.fast, write=a.write)
        ok, notes = judge(m, rep, prev)
        all_ok &= ok
        notes_all += [f"**{m.name}**: {n}" for n in notes]
        ev = rep["evaluation"]
        if ev["baselines"]:
            b = next(iter(ev["baselines"].items()))
            edge = f"{b[0]}: {b[1]['paired']['est']:+.4f} ({b[1]['paired']['ci90'][0]:+.3f} to {b[1]['paired']['ci90'][1]:+.3f})"
        else:   # calibration gate (F4's lookup tables): the ECE against the noise bar
            nb = ev.get("ece_noise", {})
            edge = f"noise bar {nb['bar']:.4f} (q{nb['q']:.2f})" if nb else "no baseline"
        rows.append({"model": m.name, "version": m.version, "kind": rep["kind"], "metric": rep["evaluation"]["metric"],
                     "now": f"{rep['evaluation']['point']:.4f}", "prev": f"{comparable_point(rep, prev):.4f}" if comparable_point(rep, prev) is not None else "none",
                     "edge": edge,
                     "gate": "pass" if rep["gate"]["passed"] else "fail",
                     "leakage": "pass" if all(v["passed"] for v in rep["leakage"].values()) else "FAIL",
                     "status": status(m.name)["state"], "ci": "ok" if ok else "BLOCK"})
    not_ported = [m.name for m in registry.models() if "backtest" not in m.card]
    skipped = [m.name for m in ported if m not in pick]
    md = "## Backtests\n\n" + table(rows) + "\n\n" + ("\n".join(f"- {n}" for n in notes_all) + "\n\n" if notes_all else "") \
         + (f"Not run (this change doesn't touch them): {', '.join(skipped)}.\n\n" if skipped else "") \
         + (f"Not yet in the harness (A5-port): {', '.join(not_ported)}.\n" if not_ported else "")
    print(md)
    if a.summary:
        with open(a.summary, "w", encoding="utf-8") as f:
            f.write(md)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
