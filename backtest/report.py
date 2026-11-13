"""Backtest reports and the run log (PLAN-backtest §3, principles 6-7).

- models/<sport>/<model>/backtest/<version>.json: the latest backtest of that model version (what the
  publish gate and the method boxes read).
- backtest/runs/runs.jsonl: every run, passing or not, appended and never edited.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from sparvi.config import ROOT

RUNS = ROOT / "backtest" / "runs" / "runs.jsonl"


def code_sha() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def lock_hash() -> str:
    return hashlib.sha256((ROOT / "uv.lock").read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def write(model_dir: Path, version: str, report: dict, runs: Path = RUNS) -> Path:
    out = model_dir / "backtest" / f"{version}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    runs.parent.mkdir(parents=True, exist_ok=True)
    line = {k: report[k] for k in ("run_id", "model", "version", "ran_at", "code_sha", "kind", "passed", "primary", "revision_safe")}
    with runs.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(line, default=str, separators=(",", ":")) + "\n")
    return out
