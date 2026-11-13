"""Data checks at ingest (PLAN-scout A0, stage 1). They run right after a pull and before any build.

Each sport's `ingest/<sport>/checks.py` checks its raw files against `contracts/<sport>_ingest.yml`
(columns and types the downstream stages read, duplicate keys, null rates) plus its own rules
(coverage, freshness), then writes `site/data/status/<sport>.json` for the /status/ page.

A failed check means the refresh stops before rebuilding anything: the site keeps serving the last
good data, and the status file (committed by the workflow) says what failed. Row counts are compared
with the last *accepted* counts in that file, so a failed pull can't lower the baseline.

The status file holds only data-derived values (no clock times), so rerunning the checks on
unchanged data doesn't change it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import yaml

from sparvi.config import ROOT
from sparvi.io import write_json

CONTRACTS = ROOT / "contracts"
STATUS = Path(os.environ.get("SPARVI_STATUS_DIR", ROOT / "site" / "data" / "status"))
ROW_DROP_TOLERANCE = 0.01    # a table may shrink by at most 1% vs. the last accepted pull (revisions)


def contract(sport: str) -> dict:
    return yaml.safe_load((CONTRACTS / f"{sport}_ingest.yml").read_text(encoding="utf-8"))["tables"]


def columns(path: Path) -> dict[str, str]:
    return {n: t for n, t, *_ in duckdb.sql(f"describe select * from read_parquet('{path.as_posix()}')").fetchall()}


class Checks:
    def __init__(self, sport: str):
        self.sport = sport
        self.contract = contract(sport)
        self.results: list[dict] = []
        self.rows: dict[str, int] = {}
        prev = STATUS / f"{sport}.json"
        self.prev_rows = json.loads(prev.read_text(encoding="utf-8")).get("rows", {}) if prev.exists() else {}

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append({"name": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    def sql(self, q: str):
        return duckdb.sql(q).fetchone()

    def table(self, name: str, path: Path, kind: str) -> bool:
        """Present, schema per the contract, rows vs. the last accepted pull, unique key, null rates.
        Returns False if the file is missing or unreadable (later checks on it are skipped)."""
        spec = self.contract[kind]
        if not path.exists():
            return self.add(f"{name}: present", False, f"{path.name} is missing")
        rel = f"read_parquet('{path.as_posix()}')"
        have = columns(path)
        want = spec["columns"]
        missing = [c for c in want if c not in have]
        changed = [f"{c} {want[c]} -> {have[c]}" for c in want if c in have and have[c] != want[c]]
        problems = ([f"missing {', '.join(missing)}"] if missing else []) + changed
        self.add(f"{name}: columns and types", not problems,
                 "; ".join(problems) if problems else f"{len(want)} contract columns")

        n = self.sql(f"select count(*) from {rel}")[0]
        self.rows[name] = n
        prev = self.prev_rows.get(name)
        floor = max(spec.get("min_rows", 1), int(prev * (1 - ROW_DROP_TOLERANCE)) if prev else 0)
        detail = f"{n:,} rows" + (f" (last accepted {prev:,})" if prev else "")
        self.add(f"{name}: row count", n >= floor, detail)

        if keys := spec.get("key"):
            if all(k in have for k in keys):
                d = self.sql(f"select count(*) - count(distinct ({', '.join(keys)})) from {rel}")[0]
                self.add(f"{name}: one row per {' + '.join(keys)}", d == 0, f"{d:,} duplicate rows" if d else "")
        for col, max_rate in (spec.get("max_null") or {}).items():
            if col not in have:
                continue
            r = self.sql(f"select coalesce(avg(({col} is null)::double), 0) from {rel}")[0]
            self.add(f"{name}: {col} missing", r <= max_rate, f"{r:.2%} (limit {max_rate:.0%})")
        return not missing

    def finish(self, data_through: dict) -> bool:
        ok = all(r["ok"] for r in self.results)
        rows = self.rows if ok else self.prev_rows    # keep the last accepted counts as the baseline
        write_json(STATUS / f"{self.sport}.json", {
            "sport": self.sport, "ok": ok, "data_through": data_through,
            "n_checks": len(self.results), "n_failed": sum(not r["ok"] for r in self.results),
            "rows": dict(sorted(rows.items())), "checks": self.results,
        }, indent=1)
        for r in self.results:
            if not r["ok"]:
                print(f"FAILED  {r['name']}: {r['detail']}")
        print(f"{self.sport}: {len(self.results)} checks, {sum(not r['ok'] for r in self.results)} failed"
              f" -> {(STATUS / f'{self.sport}.json').relative_to(ROOT) if STATUS.is_relative_to(ROOT) else STATUS}")
        return ok
