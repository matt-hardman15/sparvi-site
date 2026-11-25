"""Standing plausibility tests on what each tool publishes.

The ingest checks (`ingest/<sport>/checks.py`) run before a build and ask whether the raw data arrived intact. These
run after a build and ask whether the numbers the tools show can be true: a "+" metric averages 100 on the scale its
label states, an opponent adjustment moves teams by a non-trivial but bounded amount, ranks follow the values they
rank, no team is credited with plays from a game it didn't play.

    python -m plausibility <sport>          # baseball | nfl | cfb | all; exit status 1 if any test fails
    python -m plausibility all --check      # the same, writing nothing (deploy.yml, pull-request CI)

Each sport's module (`plausibility/<sport>.py`) defines `run(suite)`. Results go to
`site/data/status/<sport>_plausibility.json`, shown on /status/ next to the data checks. A failing test blocks
publishing the same way a failing data check does: the refresh workflows run this after the build and commit the
rebuilt data only if it passes, and `tests/test_plausibility.py` runs every suite on the committed site data, so a
pull request that would publish implausible numbers can't merge.

Two kinds of test:
  - published: reads only committed files (site/data, model outputs). Runs everywhere, including pull-request CI.
  - build:     needs build intermediates that are never committed (the dbt database, raw files). Declared with
               `suite.needs(path)`; where the file is absent the test is recorded as not run, and its last recorded
               result (from a refresh, where the file exists) is carried forward in the status file.

The status file holds only data-derived values (no clock times), so a rerun on unchanged data doesn't change it.
"""
from __future__ import annotations

import importlib
import json
import math
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from sparvi.config import ROOT
from sparvi.io import write_json

STATUS = Path(os.environ.get("SPARVI_STATUS_DIR", ROOT / "site" / "data" / "status"))
SPORTS = ("baseball", "nfl", "cfb")


class Suite:
    """Collects one sport's results. `tool` is the page or tool a test protects (shown as a group on /status/)."""

    def __init__(self, sport: str, root: Path = ROOT, build: bool = True):
        self.sport = sport
        self.root = Path(root)
        self.build = build       # False: build-only tests are not run even where their files exist (see `needs`)
        self.results: list[dict] = []

    # -- recording -------------------------------------------------------------------------------------------
    def check(self, tool: str, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append({"tool": tool, "name": name, "ok": bool(ok), "detail": str(detail)})
        return bool(ok)

    def not_run(self, tool: str, name: str, why: str) -> None:
        """A build-only test whose inputs aren't here (see `needs`). Neither a pass nor a failure."""
        self.results.append({"tool": tool, "name": name, "ok": None, "detail": why})

    def needs(self, tool: str, name: str, *paths: Path) -> bool:
        """True if every path exists; otherwise records the test as not run and returns False. With `build=False` the
        test is never run: a laptop's dbt database and raw files are its own last build, not the one the committed
        data came from, so a comparison with the committed status file must leave build-only tests out (PLAUS-NEXT)."""
        if not self.build:
            self.not_run(tool, name, "not run here: build-only test, left out of this run")
            return False
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            self.not_run(tool, name, f"not run here: needs {', '.join(_rel(p, self.root) for p in missing)}")
        return not missing

    # -- reading ---------------------------------------------------------------------------------------------
    def path(self, rel: str) -> Path:
        return self.root / rel

    def load(self, rel: str):
        return json.loads(self.path(rel).read_text(encoding="utf-8"))

    # -- reusable tests --------------------------------------------------------------------------------------
    def mean_near(self, tool: str, name: str, values: Iterable[float], target: float, tol: float,
                  weights: Iterable[float] | None = None, unit: str = "") -> bool:
        """A (weighted) mean within `tol` of `target`: e.g. a "+" metric's league mean on its stated scale."""
        m = wmean(values, weights)
        return self.check(tool, name, m is not None and abs(m - target) <= tol,
                          "no values" if m is None else f"mean {m:.2f}{unit} (expected {target:g} ± {tol:g})")

    def ranks_match(self, tool: str, name: str, rows: Sequence[dict], rank: str, value: str,
                    descending: bool = True, by: Sequence[str] = (), exact: bool = True) -> bool:
        """Within each group (`by`), ranks follow the values: a better value never has a worse rank, equal values
        share a rank, ranks start at 1, and a rank is 1 + the number of rows that are strictly better
        (competition ranking: 1, 2, 2, 4). `exact=False` is for values published rounded, or a group that is only
        part of what was ranked: it requires only that a strictly better value never has a worse (or equal) rank."""
        groups: dict[tuple, list[dict]] = {}
        for r in rows:
            if r.get(rank) is None or r.get(value) is None:
                continue
            groups.setdefault(tuple(r[k] for k in by), []).append(r)
        bad, n = [], 0
        for key, g in groups.items():
            vals = [r[value] for r in g]
            for r in g:
                n += 1
                better = sum(v > r[value] for v in vals) if descending else sum(v < r[value] for v in vals)
                if not exact:
                    beaten_by = [o for o in g if (o[value] < r[value] if descending else o[value] > r[value])
                                 and o[rank] <= r[rank]]
                    if beaten_by or r[rank] < 1:
                        bad.append(f"{'/'.join(map(str, key)) + ': ' if key else ''}{value} {r[value]} has rank {r[rank]}"
                                   f" but {beaten_by[0][value] if beaten_by else 'nothing'} has rank"
                                   f" {beaten_by[0][rank] if beaten_by else '< 1'}")
                elif r[rank] != better + 1:
                    bad.append(f"{'/'.join(map(str, key)) + ': ' if key else ''}{value} {r[value]} has rank {r[rank]},"
                               f" expected {better + 1}")
        return self.check(tool, name, not bad and n > 0,
                          f"{len(bad)} of {n} rows, e.g. {bad[0]}" if bad else f"{n:,} rows in {len(groups):,} groups")

    def within(self, tool: str, name: str, value: float | None, lo: float, hi: float, fmt: str = ".3f",
               what: str = "") -> bool:
        ok = value is not None and not math.isnan(value) and lo <= value <= hi
        shown = "missing" if value is None else format(value, fmt)
        return self.check(tool, name, ok, f"{what + ' ' if what else ''}{shown} (allowed {lo:g} to {hi:g})")

    # -- output ----------------------------------------------------------------------------------------------
    def finish(self, write: bool = True, status_dir: Path | None = None) -> bool:
        """Print failures; write the status file; True if no test failed. Tests not run here keep their last
        recorded result, so a pull-request run (no build intermediates) can't erase what a refresh found."""
        out = (status_dir or STATUS) / f"{self.sport}_plausibility.json"
        prev = {}
        if out.exists():
            prev = {(r["tool"], r["name"]): r for r in json.loads(out.read_text(encoding="utf-8")).get("checks", [])}
        results = []
        for r in self.results:
            if r["ok"] is None:
                last = prev.get((r["tool"], r["name"]))
                r = dict(last) if last and last["ok"] is not None else r
            results.append(r)
        failed = [r for r in results if r["ok"] is False]
        ok = not failed
        if write:
            write_json(out, {
                "sport": self.sport, "ok": ok, "n_checks": len(results), "n_failed": len(failed),
                "n_not_run": sum(r["ok"] is None for r in results), "checks": results,
            }, indent=1)
        for r in failed:
            print(f"FAILED  [{r['tool']}] {r['name']}: {r['detail']}")
        print(f"{self.sport} plausibility: {len(results)} tests, {len(failed)} failed,"
              f" {sum(r['ok'] is None for r in results)} not run")
        return ok


def wmean(values: Iterable[float], weights: Iterable[float] | None = None) -> float | None:
    vals = list(values)
    wts = [1.0] * len(vals) if weights is None else list(weights)
    pairs = [(v, w) for v, w in zip(vals, wts, strict=True) if v is not None and w is not None and not math.isnan(v)]
    tot = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / tot if tot else None


def sd(values: Iterable[float]) -> float:
    v = [x for x in values if x is not None]
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def columnar(obj: dict) -> list[dict]:
    """Rows from a columnar site file ({"columns": [...], "rows": [[...]]} or {"col": [values]})."""
    if "columns" in obj and "rows" in obj:
        return [dict(zip(obj["columns"], r, strict=True)) for r in obj["rows"]]
    keys = [k for k, v in obj.items() if isinstance(v, list)]
    return [dict(zip(keys, vals, strict=True)) for vals in zip(*(obj[k] for k in keys), strict=True)]


def _rel(p: Path, root: Path) -> str:
    p = Path(p)
    return p.relative_to(root).as_posix() if p.is_relative_to(root) else p.name


def run_sport(sport: str, write: bool = True, root: Path = ROOT, status_dir: Path | None = None, build: bool = True) -> bool:
    suite = Suite(sport, root, build=build)
    importlib.import_module(f"plausibility.{sport}").run(suite)
    return suite.finish(write=write, status_dir=status_dir)


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    write = "--check" not in args
    args = [a for a in args if a != "--check"] or ["all"]
    sports = SPORTS if args[0] == "all" else (args[0],)
    return 0 if all([run_sport(s, write=write) for s in sports]) else 1
