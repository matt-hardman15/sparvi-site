"""Carry a development run's result files from a GitHub Actions job back to a checkout (DEV-CI).

The `dev-baseball` workflow runs a development script on a GitHub runner, where a model fit takes a fraction of the
time it does on a laptop. Its results are small aggregate JSON files, the same ones that get committed. This prints
them into the job log in pieces, and reads them back out of a finished run's log:

    python scripts/dev_results.py emit FILE [FILE ...]          # in the workflow: print each file into the log
    python scripts/dev_results.py fetch RUN_ID [--into DIR]     # locally (needs `gh`): write them under DIR

`fetch` writes under `dev_results/<run id>/` by default, so a result never silently replaces a file in the working
tree: compare, then copy. Only files named baseball/<project>/output/development*.json are accepted in either
direction. Nothing here touches raw or pitch-level data: the workflow's caches hold those and they are never printed.
"""
from __future__ import annotations

import argparse
import base64
import re
import subprocess
import sys
import zlib
from pathlib import Path

from sparvi.config import ROOT

MARK = "DEV-RESULT"
CHUNK = 3000
ALLOWED = re.compile(r"^baseball/[a-z0-9_]+/output/development[a-z0-9_]*\.json$")
LINE = re.compile(rf"{MARK} (\S+) (\d+)/(\d+) ([A-Za-z0-9+/=]+)\s*$")


def emit(files: list[str], out=sys.stdout) -> int:
    for f in files:
        rel = Path(f).resolve().relative_to(ROOT).as_posix()
        if not ALLOWED.match(rel):
            print(f"not a development result file: {rel}", file=sys.stderr)
            return 1
        data = base64.b64encode(zlib.compress(Path(f).read_bytes(), 9)).decode()
        parts = [data[i:i + CHUNK] for i in range(0, len(data), CHUNK)] or [""]
        for i, p in enumerate(parts, 1):
            print(f"{MARK} {rel} {i}/{len(parts)} {p}", file=out)
    return 0


def parse(log: str) -> dict[str, bytes]:
    """{path: bytes} for every complete file in a job log. A file printed twice keeps its last copy."""
    got: dict[str, dict[int, str]] = {}
    total: dict[str, int] = {}
    for line in log.splitlines():
        m = LINE.search(line)
        if not m:
            continue
        rel, i, n, chunk = m[1], int(m[2]), int(m[3]), m[4]
        if not ALLOWED.match(rel):
            raise ValueError(f"the log names a file that is not a development result: {rel}")
        if i == 1:
            got[rel] = {}
        got.setdefault(rel, {})[i] = chunk
        total[rel] = n
    out = {}
    for rel, parts in got.items():
        if sorted(parts) != list(range(1, total[rel] + 1)):
            raise ValueError(f"{rel}: the log holds {len(parts)} of {total[rel]} pieces")
        out[rel] = zlib.decompress(base64.b64decode("".join(parts[i] for i in sorted(parts))))
    return out


def fetch(run_id: str, into: Path) -> int:
    log = subprocess.run(["gh", "run", "view", run_id, "--log"], capture_output=True, text=True, encoding="utf-8",
                         errors="replace", check=True, cwd=ROOT).stdout
    files = parse(log)
    if not files:
        print("no result files in that run's log", file=sys.stderr)
        return 1
    for rel, data in files.items():
        p = into / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        print(f"{p}  ({len(data):,} bytes)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit")
    e.add_argument("files", nargs="+")
    f = sub.add_parser("fetch")
    f.add_argument("run_id")
    f.add_argument("--into", type=Path)
    a = ap.parse_args()
    if a.cmd == "emit":
        return emit(a.files)
    return fetch(a.run_id, a.into or ROOT / "dev_results" / a.run_id)


if __name__ == "__main__":
    sys.exit(main())
