"""Fingerprint of everything the baseball build reads: raw data files and the code that turns them
into site data.

    python scripts/build_fingerprint.py           # prints the fingerprint
    python scripts/build_fingerprint.py --check   # exit 0 if it equals the stored one, 1 if not
    python scripts/build_fingerprint.py --write   # store it (the refresh does this when it commits)

Model training is repeatable on one machine but not bit-for-bit across CPU models, so rebuilding
unchanged data on a different GitHub runner can still move published numbers in the last digits.
The refresh skips the rebuild when this fingerprint matches the one stored with the last refresh.
Raw files are hashed by content; code by git blob id (committed outputs are left out).

Beside the fingerprint, `--write` stores each input's own hash (baseball/build_inputs.json: file names and hashes,
no data), and a `--check` that doesn't match prints which inputs were added, removed or changed since (FP-DIAG).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys

from sparvi.config import ROOT

STORED = ROOT / "baseball" / "build_inputs.sha256"
INPUTS = ROOT / "baseball" / "build_inputs.json"     # each input's own hash, so a mismatch can name what changed
CODE = ["baseball", "sparvi", "build.sh", "sparvi.toml", "pyproject.toml", "uv.lock"]
EXCLUDE = [":(exclude,glob)baseball/*/output/**", f":(exclude){STORED.relative_to(ROOT).as_posix()}",
           f":(exclude){INPUTS.relative_to(ROOT).as_posix()}"]
RAW_PREFIX = "data/raw/"


def inputs() -> tuple[dict[str, str], str]:
    """Every input's hash ({"data/raw/<file>": sha256, "<code path>": "<mode> <git blob id>"}) and the git listing the
    fingerprint hashes."""
    raw = ROOT / "data" / "raw"
    out = {RAW_PREFIX + p.name: hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(x for x in raw.iterdir() if x.is_file())}   # top level only: data/raw/nfl is football's
    code = subprocess.run(["git", "ls-files", "-s", "--", *CODE, *EXCLUDE], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout
    for line in code.splitlines():
        meta, path = line.split("\t", 1)
        mode, blob, _stage = meta.split()
        out[path] = f"{mode} {blob}"
    return out, code


def fingerprint(parts: tuple[dict[str, str], str] | None = None) -> str:
    per_input, code = parts or inputs()
    h = hashlib.sha256()
    for name, digest in per_input.items():
        if name.startswith(RAW_PREFIX):
            h.update(f"{name.removeprefix(RAW_PREFIX)}\0{digest}\n".encode())
    h.update(code.encode())
    return h.hexdigest()


def differences(old: dict[str, str], new: dict[str, str]) -> list[str]:
    """One line per input that was added, removed or changed between two records."""
    return ([f"added: {k}" for k in sorted(new.keys() - old.keys())] + [f"removed: {k}" for k in sorted(old.keys() - new.keys())]
            + [f"changed: {k}" for k in sorted(old.keys() & new.keys()) if old[k] != new[k]])


def explain(new: dict[str, str]) -> list[str]:
    """What to print when the fingerprint doesn't match the stored one."""
    if not INPUTS.exists():
        return [f"no per-input record yet ({INPUTS.name} is written by the next rebuild): can't say which input changed"]
    diff = differences(json.loads(INPUTS.read_text(encoding="utf-8")), new)
    return diff or ["no input differs from the per-input record: the stored fingerprint and that record disagree "
                    "(written by different runs, or the fingerprint's recipe changed)"]


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    a = ap.parse_args()
    parts = inputs()
    fp = fingerprint(parts)
    if a.write:
        STORED.write_text(fp + "\n", encoding="utf-8", newline="\n")
        INPUTS.write_text(json.dumps(parts[0], indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    elif a.check:
        old = STORED.read_text(encoding="utf-8").strip() if STORED.exists() else ""
        print("unchanged" if fp == old else "changed")
        if fp != old:
            print("\n".join(explain(parts[0])))
        return 0 if fp == old else 1
    else:
        print(fp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
