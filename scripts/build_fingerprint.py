"""Fingerprint of everything the baseball build reads: raw data files and the code that turns them
into site data.

    python scripts/build_fingerprint.py           # prints the fingerprint
    python scripts/build_fingerprint.py --check   # exit 0 if it equals the stored one, 1 if not
    python scripts/build_fingerprint.py --write   # store it (the refresh does this when it commits)

Model training is repeatable on one machine but not bit-for-bit across CPU models, so rebuilding
unchanged data on a different GitHub runner can still move published numbers in the last digits.
The refresh skips the rebuild when this fingerprint matches the one stored with the last refresh.
Raw files are hashed by content; code by git blob id (committed outputs are left out).
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys

from sparvi.config import ROOT

STORED = ROOT / "baseball" / "build_inputs.sha256"
CODE = ["baseball", "sparvi", "build.sh", "sparvi.toml", "pyproject.toml", "uv.lock"]
EXCLUDE = [":(exclude,glob)baseball/*/output/**", f":(exclude){STORED.relative_to(ROOT).as_posix()}"]


def fingerprint() -> str:
    h = hashlib.sha256()
    raw = ROOT / "data" / "raw"
    for p in sorted(x for x in raw.iterdir() if x.is_file()):   # top level only: data/raw/nfl is football's
        h.update(f"{p.name}\0{hashlib.sha256(p.read_bytes()).hexdigest()}\n".encode())
    code = subprocess.run(["git", "ls-files", "-s", "--", *CODE, *EXCLUDE], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout
    h.update(code.encode())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    a = ap.parse_args()
    fp = fingerprint()
    if a.write:
        STORED.write_text(fp + "\n", encoding="utf-8", newline="\n")
    elif a.check:
        old = STORED.read_text(encoding="utf-8").strip() if STORED.exists() else ""
        print("unchanged" if fp == old else "changed")
        return 0 if fp == old else 1
    else:
        print(fp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
