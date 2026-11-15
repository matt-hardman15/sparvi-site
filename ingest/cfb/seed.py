"""Seed for the CFB raw-data cache: data/raw/cfb (parquet files and CFBD's cached responses) as one archive in the
private R2 bucket, so a GitHub Actions run with no cache restores it instead of re-pulling every season from CFBD
(thousands of calls). Raw data stays private: R2 only, never committed or published (CFBD terms).

    python -m ingest.cfb.seed upload     # from a machine with the full data/raw/cfb (writes a new dated object)
    python -m ingest.cfb.seed restore    # in Actions, when the cache missed: the newest seed into data/raw/cfb

Objects are seeds/cfb/raw-<UTC timestamp>.tar.gz, written once and never overwritten (put_new), like the snapshots.
"""
from __future__ import annotations

import io
import sys
import tarfile
from datetime import UTC, datetime

from ingest.common.snapshots import store_from_env
from sparvi.config import RAW

PREFIX = "seeds/cfb/"


def upload() -> int:
    store = store_from_env()
    if store is None:
        print("R2 is not configured; nothing uploaded.")
        return 1
    src = RAW / "cfb"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                tar.add(p, arcname=p.relative_to(src).as_posix())
    key = f"{PREFIX}raw-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.tar.gz"
    store.put_new(key, buf.getvalue())
    print(f"uploaded {key} ({buf.tell() / 1e6:.0f} MB)")
    return 0


def restore() -> int:
    store = store_from_env()
    if store is None:
        print("R2 is not configured; no seed restored.")
        return 1
    keys = [k for k in store.list(PREFIX) if k.endswith(".tar.gz")]
    if not keys:
        print("no CFB seed in R2")
        return 1
    data = store.get(keys[-1])
    dst = RAW / "cfb"
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dst, filter="data")
    n = sum(1 for p in dst.rglob("*") if p.is_file())
    print(f"restored {keys[-1]}: {n} files into data/raw/cfb")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"upload": upload, "restore": restore}.get(cmd, lambda: (print(__doc__), 2)[1])())
