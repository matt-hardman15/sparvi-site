"""Append-only snapshots of raw data (PLAN-backtest §2, milestone A5a).

After each pull passes its data checks, every raw table is compared with its last snapshot and only
the differences are stored, in a private Cloudflare R2 bucket:

    snapshots/<sport>/<table>/<stamp>.parquet           new and changed rows (all columns, plus
                                                        _op = insert | update, _ingested_at,
                                                        _available_at, _row_hash)
    snapshots/<sport>/<table>/<stamp>.deletes.parquet   keys of rows that disappeared from the source
    snapshots/<sport>/<table>/manifest.jsonl            one line per snapshot (counts, files)
    state/<sport>/<table>.parquet                       key + row hash of the latest version (to diff)

`_ingested_at` is when Sparvi pulled the row; `_available_at` is the earliest moment Sparvi could
have known it (e.g. the day after the game). Together they answer "what did the data say on date
D?" (ingest/common/asof.py), which is what makes backtests revision-safe. Snapshot files are never
overwritten (conditional writes); a fix is a new snapshot. The very first snapshot of a table holds
history that predates the record, so as-of reads before it are flagged `revision_safe: false`.

The working files in data/raw/ are untouched: the pipelines keep reading them as before.
"""
from __future__ import annotations

import io
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sparvi.config import ROOT

HASH_VERSION = 1                 # bump if the row-hash recipe changes (it would look like a mass revision)
MAX_UPDATE_SHARE = 0.20          # more than this share of existing rows changing at once is refused...
MIN_MASS_UPDATE = 500            # ...if it's also more than this many rows (small tables change wholesale)


# ---- storage --------------------------------------------------------------------------------------
class SnapshotExists(RuntimeError):
    pass


class LocalStore:
    """A folder standing in for the bucket (tests, dry runs)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def get(self, key: str) -> bytes | None:
        p = self.root / key
        return p.read_bytes() if p.exists() else None

    def put(self, key: str, data: bytes) -> None:
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def put_new(self, key: str, data: bytes) -> None:
        p = self.root / key
        if p.exists():
            raise SnapshotExists(key)
        self.put(key, data)

    def list(self, prefix: str) -> list[str]:
        base = self.root / prefix
        return sorted(p.relative_to(self.root).as_posix() for p in base.rglob("*") if p.is_file()) if base.exists() else []


class R2Store:
    """The private R2 bucket (S3 API). Credentials from the environment or the local .env."""

    def __init__(self, endpoint: str, bucket: str, key_id: str, secret: str):
        import boto3
        self.bucket = bucket
        self.s3 = boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=key_id,
                               aws_secret_access_key=secret, region_name="auto")

    def get(self, key: str) -> bytes | None:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    def put(self, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)

    def put_new(self, key: str, data: bytes) -> None:
        from botocore.exceptions import ClientError
        try:
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, IfNoneMatch="*")
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("PreconditionFailed", "412"):
                raise SnapshotExists(key) from e
            raise

    def list(self, prefix: str) -> list[str]:
        keys, token = [], None
        while True:
            kw = {"Bucket": self.bucket, "Prefix": prefix} | ({"ContinuationToken": token} if token else {})
            r = self.s3.list_objects_v2(**kw)
            keys += [o["Key"] for o in r.get("Contents", [])]
            if not r.get("IsTruncated"):
                return sorted(keys)
            token = r["NextContinuationToken"]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    return env


def store_from_env() -> R2Store | None:
    e = _env()
    need = ("R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
    if not all(e.get(k) for k in need):
        return None
    return R2Store(e["R2_ENDPOINT"], e["R2_BUCKET"], e["R2_ACCESS_KEY_ID"], e["R2_SECRET_ACCESS_KEY"])


# ---- tables ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Table:
    name: str                                     # e.g. statcast_mlb_2026
    path: Path                                    # the working file in data/raw/
    key: tuple[str, ...]                          # one row per key (the data checks enforce it)
    available_at: Callable[[pd.DataFrame, pd.Timestamp], pd.Series] | None = None   # None: when first seen
    drop: tuple[str, ...] = ()                    # columns not worth versioning (e.g. the pull date)


def day_after(col: str):
    """available_at = the calendar day after the event date (overnight publication)."""
    def f(df: pd.DataFrame, ingested: pd.Timestamp) -> pd.Series:
        return pd.to_datetime(df[col]).dt.normalize() + pd.Timedelta(days=1)
    return f


def row_hashes(df: pd.DataFrame) -> np.ndarray:
    """A stable 64-bit hash of each row's values (column order as in the file)."""
    return pd.util.hash_pandas_object(df, index=False, categorize=False).to_numpy(dtype="uint64")


def _parquet(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd")
    return buf.getvalue()


def _read(data: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(data))


def stamp(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%dT%H%M%SZ")


def snapshot(store, sport: str, t: Table, ingested_at: pd.Timestamp, allow_mass_update: bool = False) -> dict:
    """Store what changed in table `t` since its last snapshot. Returns a summary dict."""
    df = pd.read_parquet(t.path).drop(columns=list(t.drop)).reset_index(drop=True)
    key = list(t.key)
    if df.duplicated(key).any():
        raise ValueError(f"{t.name}: duplicate keys {key}; the data checks should have stopped this pull")
    cur = df[key].copy()
    cur["_row_hash"] = row_hashes(df)

    state_key = f"state/{sport}/{t.name}.parquet"
    prev_bytes = store.get(state_key)
    baseline = prev_bytes is None
    if baseline:
        ins, upd = np.arange(len(df)), np.array([], dtype=int)
        deleted = cur.iloc[0:0][key]
    else:
        prev = _read(prev_bytes)
        m = cur.reset_index().merge(prev, on=key, how="outer", suffixes=("", "_prev"), indicator=True)
        ins = m.loc[m["_merge"] == "left_only", "index"].astype(int).to_numpy()
        both = m[m["_merge"] == "both"]
        upd = both.loc[both["_row_hash"] != both["_row_hash_prev"], "index"].astype(int).to_numpy()
        deleted = m.loc[m["_merge"] == "right_only", key]
        if not allow_mass_update and len(upd) > max(MAX_UPDATE_SHARE * len(prev), MIN_MASS_UPDATE):
            raise RuntimeError(f"{t.name}: {len(upd):,} of {len(prev):,} rows changed at once. That's more like a "
                               "format or library change than real revisions; rerun with --allow-mass-update if it's real.")

    summary = {"table": t.name, "ingested_at": ingested_at.isoformat(), "baseline": baseline, "rows": len(df),
               "inserts": int(len(ins)), "updates": int(len(upd)), "deletes": int(len(deleted)), "files": []}
    if not (len(ins) or len(upd) or len(deleted)):
        return summary

    prefix = f"snapshots/{sport}/{t.name}/{stamp(ingested_at)}"
    rows = np.concatenate([ins, upd])
    if len(rows):
        out = df.iloc[rows].copy()
        out["_op"] = ["insert"] * len(ins) + ["update"] * len(upd)
        out["_ingested_at"] = ingested_at.tz_localize(None)
        out["_available_at"] = (t.available_at(out, ingested_at) if t.available_at else
                                pd.Series(ingested_at.tz_localize(None).normalize(), index=out.index))
        out["_available_at"] = pd.to_datetime(out["_available_at"])
        out["_row_hash"] = cur["_row_hash"].to_numpy()[rows]
        store.put_new(prefix + ".parquet", _parquet(out))
        summary["files"].append(prefix + ".parquet")
    if len(deleted):
        d = deleted.copy()
        d["_op"], d["_ingested_at"] = "delete", ingested_at.tz_localize(None)
        store.put_new(prefix + ".deletes.parquet", _parquet(d))
        summary["files"].append(prefix + ".deletes.parquet")

    # State after the snapshot file: a crash in between only means the next run records the same
    # change again under a later stamp, which as-of reads resolve the same way.
    store.put(state_key, _parquet(cur))
    man_key = f"snapshots/{sport}/{t.name}/manifest.jsonl"
    old = (store.get(man_key) or b"").decode("utf-8")
    line = json.dumps(summary | {"key": key, "hash_version": HASH_VERSION}, separators=(",", ":"))
    store.put(man_key, (old + line + "\n").encode("utf-8"))
    return summary


def now_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(UTC)).floor("s")
