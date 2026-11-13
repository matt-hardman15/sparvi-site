"""Read a snapshotted table as it stood at a moment (minimal as-of reader; A5's `asof.load` builds on it).

    df, meta = read_asof(store, "baseball", "statcast_mlb_2026", pd.Timestamp("2026-06-01"))

Rules (PLAN-backtest §2):
- A row version counts if Sparvi had ingested it by `as_of` and it was available by `as_of`
  (`_available_at <= as_of`). The latest such version of each key wins; a later delete removes it.
- The table's first snapshot holds history from before the record started. For dates before that
  snapshot, its rows count from their `_available_at` (there's nothing better), and the result is
  flagged `revision_safe: False`: it shows the data as first recorded, not necessarily as it was then.
"""
from __future__ import annotations

import io

import pandas as pd


def _read(store, key: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(store.get(key)))


def read_asof(store, sport: str, table: str, as_of: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    as_of = pd.Timestamp(as_of).tz_localize(None) if pd.Timestamp(as_of).tzinfo else pd.Timestamp(as_of)
    keys = [k for k in store.list(f"snapshots/{sport}/{table}/") if k.endswith(".parquet")]
    if not keys:
        raise FileNotFoundError(f"no snapshots for {sport}/{table}")
    stamps = sorted({k.rsplit("/", 1)[1].split(".")[0] for k in keys})
    first = pd.Timestamp(stamps[0].replace("Z", ""))
    versions, deletes = [], []
    for k in keys:
        st = pd.Timestamp(k.rsplit("/", 1)[1].split(".")[0].replace("Z", ""))
        baseline = st == first
        if not baseline and st > as_of:
            continue
        df = _read(store, k)
        if k.endswith(".deletes.parquet"):
            deletes.append(df)
        else:
            if baseline:
                df["_ingested_at"] = df["_available_at"].where(df["_ingested_at"] > as_of, df["_ingested_at"])
            versions.append(df)
    v = pd.concat(versions, ignore_index=True)
    v = v[v["_available_at"] <= as_of]
    meta_cols = ["_op", "_ingested_at", "_available_at", "_row_hash"]
    key = [c for c in deletes[0].columns if c not in ("_op", "_ingested_at")] if deletes else None
    if key is None:
        # The key is whatever identifies rows in the state; recover it from the manifest.
        import json
        man = (store.get(f"snapshots/{sport}/{table}/manifest.jsonl") or b"").decode().splitlines()
        key = json.loads(man[0])["key"]
    v = v.sort_values("_ingested_at", kind="stable").drop_duplicates(key, keep="last")
    if deletes:
        d = pd.concat(deletes, ignore_index=True)
        d = d[d["_ingested_at"] <= as_of].sort_values("_ingested_at").drop_duplicates(key, keep="last")
        m = v.merge(d[key + ["_ingested_at"]].rename(columns={"_ingested_at": "_deleted_at"}), on=key, how="left")
        v = m[m["_deleted_at"].isna() | (m["_deleted_at"] < m["_ingested_at"])].drop(columns="_deleted_at")
    out = v.drop(columns=meta_cols).reset_index(drop=True)
    return out, {"as_of": as_of.isoformat(), "revision_safe": bool(as_of >= first), "first_snapshot": first.isoformat(),
                 "rows": len(out)}
