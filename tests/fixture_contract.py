"""Make a synthetic fixture table match its ingest contract (contracts/<sport>_ingest.yml): the
contract's column types, and the columns the fixture doesn't simulate added as typed nulls. CI then
runs the pipeline on files shaped like the real ones, and the ingest checks pass on them."""
from __future__ import annotations

import duckdb
import pandas as pd
import yaml

from sparvi.config import ROOT


def contract_columns(sport: str, table: str) -> dict[str, str]:
    c = yaml.safe_load((ROOT / "contracts" / f"{sport}_ingest.yml").read_text(encoding="utf-8"))
    return c["tables"][table]["columns"]


def write_conformed(df: pd.DataFrame, path, sport: str, table: str) -> None:
    """Write `df` to parquet with the contract's types (extra fixture columns are kept as they are)."""
    want = contract_columns(sport, table)
    con = duckdb.connect()
    con.register("df", df)
    have = set(df.columns)
    sel = [f'cast("{c}" as {t}) as "{c}"' if c in have else f'cast(null as {t}) as "{c}"' for c, t in want.items()]
    sel += [f'"{c}"' for c in df.columns if c not in want]
    con.sql(f"copy (select {', '.join(sel)} from df) to '{path.as_posix()}' (format parquet)")
    con.close()
