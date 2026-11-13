"""Provenance block for model outputs: which data, which library versions, which seeds.

Kept deterministic on purpose (no timestamps, no git SHA), so a rebuild on unchanged
data produces byte-identical outputs and the daily refresh only commits real changes.
"""
from __future__ import annotations

import platform
from importlib import metadata
from pathlib import Path

import duckdb

PACKAGES = ["pandas", "numpy", "duckdb", "lightgbm", "scikit-learn", "dbt-duckdb"]


def library_versions() -> dict[str, str]:
    out = {"python": ".".join(platform.python_version_tuple()[:2])}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def data_summary(files: list[Path], date_expr: str = "game_date") -> dict[str, dict]:
    """Row count and game-date range for each raw file used (`date_expr`: the date column or expression)."""
    out = {}
    for f in sorted(files):
        rows, lo, hi = duckdb.sql(
            f"select count(*), min({date_expr})::date::varchar, max({date_expr})::date::varchar "
            f"from read_parquet('{f.as_posix()}')"
        ).fetchone()
        out[f.name] = {"rows": int(rows), "first_game": lo, "last_game": hi}
    return out


def provenance(files: list[Path], seeds: dict[str, int] | None = None, date_expr: str = "game_date", **extra) -> dict:
    return {"data": data_summary(files, date_expr), "libraries": library_versions(), "seeds": seeds or {}, **extra}
