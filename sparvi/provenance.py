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


def data_summary(files: list[Path]) -> dict[str, dict]:
    """Row count and game-date range for each raw Statcast file used."""
    out = {}
    for f in sorted(files):
        rows, lo, hi = duckdb.sql(
            f"select count(*), min(game_date)::date::varchar, max(game_date)::date::varchar "
            f"from read_parquet('{f.as_posix()}')"
        ).fetchone()
        out[f.name] = {"rows": int(rows), "first_game": lo, "last_game": hi}
    return out


def provenance(files: list[Path], seeds: dict[str, int] | None = None, **extra) -> dict:
    return {"data": data_summary(files), "libraries": library_versions(), "seeds": seeds or {}, **extra}
