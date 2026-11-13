"""Point-in-time data access for backtests (PLAN-backtest §2). The only way model code reads data.

    loader = Loader()                                   # local raw files + the snapshot available_at rules
    df = loader.load("statcast_mlb_2025", as_of="2026-03-01")

Every table is defined once, in ingest/snapshot.py (path, key, available_at rule). Two kinds:
- event tables (pitches, plays, games) have an `available_at` per row (e.g. the day after the game);
  `load` returns only rows with available_at <= as_of.
- attribute tables (heights, rosters, teams, affiliates) have no event time; the source overwrites
  old values. Before snapshots existed they're returned as they are now and flagged `approximate`.

Revision safety: snapshots of what the sources said started on SNAPSHOTS_START. For an `as_of`
before that, the loader can only show the data as first recorded, not as it was then, so the
result is marked revision_safe = False (and every backtest report says so). With source="r2" the
loader reads the snapshots themselves (ingest/common/asof.py), which is revision-safe after the
first snapshot.

The loader also keeps a ledger of what it served (table, as_of, rows, the latest available_at it
handed out), which the leakage tests audit, and a hook (`poison`) that the future-poisoning test
uses to corrupt every row after a date.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from ingest.snapshot import TABLES, Table

SNAPSHOTS_START = pd.Timestamp("2026-11-13")       # first snapshot (A5a); earlier history is not revision-safe


def _all_tables() -> dict[str, tuple[str, Table]]:
    out = {}
    for sport, fn in TABLES.items():
        for t in fn():
            out[t.name] = (sport, t)
    return out


@dataclass
class Served:
    table: str
    as_of: pd.Timestamp
    rows: int
    max_available_at: pd.Timestamp | None
    kind: str


@dataclass
class Loader:
    source: str = "raw"                                   # "raw" (local files) or "r2" (the snapshots)
    store: object | None = None                           # snapshot store when source == "r2"
    poison: Callable[[pd.DataFrame, pd.Timestamp], pd.DataFrame] | None = None   # leakage test hook
    served: list[Served] = field(default_factory=list)

    def __post_init__(self):
        self.tables = _all_tables()

    def load(self, table: str, as_of, columns: list[str] | None = None) -> pd.DataFrame:
        as_of = pd.Timestamp(as_of)
        sport, t = self.tables[table]
        if self.source == "r2":
            from ingest.common.asof import read_asof
            df, _ = read_asof(self.store, sport, table, as_of)
            avail = None
        else:
            df = pd.read_parquet(t.path, columns=None)
            avail = t.available_at(df, pd.Timestamp(as_of, tz="UTC")) if t.available_at else None
        kind = "event" if t.available_at else "attribute"
        if avail is not None:
            avail = pd.to_datetime(avail)
            if self.poison is not None:
                df = self.poison(df, avail, as_of)
            keep = (avail <= as_of).to_numpy()
            df, avail = df[keep], avail[keep]
        if columns is not None:
            df = df[columns]
        self.served.append(Served(table, as_of, len(df), (avail.max() if avail is not None and len(avail) else None), kind))
        return df.reset_index(drop=True)

    def revision_safe(self) -> bool:
        """True only if every read came from snapshots taken before its as_of date."""
        if self.source != "r2":
            return False          # local files show the data as last pulled, not as it was on as_of
        return all(s.as_of >= SNAPSHOTS_START for s in self.served)

    def approximate_tables(self) -> list[str]:
        return sorted({s.table for s in self.served if s.kind == "attribute" and s.as_of < SNAPSHOTS_START})

    def max_available_at(self, since: int = 0) -> pd.Timestamp | None:
        ts = [s.max_available_at for s in self.served[since:] if s.max_available_at is not None]
        return max(ts) if ts else None
