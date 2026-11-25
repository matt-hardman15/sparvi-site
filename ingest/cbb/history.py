"""Past college basketball seasons for a machine that never pulled them (CBB-0): the latest state of each history
table, read from the private R2 snapshots into data/raw/cbb.

The daily workflow pulls only the current season; the past seasons (CBB-HIST) were pulled once by hand and snapshotted.
The backtest workflow restores the tables the team model reads, so its registered test can run in CI without any
API calls. Raw data stays private: R2 and data/raw only.

    python -m ingest.cbb.history restore                 # every history season, the model's tables
    python -m ingest.cbb.history restore 2025 2026       # those seasons only; a file already there is left alone
"""
from __future__ import annotations

import sys

import pandas as pd

from ingest.common.asof import read_asof
from ingest.common.snapshots import now_utc, store_from_env
from sparvi.config import RAW, cbb

TABLES = ("games", "team_games", "player_games", "lines", "portal")      # what cbb/ratings reads (lines: aggregate scoring only)


def restore(seasons: list[int] | None = None, store=None, raw=RAW, tables: tuple[str, ...] = TABLES) -> int:
    """Write <table>_<season>.parquet for each history season's table that isn't on the machine. Returns 0 when every
    one is there afterwards."""
    history = cbb().history_seasons
    seasons = history if not seasons else seasons
    bad = [s for s in seasons if s not in history]
    if bad:
        print(f"not history seasons: {bad} (history is {history})")
        return 2
    d = raw / "cbb"
    todo = [(t, s) for s in seasons for t in tables if not (d / f"{t}_{s}.parquet").exists()]
    if not todo:
        print(f"cbb history seasons {seasons}: already in data/raw/cbb")
        return 0
    store = store or store_from_env()
    if store is None:
        print(f"R2 is not configured; {len(todo)} cbb history tables not restored.")
        return 1
    as_of = pd.Timestamp(now_utc()).tz_localize(None)
    d.mkdir(parents=True, exist_ok=True)
    missing = []
    for t, s in todo:
        try:
            df, meta = read_asof(store, "cbb", f"cbb_{t}_{s}", as_of)
        except FileNotFoundError:
            missing.append(f"cbb_{t}_{s}")
            continue
        df.to_parquet(d / f"{t}_{s}.parquet", index=False)
        print(f"restored cbb_{t}_{s}: {len(df):,} rows (first snapshot {meta['first_snapshot'][:10]})")
    if missing:
        print(f"no snapshots for: {', '.join(missing)}")
    return 1 if missing else 0


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "restore":
        print(__doc__)
        sys.exit(2)
    sys.exit(restore([int(a) for a in sys.argv[2:]]))
