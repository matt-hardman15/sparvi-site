"""History seasons for a machine that never pulled them (PQ-TEST-2, M5-EB): the latest state of a season's Statcast
table, read from the private R2 snapshots into data/raw.

The refresh pulls only the seasons the site shows. The Stuff+ backtest trains on two seasons, the earlier of which
is a history season (sparvi.toml `history_first`), so the backtest workflow restores it from the snapshots that
`pull.py --history` and `python -m ingest.snapshot baseball` wrote. Raw data stays private: R2 and data/raw only.

The Triple-A translation's backtest fits on Triple-A seasons before the site's window (`aaa_history_first`) the same way.

    python -m ingest.baseball.history restore 2023             # MLB; skips a season whose file is already there
    python -m ingest.baseball.history restore aaa:2023 aaa:2024
"""
from __future__ import annotations

import sys

import pandas as pd

from ingest.baseball.pull import KEY, ORDER
from ingest.common.asof import read_asof
from ingest.common.snapshots import now_utc, store_from_env
from sparvi.config import RAW, baseball


def restore(seasons: list[int], store=None, raw=RAW, level: str = "mlb") -> int:
    """Write statcast_<level>_<season>.parquet for each season that isn't on the machine. Returns 0 when every season
    is there afterwards."""
    cfg = baseball()
    history = cfg.history_seasons if level == "mlb" else cfg.aaa_history_seasons
    bad = [s for s in seasons if s not in history]
    if bad:
        print(f"not {level} history seasons: {bad} (history is {history})")
        return 2
    todo = [s for s in seasons if not (raw / f"statcast_{level}_{s}.parquet").exists()]
    if not todo:
        print(f"{level} history seasons {seasons}: already in data/raw")
        return 0
    store = store or store_from_env()
    if store is None:
        print(f"R2 is not configured; {level} history seasons {todo} not restored.")
        return 1
    as_of = pd.Timestamp(now_utc()).tz_localize(None)
    for s in todo:
        df, meta = read_asof(store, "baseball", f"statcast_{level}_{s}", as_of)
        # The pull's own row order (training reads rows in file order), whatever order the snapshots came in.
        df = df.sort_values(KEY, kind="stable").sort_values(ORDER, kind="stable").reset_index(drop=True)
        raw.mkdir(parents=True, exist_ok=True)
        df.to_parquet(raw / f"statcast_{level}_{s}.parquet", index=False)
        print(f"restored statcast_{level}_{s}: {len(df):,} pitches (first snapshot {meta['first_snapshot'][:10]})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "restore":
        print(__doc__)
        sys.exit(2)
    args = [a.split(":") if ":" in a else ["mlb", a] for a in sys.argv[2:]]
    sys.exit(max(restore([int(s) for lv, s in args if lv == level], level=level) for level in sorted({lv for lv, _ in args})))
