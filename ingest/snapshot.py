"""Snapshot a sport's raw tables to the private R2 bucket (A5a). Run after the data checks pass.

    python -m ingest.snapshot baseball            # skips quietly if R2 isn't configured
    python -m ingest.snapshot nfl --require       # (workflows) fail if R2 isn't configured
    python -m ingest.snapshot baseball --local DIR    # write to a folder instead (dry run)

What each table's `available_at` means is set here (PLAN-backtest §2): the day after the game for
game data; for everything else, when Sparvi first saw the row.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from ingest.common.snapshots import LocalStore, Table, day_after, now_utc, snapshot, store_from_env
from sparvi.config import RAW, baseball, cfb, nfl


def baseball_tables() -> list[Table]:
    cfg = baseball()
    t = [Table(f"statcast_{lvl}_{s}", RAW / f"statcast_{lvl}_{s}.parquet",
               ("game_pk", "at_bat_number", "pitch_number"), day_after("game_date"))
         for lvl, seasons in (("aaa", cfg.aaa_seasons), ("mlb", cfg.mlb_seasons)) for s in seasons]
    t.append(Table("batter_heights", RAW / "batter_heights.parquet", ("batter",)))
    t.append(Table(f"milb_players_{cfg.season}", RAW / f"milb_players_{cfg.season}.parquet", ("player_id", "level", "team_id")))
    t += [Table(f"aaa_affiliates_{s}", RAW / f"aaa_affiliates_{s}.parquet", ("team_id",)) for s in cfg.aaa_seasons]
    rosters = sorted(RAW.glob("rosters_40man_*.parquet"))       # dated files from pull_rosters.py; newest wins
    if rosters:
        t.append(Table("rosters_40man", rosters[-1], ("player_id",), drop=("snapshot_date",)))
    return t


def _schedule_available(df: pd.DataFrame, ingested: pd.Timestamp) -> pd.Series:
    """A result is known the day after the game; an unplayed game is known when first seen."""
    after = pd.to_datetime(df["gameday"]).dt.normalize() + pd.Timedelta(days=1)
    first_seen = pd.Series(ingested.tz_localize(None).normalize(), index=df.index)
    return after.where(df["home_score"].notna(), first_seen)


def nfl_tables() -> list[Table]:
    cfg = nfl()
    d = RAW / "nfl"
    t = [Table(f"pbp_{s}", d / f"pbp_{s}.parquet", ("game_id", "play_id"), day_after("game_date")) for s in cfg.seasons]
    t += [Table(f"team_stats_{s}", d / f"team_stats_{s}.parquet", ("season", "team", "season_type")) for s in cfg.seasons]
    # The source's own team codes (ingest/nfl/pull.py normalizes the working copies).
    t.append(Table("schedules", d / "source" / "schedules.parquet", ("game_id",), _schedule_available))
    t.append(Table("teams", d / "source" / "teams.parquet", ("team_abbr",)))
    return t


def _game_available(df: pd.DataFrame, ingested: pd.Timestamp) -> pd.Series:
    """A finished game's result is known the day after; an unplayed game when first seen."""
    after = pd.to_datetime(df["startDate"], utc=True).dt.tz_localize(None).dt.normalize() + pd.Timedelta(days=1)
    first_seen = pd.Series(ingested.tz_localize(None).normalize(), index=df.index)
    return after.where(df["completed"].fillna(False).astype(bool), first_seen)


def _plays_available(games_path):
    """Plays carry no date; take the day after their game's start (from that season's games file)."""
    def f(df: pd.DataFrame, ingested: pd.Timestamp) -> pd.Series:
        g = pd.read_parquet(games_path, columns=["id", "startDate"])
        start = pd.to_datetime(g["startDate"], utc=True).dt.tz_localize(None).dt.normalize() + pd.Timedelta(days=1)
        out = df["gameId"].map(dict(zip(g["id"], start, strict=True)))
        return out.fillna(ingested.tz_localize(None).normalize())
    return f


def cfb_tables() -> list[Table]:
    cfg = cfb()
    d = RAW / "cfb"
    t = []
    for s in cfg.seasons:
        t += [Table(f"plays_{s}", d / f"plays_{s}.parquet", ("id",), _plays_available(d / f"games_{s}.parquet")),
              Table(f"games_{s}", d / f"games_{s}.parquet", ("id",), _game_available),
              Table(f"teams_{s}", d / f"teams_{s}.parquet", ("id",)),
              Table(f"rankings_{s}", d / f"rankings_{s}.parquet", ("season", "seasonType", "week", "poll", "school")),
              # Lines move until kickoff; each version is recorded when first seen. Benchmark only.
              Table(f"lines_{s}", d / f"lines_{s}.parquet", ("game_id", "provider"))]
    return t


TABLES = {"baseball": baseball_tables, "nfl": nfl_tables, "cfb": cfb_tables}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sport", choices=sorted(TABLES))
    ap.add_argument("--require", action="store_true", help="fail if R2 isn't configured")
    ap.add_argument("--local", help="write to this folder instead of R2")
    ap.add_argument("--allow-mass-update", action="store_true", help="accept >20%% of a table's rows changing at once")
    a = ap.parse_args()

    store = LocalStore(a.local) if a.local else store_from_env()
    if store is None:
        print("R2 is not configured (R2_ENDPOINT, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY); no snapshot taken.")
        return 1 if a.require else 0
    ingested = now_utc()
    failed = False
    for t in TABLES[a.sport]():
        if not t.path.exists():
            print(f"  {t.name:<24} missing ({t.path.name}), skipped")
            continue
        try:
            s = snapshot(store, a.sport, t, ingested, allow_mass_update=a.allow_mass_update)
        except Exception as e:                       # keep going: one table's problem shouldn't drop the others
            failed = True
            print(f"  {t.name:<24} FAILED: {e}")
            continue
        what = "first snapshot" if s["baseline"] else ("no change" if not s["files"] else "changes")
        print(f"  {t.name:<24} {s['rows']:>9,} rows  +{s['inserts']:,} ~{s['updates']:,} -{s['deletes']:,}  {what}")
    print(f"{a.sport} snapshot {ingested.isoformat()}: {'with failures' if failed else 'done'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
