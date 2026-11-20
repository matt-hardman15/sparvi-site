"""Pull the current college basketball season from CollegeBasketballData.com (CBB-0a): ingest and private snapshots
only, no pages yet. Snapshots can't be made after the fact, and several of these tables are overwritten at the source
(ratings, lines, rosters), so the pull started in the season's first month.

    python ingest/cbb/pull.py --estimate     # how many uncached calls a run needs (no calls)
    python ingest/cbb/pull.py                # pull (asks for --yes above the approval line)

Output (data/raw/cbb/, never committed; raw data is not redistributed), one file per table for the season in
sparvi.toml [cbb]:
    teams_{season}             Division I teams and conferences
    games_{season}             every game involving one of them: status, score, pre- and post-game Elo
    team_games_{season}        team box score and advanced metrics, one row per game x team
    player_games_{season}      player box score, one row per game x player
    lines_{season}             one row per game x sportsbook (benchmark only, never per game on the site)
    rankings_{season}          AP and Coaches polls by week
    ratings_adjusted_{season}, ratings_srs_{season}, ratings_elo_{season}
                               the source's own team ratings as of the pull (it overwrites them)
    roster_{season}            one row per team x player
    portal_{season}            transfer portal entries (empty until the spring window)
    cache/, _calls.csv         raw responses and the call log (ingest/cfb/client.py)

The API returns at most 3,000 games a call (1,000 for player box scores), so games are pulled by calendar month and
player box scores by week; a window that comes back at its cap stops the pull. A window is cached for good a week
after it ends. Columns and types are cast to contracts/cbb_ingest.yml: a column the contract doesn't list is not
kept, and a contract column the source stops sending fails the checks.

Calls: about a dozen a run. The quota (5,000 a month) is shared with college football; the pull stops at the
reserve in sparvi.toml [cbb].

Terms (effective Aug 12 2026): private caching and storage are permitted; no bulk redistribution of raw data; the
key never reaches the browser. Credit "Data provided by CollegeBasketballData.com" where the data is used.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta

import pandas as pd
import yaml

from ingest.cfb.client import BudgetExhausted, Client
from sparvi.config import RAW, ROOT, cbb

BASE = "https://api.collegebasketballdata.com"
OUT = RAW / "cbb"
FINISHED_DAYS = 7            # a window is final (cacheable) this long after it ends
CAPS = {"/games": 3000, "/games/teams": 3000, "/lines": 3000, "/games/players": 1000}    # games per call (API docs)
TYPES = {"BIGINT": "Int64", "DOUBLE": "float64", "BOOLEAN": "boolean", "VARCHAR": "string"}
NOT_KEPT = {"teams": ("logos",)}      # columns left out on purpose (everything else not in the contract is named on the console)


def _now() -> datetime:
    return datetime.now(UTC)


def client() -> Client:
    return Client(reserve=cbb().call_reserve, base=BASE, cache=OUT / "cache", log=OUT / "_calls.csv")


def windows(days: int | None) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """[start, end) windows covering the season: calendar months (days None) or runs of `days` from the season's first day."""
    cfg = cbb()
    first, last = pd.Timestamp(cfg.start, tz="UTC"), pd.Timestamp(cfg.end, tz="UTC") + pd.Timedelta(days=1)
    out, a = [], first
    while a < last:
        b = (a + pd.offsets.MonthBegin(1)).normalize() if days is None else a + pd.Timedelta(days=days)
        out.append((a, min(b, last)))
        a = b
    return out


def requests_for(now: datetime | None = None) -> list[tuple[str, str, dict, bool]]:
    """(table, endpoint, params, completed) for the season, windows not yet begun left out."""
    now = now or _now()
    s = cbb().season
    r = [("teams", "/teams", {"season": s}, False), ("rankings", "/rankings", {"season": s}, False),
         ("ratings_adjusted", "/ratings/adjusted", {"season": s}, False), ("ratings_srs", "/ratings/srs", {"season": s}, False),
         ("ratings_elo", "/ratings/elo", {"season": s}, False), ("roster", "/teams/roster", {"season": s}, False),
         ("portal", "/recruiting/portal", {"year": s}, False)]
    for table, ep, days in (("games", "/games", None), ("team_games", "/games/teams", None), ("lines", "/lines", None),
                            ("player_games", "/games/players", 7)):
        for a, b in windows(days):
            if a > now:
                continue
            prm = {"season": s, "startDateRange": a.strftime("%Y-%m-%dT%H:%M:%SZ"), "endDateRange": b.strftime("%Y-%m-%dT%H:%M:%SZ")}
            r.append((table, ep, prm, b < now - timedelta(days=FINISHED_DAYS)))
    return r


# ---- frames ---------------------------------------------------------------------------------------------------
def _flat(raw: list[dict], record_path: str | None = None, meta: list[str] | None = None) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    if record_path:
        raw = [g for g in raw if g.get(record_path)]
        if not raw:
            return pd.DataFrame()
        return pd.json_normalize(raw, record_path=record_path, meta=meta, sep="_", record_prefix="player_" if record_path == "players" else "")
    return pd.json_normalize(raw, sep="_")


GAME_META = ["gameId", "season", "seasonType", "startDate", "teamId", "team", "conference", "opponentId", "opponent", "isHome",
             "gameMinutes", "gamePace"]
LINE_META = ["gameId", "season", "seasonType", "startDate", "homeTeamId", "homeTeam", "awayTeamId", "awayTeam", "homeScore", "awayScore"]
ROSTER_META = ["teamId", "team", "conference", "season"]
FRAMES = {
    "teams": lambda raw: _flat(raw), "games": lambda raw: _flat(raw), "team_games": lambda raw: _flat(raw),
    "player_games": lambda raw: _flat(raw, "players", GAME_META), "lines": lambda raw: _flat(raw, "lines", LINE_META),
    "rankings": lambda raw: _flat(raw), "ratings_adjusted": lambda raw: _flat(raw), "ratings_srs": lambda raw: _flat(raw),
    "ratings_elo": lambda raw: _flat(raw), "roster": lambda raw: _flat(raw, "players", ROSTER_META), "portal": lambda raw: _flat(raw),
}


def contract() -> dict:
    p = ROOT / "contracts" / "cbb_ingest.yml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))["tables"] if p.exists() else {}


def _jsonable(df: pd.DataFrame) -> pd.DataFrame:
    """Lists and dicts left after flattening (period scores, logo lists) become JSON text."""
    for c in df.columns:
        if df[c].dtype == object and df[c].map(lambda v: isinstance(v, (list, dict))).any():
            df[c] = df[c].map(lambda v: json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict)) else v)
    return df


def conform(df: pd.DataFrame, table: str, spec: dict) -> pd.DataFrame:
    """The contract's columns, in its order and types. Extra columns are dropped (named once on the console); a
    missing one is left missing for the checks to report."""
    df = _jsonable(df)
    cols = spec["columns"]
    extra = [c for c in df.columns if c not in cols and c not in NOT_KEPT.get(table, ())]
    if extra:
        print(f"  {table}: not in the contract, not kept: {', '.join(extra[:8])}{' ...' if len(extra) > 8 else ''}")
    out = pd.DataFrame(index=df.index)
    for c, t in cols.items():
        if c not in df.columns:
            continue
        v = df[c]
        if t in ("BIGINT", "DOUBLE"):
            v = pd.to_numeric(v, errors="coerce")
        elif t == "VARCHAR":
            v = v.map(lambda x: None if x is None or (isinstance(x, float) and pd.isna(x)) else str(x))
        out[c] = v.astype(TYPES[t])
    return out.drop_duplicates(spec["key"], keep="last").sort_values(spec["key"], kind="stable").reset_index(drop=True)


def infer(df: pd.DataFrame) -> dict[str, str]:
    """A first contract from a real pull (review by hand: an all-missing column can only be guessed)."""
    out = {}
    for c in _jsonable(df).columns:
        v = df[c].dropna()
        if len(v) and v.map(lambda x: isinstance(x, bool)).all():
            out[c] = "BOOLEAN"
        elif len(v) and pd.api.types.is_numeric_dtype(pd.to_numeric(v, errors="coerce")) and pd.to_numeric(v, errors="coerce").notna().all() \
                and not v.map(lambda x: isinstance(x, str)).any():
            is_id = c in ("id", "season", "week", "year") or c.endswith(("Id", "_id", "Seed"))
            out[c] = "BIGINT" if is_id else "DOUBLE"
        else:
            out[c] = "VARCHAR"
    return out


# ---- the pull ---------------------------------------------------------------------------------------------------
def pull(c: Client, infer_only: bool = False) -> dict[str, pd.DataFrame]:
    s = cbb().season
    raw: dict[str, list] = {}
    for table, ep, prm, done in requests_for():
        data = c.get(ep, prm, done)
        if ep in CAPS:
            games = len({g.get("gameId", g.get("id")) for g in data})
            if games >= CAPS[ep]:
                raise RuntimeError(f"{ep} {prm['startDateRange'][:10]} to {prm['endDateRange'][:10]} returned {games} games, the "
                                   f"API's cap: shorten this table's window in ingest/cbb/pull.py")
        raw.setdefault(table, []).extend(data)
    specs = contract()
    frames = {}
    d1 = {t["id"] for t in raw.get("teams", [])}
    if "roster" in raw:            # the roster endpoint returns every school the source knows; keep Division I
        raw["roster"] = [t for t in raw["roster"] if t.get("teamId") in d1]
    for table, rows in raw.items():
        df = FRAMES[table](rows)
        if infer_only:
            frames[table] = df
            continue
        if df.empty or table not in specs:      # e.g. the portal before its window: no rows, so no contract yet
            print(f"  {table}_{s}: {'no rows yet' if df.empty else 'rows but no contract (add the table to contracts/cbb_ingest.yml)'}")
            continue
        df = conform(df, table, specs[table])
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT / f"{table}_{s}.parquet", index=False)
        frames[table] = df
        print(f"  {table}_{s}: {len(df):,} rows")
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--yes", action="store_true", help="go ahead above the approval line")
    ap.add_argument("--infer", action="store_true", help="print a contract inferred from this pull (writes no table)")
    a = ap.parse_args()
    cfg = cbb()
    c = client()
    need = sum(1 for _, ep, prm, done in requests_for() if not (done and c.cached(ep, prm)))
    info = c.info()                                   # free: it consumes no calls
    c.remaining = info.get("remainingCalls")
    print(f"season {cfg.season}: {need} uncached calls needed; {c.remaining} left this month (shared with football), "
          f"basketball stops at {cfg.call_reserve}")
    if a.estimate:
        return 0
    if need > cfg.approve_over and not a.yes:
        print(f"more than {cfg.approve_over} calls: rerun with --yes")
        return 2
    try:
        frames = pull(c, infer_only=a.infer)
    except BudgetExhausted as e:
        print(f"stopped: {e}")
        return 1
    if a.infer:
        print(yaml.safe_dump({t: {"columns": infer(df)} for t, df in frames.items() if not df.empty}, sort_keys=False, width=200))
    print(f"{c.calls - 1} calls used; {c.remaining} left")
    return 0


if __name__ == "__main__":
    sys.exit(main())
