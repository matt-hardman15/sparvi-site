"""Baseball ingest checks: Statcast (every level x season in the config window), batter heights,
Triple-A rosters and affiliates, and the free-agent list when one has been pulled. Exit status 1 if any check fails.

    python -m ingest.baseball.checks
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

from ingest.common.checks import Checks
from sparvi.config import RAW, baseball

FRESH_DAYS = 5          # in season, the newest game must be at most this old (covers the All-Star break)


def run(today: date | None = None, n_orgs: int = 30) -> bool:
    today = today or date.today()
    cfg = baseball()
    c = Checks("baseball")
    through = {}
    for level, seasons in (("aaa", cfg.aaa_seasons), ("mlb", cfg.mlb_seasons)):
        for s in seasons:
            name = f"statcast_{level}_{s}"
            path = RAW / f"{name}.parquet"
            if not c.table(name, path, "statcast"):
                continue
            rel = f"read_parquet('{path.as_posix()}')"
            lo, hi = cfg.season_windows[s]
            first, last, bad_year, outside = c.sql(
                f"select min(game_date)::date, max(game_date)::date, count(*) filter (where game_year <> {s}),"
                f" count(*) filter (where game_date::date not between '{lo}' and '{hi}') from {rel}")
            c.add(f"{name}: dates in the season window", bad_year == 0 and outside == 0,
                  f"{first} to {last}" + (f"; {bad_year:,} rows from another season" if bad_year else "")
                  + (f"; {outside:,} rows outside {lo}..{hi}" if outside else ""))
            if s == cfg.season:
                through[level] = str(last)
                start, end = (date.fromisoformat(x) for x in (lo, hi))
                in_season = start + timedelta(days=FRESH_DAYS) <= today <= end + timedelta(days=FRESH_DAYS)
                c.add(f"{name}: fresh", not in_season or (today - last).days <= FRESH_DAYS,
                      f"newest game {last}" + ("" if in_season else " (off-season)"))

    c.table("batter_heights", RAW / "batter_heights.parquet", "batter_heights")
    c.table(f"milb_players_{cfg.season}", RAW / f"milb_players_{cfg.season}.parquet", "milb_players")
    fa = RAW / f"free_agents_{cfg.season}.parquet"      # pulled by hand each off-season (pull_free_agents.py)
    if fa.exists() and c.table(f"free_agents_{cfg.season}", fa, "free_agents"):
        n_p = c.sql(f"select count(*) filter (where position_type = 'Pitcher') from read_parquet('{fa.as_posix()}')")[0]
        c.add(f"free_agents_{cfg.season}: a full list", c.rows[f"free_agents_{cfg.season}"] >= 100 and n_p >= 40,
              f"{c.rows[f'free_agents_{cfg.season}']} players, {n_p} pitchers")
    for s in cfg.aaa_seasons:
        name = f"aaa_affiliates_{s}"
        if c.table(name, RAW / f"{name}.parquet", "aaa_affiliates"):
            n = c.rows[name]
            c.add(f"{name}: one affiliate per org", n == n_orgs, f"{n} clubs (want {n_orgs})")
    return c.finish(through)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
