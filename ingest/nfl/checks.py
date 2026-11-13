"""NFL ingest checks: play-by-play per season, schedules, teams, team stats. Exit status 1 if any
check fails.

    python -m ingest.nfl.checks
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

from ingest.common.checks import Checks
from sparvi.config import RAW, nfl

STALE_DAYS = 3          # a game this many days old must have a final score in the schedule


def run(today: date | None = None, n_teams: int = 32) -> bool:
    today = today or date.today()
    cfg = nfl()
    d = RAW / "nfl"
    c = Checks("nfl")
    pbp_ok = True
    for s in cfg.seasons:
        pbp_ok &= c.table(f"pbp_{s}", d / f"pbp_{s}.parquet", "pbp")
        c.table(f"team_stats_{s}", d / f"team_stats_{s}.parquet", "team_stats")

    through = {}
    if c.table("schedules", d / "schedules.parquet", "schedules") and pbp_ok:
        pbp = f"read_parquet('{(d / 'pbp_*.parquet').as_posix()}')"
        sch = f"read_parquet('{(d / 'schedules.parquet').as_posix()}')"
        # Every completed game in the window has plays, and every game with plays is on the schedule.
        n_missing, example = c.sql(f"""
            select count(*), min(game_id) from {sch} s
            where s.home_score is not null and s.season between {cfg.first_season} and {cfg.season}
              and not exists (select 1 from {pbp} p where p.game_id = s.game_id)""")
        c.add("every completed game has play-by-play", n_missing == 0,
              f"{n_missing} games without plays, e.g. {example}" if n_missing else "")
        n_extra, example = c.sql(f"""
            select count(distinct game_id), min(game_id) from {pbp} p
            where not exists (select 1 from {sch} s where s.game_id = p.game_id)""")
        c.add("every game with plays is on the schedule", n_extra == 0,
              f"{n_extra} games, e.g. {example}" if n_extra else "")
        # Freshness: games played more than a few days ago have final scores.
        cutoff = today - timedelta(days=STALE_DAYS)
        n_stale, example = c.sql(f"""
            select count(*), min(game_id) from {sch}
            where season = {cfg.season} and home_score is null and gameday::date <= '{cutoff}'""")
        c.add("schedule results are current", n_stale == 0,
              f"{n_stale} games played by {cutoff} without a score, e.g. {example}" if n_stale else "")
        week, last = c.sql(f"select max(week), max(game_date) from {pbp} where season = {cfg.season}")
        through = {"season": cfg.season, "week": week, "last_game": last}
        # Each regular-season week has every team that played (catches a partly loaded week).
        n_short = c.sql(f"""
            with g as (select season, week, count(distinct posteam) n from {pbp}
                       where season_type = 'REG' and posteam is not null group by 1, 2),
                 want as (select season, week, 2 * count(*) n from {sch}
                          where game_type = 'REG' and home_score is not null group by 1, 2)
            select count(*) from want join g using (season, week) where g.n <> want.n""")[0]
        c.add("every team that played has plays in its week", n_short == 0,
              f"{n_short} season-weeks short" if n_short else "")

    if c.table("teams", d / "teams.parquet", "teams"):
        c.add("teams: one row per franchise", c.rows["teams"] == n_teams, f"{c.rows['teams']} (want {n_teams})")
    return c.finish(through)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
