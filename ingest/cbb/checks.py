"""College basketball ingest checks (CBB-0a): each table of the current season against contracts/cbb_ingest.yml, plus
coverage and freshness. They run after every pull and before the snapshot; a failure means no snapshot is taken.

- every final game has both scores, and its team box scores add up to them;
- final games have a team box score for both sides and a player box score (measured two days after the game: the
  source fills them in overnight);
- the source's three ratings each cover the Division I teams, and rosters cover them;
- in season, a game that started days ago is not still "scheduled", and the newest final game is recent.

There are no basketball pages yet, so the status file is kept with the raw data (data/raw/cbb/_status/cbb.json), not
in site/data/status/. It holds data-derived values only.

    python -m ingest.cbb.checks
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

import ingest.common.checks as common
from sparvi.config import RAW, cbb

OUT = RAW / "cbb"
TABLES = ("teams", "games", "team_games", "player_games", "lines", "rankings", "ratings_adjusted", "ratings_srs", "ratings_elo", "roster")
SETTLE_DAYS = 2              # box scores are expected this long after a game
MIN_TEAM_BOX, MIN_PLAYER_BOX = 0.98, 0.97     # Nov 20 2026: 99.8% and 99.7%
MIN_POINTS_AGREE = 0.99                       # team box points equal the game's score (100% on Nov 20 2026)
MIN_TEAMS_RATED, MIN_TEAMS_ROSTERED, MIN_ROSTER = 0.95, 0.95, 8
MAX_STALE, STALE_DAYS, FRESH_DAYS = 0.01, 3, 4


def run(today: date | None = None) -> bool:
    cfg = cbb()
    s, today = cfg.season, today or date.today()
    common.STATUS = OUT / "_status"
    c = common.Checks("cbb")
    ok = {t: c.table(f"{t}_{s}", OUT / f"{t}_{s}.parquet", t) for t in TABLES}
    r = lambda t: f"read_parquet('{(OUT / f'{t}_{s}.parquet').as_posix()}')"  # noqa: E731
    through = {}
    if ok["games"]:
        n, bad = c.sql(f"select count(*), count(*) filter (where homePoints is null or awayPoints is null) from {r('games')} where status = 'final'")
        c.add("games: every final game has both scores", bad == 0, f"{bad} of {n:,} without a score" if bad else f"{n:,} final games")
        last = c.sql(f"select max(cast(startDate as timestamp)) from {r('games')} where status = 'final'")[0]
        through = {"last_final_game": str(last.date()) if last else None}
        cut = f"timestamp '{today - timedelta(days=STALE_DAYS)}'"
        stale, played = c.sql(f"select count(*) filter (where status in ('scheduled', 'in_progress')), count(*) from {r('games')} "
                              f"where cast(startDate as timestamp) < {cut}")
        c.add("games: none left unplayed days after its date", stale <= MAX_STALE * max(played, 1),
              f"{stale} of {played:,} still scheduled or in progress more than {STALE_DAYS} days after tip-off (limit {MAX_STALE:.0%})")
        in_season = date.fromisoformat(cfg.start) + timedelta(days=FRESH_DAYS) <= today <= date.fromisoformat(cfg.end)
        if in_season:
            fresh = last is not None and last.date() >= today - timedelta(days=FRESH_DAYS)
            c.add("games: the newest final game is recent", fresh, f"last final game {through['last_final_game']}")
    if ok["games"] and ok["team_games"] and ok["player_games"]:
        settled = f"status = 'final' and cast(startDate as timestamp) < timestamp '{today - timedelta(days=SETTLE_DAYS)}'"
        n, both, players = c.sql(
            f"with g as (select id from {r('games')} where {settled}), t as (select gameId, count(*) n from {r('team_games')} group by 1), "
            f"p as (select distinct gameId from {r('player_games')}) "
            f"select count(*), count(*) filter (where t.n = 2), count(p.gameId) from g left join t on g.id = t.gameId left join p on g.id = p.gameId")
        if n:
            c.add("team_games: both sides' box scores for final games", both >= MIN_TEAM_BOX * n, f"{both:,} of {n:,} ({both / n:.1%}; bar {MIN_TEAM_BOX:.0%})")
            c.add("player_games: a player box score for final games", players >= MIN_PLAYER_BOX * n,
                  f"{players:,} of {n:,} ({players / n:.1%}; bar {MIN_PLAYER_BOX:.0%})")
        n, agree = c.sql(f"select count(*), count(*) filter (where tg.teamStats_points_total = case when tg.teamId = g.homeTeamId then g.homePoints "
                         f"else g.awayPoints end) from {r('team_games')} tg join {r('games')} g on g.id = tg.gameId where g.status = 'final'")
        if n:
            c.add("team_games: points equal the game's score", agree >= MIN_POINTS_AGREE * n, f"{agree:,} of {n:,} ({agree / n:.1%}; bar {MIN_POINTS_AGREE:.0%})")
    if ok["teams"]:
        teams = c.rows[f"teams_{s}"]
        for t in ("ratings_adjusted", "ratings_srs", "ratings_elo"):
            if ok[t]:
                n = c.sql(f"select count(*) from {r(t)} where teamId in (select id from {r('teams')})")[0]
                c.add(f"{t}: covers the Division I teams", n >= MIN_TEAMS_RATED * teams, f"{n} of {teams} teams")
        if ok["roster"]:
            n = c.sql(f"select count(*) from (select teamId from {r('roster')} group by 1 having count(*) >= {MIN_ROSTER})")[0]
            c.add("roster: teams with a roster", n >= MIN_TEAMS_ROSTERED * teams, f"{n} of {teams} teams list {MIN_ROSTER}+ players")
    return c.finish(through)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
