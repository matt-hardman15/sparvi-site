"""College football ingest checks: plays, games, teams, rankings, and lines per season against
contracts/cfb_ingest.yml, plus coverage and the score check agreed for F2 (Nov 13, 2026):

- every completed game involving an FBS team has an official result (both scores);
- completed games have play-by-play (at most 2% missing per season);
- the play-by-play's final score agrees with the result in at least 90% of each season's games
  (CFBD's play-by-play has gaps and bad score fields in ~3% of games); games off by 9+ points are
  listed in the status file;
- in season, games played more than a few days ago have a result.

Player level (CFB-PLAYERS): every player file against the contract, plus: season stats link to the
roster, usage and PPA describe the same players, heights are plausible, and every finished portal year
has entries.

    python -m ingest.cfb.checks
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta

import ingest.common.checks as common
from ingest.common.checks import Checks
from sparvi.config import RAW, cfb
from sparvi.io import write_json

MIN_SCORE_AGREEMENT = 0.90
MAX_MISSING_PLAYS = 0.02
BIG_GAP = 9
STALE_DAYS = 3
MIN_STATS_IN_ROSTER = 0.95     # player stats whose player id is on that season's roster
MIN_PORTAL_ENTRIES = 1000      # a finished portal year (2021-25 had ~2,500-4,500)


def player_checks(c: Checks, d, cfg) -> None:
    from ingest.cfb import players
    ss = players.seasons(cfg)
    for s in ss["season"]:
        ok = all([c.table(f"{t}_{s}", d / f"{f}_{s}.parquet", t) for t, f in
                  (("roster", "roster"), ("player_stats", "stats"), ("player_usage", "usage"), ("player_ppa", "ppa"), ("player_wepa", "wepa"))])
        if not ok:
            continue
        r = lambda f, s=s: f"read_parquet('{(d / f'{f}_{s}.parquet').as_posix()}')"  # noqa: E731
        n, hit = c.sql(f"select count(distinct playerId), count(distinct playerId) filter (where playerId in (select id from {r('roster')})) from {r('stats')}")
        share = hit / n if n else 1.0
        c.add(f"player_stats_{s}: players on the roster", share >= MIN_STATS_IN_ROSTER,
              f"{hit:,} of {n:,} players ({share:.1%}; bar {MIN_STATS_IN_ROSTER:.0%})")
        diff = c.sql(f"select count(*) from ((select id, team from {r('usage')}) except (select id, team from {r('ppa')}))")[0]
        c.add(f"player_usage_{s}: same players as PPA", diff == 0, f"{diff} usage rows without PPA" if diff else "")
        bad, known = c.sql(f"select count(*) filter (where height not between 60 and 86), count(height) from {r('roster')}")
        c.add(f"roster_{s}: heights plausible", bad <= 0.005 * known, f"{bad} of {known:,} outside 5'0\"-7'2\"")
    for s in ss["portal"]:
        if c.table(f"portal_{s}", d / f"portal_{s}.parquet", "portal") and players.portal_done(s):
            n = c.rows[f"portal_{s}"]
            c.add(f"portal_{s}: entries", n >= MIN_PORTAL_ENTRIES, f"{n:,} entries")
    for s in ss["recruits"]:
        if players.recruits_done(s):
            c.table(f"recruits_{s}", d / f"recruits_{s}.parquet", "recruits")


def run(today: date | None = None) -> bool:
    today = today or date.today()
    cfg = cfb()
    d = RAW / "cfb"
    c = Checks("cfb")
    ok_seasons = []
    for s in cfg.seasons:
        files_ok = all([c.table(f"{t}_{s}", d / f"{t}_{s}.parquet", t) for t in ("games", "plays", "teams", "rankings", "lines")])
        if files_ok:
            ok_seasons.append(s)
        if (d / f"teams_{s}.parquet").exists():
            n = c.rows.get(f"teams_{s}", 0)
            c.add(f"teams_{s}: FBS team count", 120 <= n <= 150, f"{n} teams")

    big_gaps, through = [], {}
    for s in ok_seasons:
        g = f"read_parquet('{(d / f'games_{s}.parquet').as_posix()}')"
        p = f"read_parquet('{(d / f'plays_{s}.parquet').as_posix()}')"
        n_done, no_result = c.sql(f"select count(*) filter (where completed), count(*) filter (where completed and (homePoints is null or awayPoints is null)) from {g}")
        c.add(f"games_{s}: completed games have results", no_result == 0, f"{no_result} of {n_done} completed games without a score" if no_result else f"{n_done} completed games")
        missing = c.sql(f"select count(*) from {g} where completed and id not in (select distinct gameId from {p})")[0]
        share = missing / n_done if n_done else 0
        c.add(f"plays_{s}: completed games have play-by-play", share <= MAX_MISSING_PLAYS,
              f"{missing} of {n_done} games without plays ({share:.1%}; limit {MAX_MISSING_PLAYS:.0%})")
        agree, total, gaps = c.sql(f"""
            with f as (select game_id, team, max(score) score from (
                         select gameId game_id, offense team, offenseScore score from {p}
                         union all select gameId, defense, defenseScore from {p}) group by all),
                 r as (select g.id, g.homeTeam, g.awayTeam, g.homePoints hp, g.awayPoints ap, h.score ph, a.score pa
                       from {g} g join f h on h.game_id = g.id and h.team = g.homeTeam
                                  join f a on a.game_id = g.id and a.team = g.awayTeam where g.completed)
            select count(*) filter (where ph = hp and pa = ap), count(*),
                   list(struct_pack(id := id, home := homeTeam, away := awayTeam, result := hp::int || '-' || ap::int,
                                    pbp := ph::int || '-' || pa::int) order by id)
                     filter (where greatest(abs(ph - hp), abs(pa - ap)) >= {BIG_GAP})
            from r""")
        rate = agree / total if total else 1.0
        c.add(f"plays_{s}: final scores agree with results", rate >= MIN_SCORE_AGREEMENT,
              f"{agree:,} of {total:,} games ({rate:.1%}; bar {MIN_SCORE_AGREEMENT:.0%}); {len(gaps or [])} off by {BIG_GAP}+ points")
        big_gaps += [{"season": s, **x} for x in (gaps or [])]
        if s == cfg.season:
            cutoff = today - timedelta(days=STALE_DAYS)
            stale = c.sql(f"select count(*) from {g} where not completed and startDate::timestamp::date <= '{cutoff}'")[0]
            c.add(f"games_{s}: results are current", stale == 0, f"{stale} games played by {cutoff} without a result" if stale else "")
            wk, last = c.sql(f"select max(week) filter (where completed and seasonType = 'regular'), max(startDate::timestamp::date) filter (where completed) from {g}")
            through = {"season": s, "week": wk, "last_game": str(last) if last else None}
    player_checks(c, d / "players", cfg)
    ok = c.finish(through)
    # Listed for /status/: games whose play-by-play is far from the official score (kept in the ratings).
    st = json.loads((common.STATUS / "cfb.json").read_text(encoding="utf-8"))
    st["score_gaps"] = big_gaps
    write_json(common.STATUS / "cfb.json", st, indent=1)
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
