-- Play-by-play integrity, one row per completed game with plays: how the play-by-play's final score compares with
-- CFBD's official result (the games table, which is the authority), and what the ratings do about it.
--
-- pbp_class (from the two finals; same function as ingest/cfb/integrity.py, tested to agree):
--   match     the play-by-play reaches the official score for both teams
--   swapped   it reaches the official score with home and away exactly exchanged (and the game wasn't a tie)
--   over      a team's running score goes above its official final, which can't happen: a bad score field
--   under     neither team's running score passes its final but at least one stops short: plays or scores missing
--
-- decision (the rule; every non-matching game gets exactly one):
--   repair_teams    swapped, and the scoring plays fit the official result only with offense and defense exchanged:
--                   the team labels are wrong. stg_cfb_plays exchanges offense and defense (and conferences) and
--                   takes home field from the games table.
--   repair_scores   swapped, and the scoring plays fit the official result as labeled: the teams are right and the
--                   two score fields are exchanged. stg_cfb_plays exchanges the score fields (garbage time only).
--   keep_no_score   swapped, but the scoring plays can't tell which (fewer than `cfb_pbp_swap_evidence_points` between
--                   the two readings): teams kept as labeled, running score ignored (no play is garbage time).
--   keep            over or under. Teams, yards and EPA don't depend on the score fields, so the plays stay. In an
--                   `over` game, a play whose running score is above either team's official final has no usable
--                   margin and is never garbage time (stg_cfb_plays).
--   exclude         off by `cfb_pbp_exclude_gap` points or more (default: never; see dbt_project.yml).
--
-- The evidence for a swapped game: points by labeled offense from offensive touchdowns (7 each) and made field
-- goals (3), compared with the official result as recorded (d_keep) and exchanged (d_swap), as the sum of the two
-- teams' absolute differences. Extra points, two-point tries, returns and safeties are left out, so it is only read
-- as a comparison between the two readings, never as a score.
with g as (
    select id as game_id, season, homeTeam as home_team, awayTeam as away_team,
           homePoints as home_points, awayPoints as away_points
    from {{ source('cfbd', 'games') }}
    where completed and homePoints is not null and awayPoints is not null
      and season between {{ var('cfb_first_season') }} and {{ var('cfb_season') }}
),
p as (
    select gameId as game_id, offense, defense, offenseScore, defenseScore, playType
    from {{ source('cfbd', 'plays') }}
    where season between {{ var('cfb_first_season') }} and {{ var('cfb_season') }}
),
running as (
    select game_id, offense as team, offenseScore as score from p
    union all
    select game_id, defense, defenseScore from p
),
final as (select game_id, team, max(score) as score from running group by all),
scoring as (
    select game_id, offense as team,
           sum(case when playType in ('Passing Touchdown', 'Rushing Touchdown') then 7
                    when playType = 'Field Goal Good' then 3 else 0 end) as pts
    from p group by all
),
j as (
    select g.*, h.score as pbp_home, a.score as pbp_away,
           coalesce(sh.pts, 0) as est_home, coalesce(sa.pts, 0) as est_away
    from g
    join final h on h.game_id = g.game_id and h.team = g.home_team
    join final a on a.game_id = g.game_id and a.team = g.away_team
    left join scoring sh on sh.game_id = g.game_id and sh.team = g.home_team
    left join scoring sa on sa.game_id = g.game_id and sa.team = g.away_team
),
c as (
    select *,
        case when pbp_home = home_points and pbp_away = away_points then 'match'
             when pbp_home = away_points and pbp_away = home_points and home_points <> away_points then 'swapped'
             when pbp_home > home_points or pbp_away > away_points then 'over'
             else 'under' end                                                       as pbp_class,
        greatest(abs(pbp_home - home_points), abs(pbp_away - away_points))          as gap,
        abs(est_home - home_points) + abs(est_away - away_points)                   as d_keep,
        abs(est_home - away_points) + abs(est_away - home_points)                   as d_swap
    from j
)
select game_id, season, home_team, away_team, home_points, away_points, pbp_home, pbp_away, pbp_class, gap,
       est_home, est_away, d_keep, d_swap,
       case when pbp_class = 'match' then 'keep'
            when gap >= {{ var('cfb_pbp_exclude_gap') }} then 'exclude'
            when pbp_class = 'swapped' and d_keep - d_swap >= {{ var('cfb_pbp_swap_evidence_points') }} then 'repair_teams'
            when pbp_class = 'swapped' and d_swap - d_keep >= {{ var('cfb_pbp_swap_evidence_points') }} then 'repair_scores'
            when pbp_class = 'swapped' then 'keep_no_score'
            else 'keep' end                                                         as decision
from c
