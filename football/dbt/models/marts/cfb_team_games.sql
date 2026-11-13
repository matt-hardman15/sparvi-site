-- One row per team x game (regular season and bowls): result, score, and that game's offense and
-- defense rates (both garbage-time variants), for the game logs.
with g0 as (
    select id as game_id, season, seasonType as season_type, week, neutralSite as neutral_site,
           homeTeam as home_team, awayTeam as away_team, homePoints as home_points, awayPoints as away_points
    from {{ source('cfbd', 'games') }}
    where season between {{ var('cfb_first_season') }} and {{ var('cfb_season') }}
),
wk as (select distinct season, game_id, week from {{ ref('stg_cfb_plays') }}),
sched as (
    select g.season, w.week, g.game_id, g.home_team as team, g.away_team as opponent, not g.neutral_site as is_home,
           g.home_points as points_for, g.away_points as points_against
    from g0 g join wk w using (season, game_id)
    union all
    select g.season, w.week, g.game_id, g.away_team, g.home_team, false, g.away_points, g.home_points
    from g0 g join wk w using (season, game_id)
),
rates as (
    select season, week, game_id, team, side, gt, plays, {{ team_rates() }} from {{ ref('int_cfb_team_games') }}
)
select s.season, s.week, s.game_id, s.team, s.opponent, s.is_home, s.points_for, s.points_against,
       case when s.points_for > s.points_against then 'W' when s.points_for < s.points_against then 'L'
            when s.points_for is not null then 'T' end as outcome,
       r.side, r.gt, r.plays, r.epa_play, r.success_rate, r.pass_epa_play, r.run_epa_play
from sched s
join rates r on r.game_id = s.game_id and r.team = s.team
