-- One row per team x regular-season game: result, score, and that game's offense and defense rates
-- (both garbage-time variants), for the team pages' game log and weekly lines.
with sched as (
    select season, week, game_id, home_team as team, away_team as opponent, true as is_home,
           home_score as points_for, away_score as points_against
    from {{ source('nflverse', 'schedules') }} where game_type = 'REG'
    union all
    select season, week, game_id, away_team, home_team, false, away_score, home_score
    from {{ source('nflverse', 'schedules') }} where game_type = 'REG'
),
g as (select * from {{ ref('int_nfl_team_games') }}),
rates as (
    select season, week, game_id, team, side, gt, plays, {{ team_rates() }} from g
)
select s.season, s.week, s.game_id, s.team, s.opponent, s.is_home, s.points_for, s.points_against,
       case when s.points_for > s.points_against then 'W' when s.points_for < s.points_against then 'L'
            when s.points_for is not null then 'T' end as outcome,
       r.side, r.gt, r.plays, r.epa_play, r.success_rate, r.pass_epa_play, r.run_epa_play
from sched s
join rates r on r.game_id = s.game_id and r.team = s.team
where s.season between {{ var('nfl_first_season') }} and {{ var('nfl_season') }}
