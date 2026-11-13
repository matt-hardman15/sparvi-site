{{ config(materialized="table") }}
-- One row per college efficiency play (passes incl. sacks, runs incl. fumbles), from the offense's
-- side, in games involving an FBS team. Regular season plus the postseason, which is numbered as one
-- more week after the last regular-season week ("bowls"). Every CFB rate on the site comes from these.
with g as (
    select id as game_id, season, seasonType as season_type, week, neutralSite as neutral_site,
           homeTeam as home_team, awayTeam as away_team
    from {{ source('cfbd', 'games') }}
    where season between {{ var('cfb_first_season') }} and {{ var('cfb_season') }}
),
last_regular as (
    select season, max(week) as w from g where season_type = 'regular' group by 1
),
p as (
    select p.*, p.offenseScore - p.defenseScore as margin
    from {{ source('cfbd', 'plays') }} p
    where p.ppa is not null and p.down between 1 and 4 and p.distance > 0
      and (p.playType in {{ cfb_pass_types() }} or p.playType in {{ cfb_run_types() }})
)
select
    g.season,
    case when g.season_type = 'postseason' then l.w + 1 else g.week end    as week,
    g.season_type,
    p.gameId                                                  as game_id,
    p.id                                                      as play_id,
    p.offense,
    p.defense,
    p.offenseConference                                       as offense_conference,
    p.defenseConference                                       as defense_conference,
    not g.neutral_site and p.offense = g.home_team            as is_home,
    p.period                                                  as qtr,
    p.down,
    p.distance,
    p.yardsGained                                             as yards_gained,
    p.ppa                                                     as epa,
    p.playType in {{ cfb_pass_types() }}                      as is_pass,
    {{ cfb_success('p.down', 'p.distance', 'p.yardsGained') }} as is_success,
    false                                                     as is_neutral_early,   -- no win probability in the CFB feed
    cast(null as double)                                      as xpass,
    case when p.playType in {{ cfb_pass_types() }} then p.yardsGained >= {{ var('explosive_pass_yds') }}
         else p.yardsGained >= {{ var('explosive_run_yds') }} end as is_explosive,
    p.playType in ('Pass Interception Return', 'Interception Return Touchdown', 'Interception',
                   'Fumble Recovery (Opponent)', 'Fumble Return Touchdown') as is_turnover,
    {{ cfb_garbage_time('p.period', 'p.margin') }}            as is_garbage
from p
join g on g.game_id = p.gameId
join last_regular l on l.season = g.season
