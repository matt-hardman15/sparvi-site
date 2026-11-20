{{ config(materialized="table") }}
-- One row per college efficiency play (passes incl. sacks, runs incl. fumbles), from the offense's
-- side, in games involving an FBS team. Regular season plus the postseason, which is numbered as one
-- more week after the last regular-season week ("bowls"). Every CFB rate on the site comes from these.
--
-- Team attribution follows the games table (int_cfb_game_integrity has the rule): a game whose play-by-play has
-- offense and defense exchanged is put right, a game with the score fields exchanged gets them back, a running
-- score above a team's official final is ignored, and a play naming a team that isn't one of the game's two teams
-- is dropped (the build fails if that is more than a sliver of plays: schema.yml).
with g as (
    select id as game_id, season, seasonType as season_type, week, neutralSite as neutral_site,
           homeTeam as home_team, awayTeam as away_team, homePoints as home_points, awayPoints as away_points
    from {{ source('cfbd', 'games') }}
    where season between {{ var('cfb_first_season') }} and {{ var('cfb_season') }}
),
last_regular as (
    select season, max(week) as w from g where season_type = 'regular' group by 1
),
raw as (
    select p.*, coalesce(i.decision, 'keep') as decision
    from {{ source('cfbd', 'plays') }} p
    left join {{ ref('int_cfb_game_integrity') }} i on i.game_id = p.gameId
    where p.ppa is not null and p.down between 1 and 4 and p.distance > 0
      and (p.playType in {{ cfb_pass_types() }} or p.playType in {{ cfb_run_types() }})
),
p as (
    -- team labels and score fields as the decision says; everything else as the feed has it
    select * replace (
        case when decision = 'repair_teams' then defense else offense end                     as offense,
        case when decision = 'repair_teams' then offense else defense end                     as defense,
        case when decision = 'repair_teams' then defenseConference else offenseConference end as offenseConference,
        case when decision = 'repair_teams' then offenseConference else defenseConference end as defenseConference,
        case when decision = 'repair_scores' then defenseScore else offenseScore end          as offenseScore,
        case when decision = 'repair_scores' then offenseScore else defenseScore end          as defenseScore)
    from raw
    where decision <> 'exclude'
),
m as (
    select p.*, g.season as g_season, g.season_type, g.week as g_week, g.neutral_site, g.home_team, g.away_team,
           case when p.decision = 'keep_no_score' then null
                -- a running score above the team's official final can't be right: no usable margin
                when p.offenseScore > case when p.offense = g.home_team then g.home_points else g.away_points end
                  or p.defenseScore > case when p.defense = g.home_team then g.home_points else g.away_points end then null
                else p.offenseScore - p.defenseScore end as margin
    from p
    join g on g.game_id = p.gameId
    -- the play's two teams are the game's two teams
    where (p.offense = g.home_team and p.defense = g.away_team) or (p.offense = g.away_team and p.defense = g.home_team)
)
select
    m.g_season                                                as season,
    case when m.season_type = 'postseason' then l.w + 1 else m.g_week end as week,
    m.season_type,
    m.gameId                                                  as game_id,
    m.id                                                      as play_id,
    m.offense,
    m.defense,
    m.offenseConference                                       as offense_conference,
    m.defenseConference                                       as defense_conference,
    not m.neutral_site and m.offense = m.home_team            as is_home,
    m.period                                                  as qtr,
    m.down,
    m.distance,
    m.yardsGained                                             as yards_gained,
    m.ppa                                                     as epa,
    m.playType in {{ cfb_pass_types() }}                      as is_pass,
    {{ cfb_success('m.down', 'm.distance', 'm.yardsGained') }} as is_success,
    false                                                     as is_neutral_early,   -- no win probability in the CFB feed
    cast(null as double)                                      as xpass,
    case when m.playType in {{ cfb_pass_types() }} then m.yardsGained >= {{ var('explosive_pass_yds') }}
         else m.yardsGained >= {{ var('explosive_run_yds') }} end as is_explosive,
    m.playType in ('Pass Interception Return', 'Interception Return Touchdown', 'Interception',
                   'Fumble Recovery (Opponent)', 'Fumble Return Touchdown') as is_turnover,
    {{ cfb_garbage_time('m.period', 'm.margin') }}            as is_garbage
from m
join last_regular l on l.season = m.g_season
