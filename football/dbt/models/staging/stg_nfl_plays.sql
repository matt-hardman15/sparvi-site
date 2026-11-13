{{ config(materialized="table") }}
-- A table, not a view: Python reads it (nfl/team_efficiency/adjusted.py) from any working directory.
-- One row per regular-season efficiency play (see nfl_efficiency_play()), from the offense's side.
-- Every NFL rate on the site is built from these rows.
select
    season,
    week,
    game_id,
    play_id,
    posteam                                                  as offense,
    defteam                                                  as defense,
    posteam = home_team                                      as is_home,
    qtr,
    down,
    wp,
    epa,
    pass = 1                                                 as is_pass,
    epa > 0                                                  as is_success,
    down in (1, 2) and wp between {{ var('nfl_neutral_wp_lo') }} and {{ var('nfl_neutral_wp_hi') }} as is_neutral_early,
    xpass,
    case when pass = 1 then yards_gained >= {{ var('explosive_pass_yds') }}
         else yards_gained >= {{ var('explosive_run_yds') }} end as is_explosive,
    coalesce(interception, 0) = 1 or coalesce(fumble_lost, 0) = 1 as is_turnover,
    {{ nfl_garbage_time() }}                                 as is_garbage
from {{ source('nflverse', 'pbp') }}
where season_type = 'REG'
  and season between {{ var('nfl_first_season') }} and {{ var('nfl_season') }}
  and posteam is not null and defteam is not null
  and {{ nfl_efficiency_play() }}
