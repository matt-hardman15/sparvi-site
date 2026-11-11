-- Starter / reliever role per pitcher, level, and season (season-to-date for the current season).
--   SP    start_share >= role_sp_min_share and starts >= role_sp_min_starts
--   RP    start_share <= role_rp_max_share
--   Swing everything in between (openers and spot starters usually land in RP or Swing)
-- Thresholds are dbt vars so the glossary can quote them. role_by_pitches (share of a pitcher's
-- pitches thrown in his starts) is kept as a tie-break for later; it isn't shown on the site yet.
with a as (select * from {{ ref('int_pitcher_appearances') }})

select
    level,
    season,
    pitcher_id,
    mode(team)                                                  as team,
    count(*)                                                    as games,
    count(*) filter (where started)                             as starts,
    count(*) filter (where not started)                         as relief_apps,
    count(*) filter (where started) / count(*)                  as start_share,
    sum(pitches)                                                as pitches,
    sum(batters_faced)                                          as batters_faced,
    sum(outs_recorded)                                          as outs_recorded,
    coalesce(sum(pitches) filter (where started), 0) / sum(pitches) as role_by_pitches,
    {{ role_case('count(*) filter (where started)', 'count(*)') }} as role,
    max(game_date)                                              as last_game
from a
group by level, season, pitcher_id
