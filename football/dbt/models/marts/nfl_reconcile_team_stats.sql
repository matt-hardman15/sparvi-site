-- Our play-by-play against nflverse's published team stats, per team x season (regular season).
-- nflverse sums qb_epa on pass plays and spikes, and epa on runs and kneel-downs, two-point tries
-- included; recomputed the same way here, the totals must agree to rounding (tested). This checks the
-- pull and the team mapping; the site's efficiency rates use the narrower play set in stg_nfl_plays.
with p as (
    select season, posteam as team,
           sum(qb_epa) filter (where play_type in ('pass', 'qb_spike')) as passing_epa,
           count(*) filter (where play_type in ('run', 'qb_kneel') and rush_attempt = 1) as carries,
           sum(epa) filter (where play_type in ('run', 'qb_kneel')) as rushing_epa
    from {{ source('nflverse', 'pbp') }}
    where season_type = 'REG' and posteam is not null
      and season between {{ var('nfl_first_season') }} and {{ var('nfl_season') }}
    group by all
)
select p.season, p.team,
       p.passing_epa, t.passing_epa as nflverse_passing_epa,
       p.rushing_epa, t.rushing_epa as nflverse_rushing_epa,
       abs(p.passing_epa - t.passing_epa) as passing_diff,
       abs(p.rushing_epa - t.rushing_epa) as rushing_diff
from p
join {{ source('nflverse', 'team_stats') }} t on t.season = p.season and t.team = p.team
