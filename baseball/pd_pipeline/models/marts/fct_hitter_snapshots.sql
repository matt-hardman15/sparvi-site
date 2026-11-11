-- Hitter x snapshot date x window ('std', 'last30'), both levels, current season. Same metrics and
-- percentile directions as fct_hitter_report; the pool at each snapshot is hitters with
-- PA >= the prorated minimum (macro qualifier), against their own level.
with pitches as (
    select * from {{ ref('stg_pitches') }} where season = {{ var('season') }} and game_date <= {{ as_of_by_level() }}
),
w as ({{ snapshot_windows('pitches') }}),

base as (
    select r.*,
           {{ qualifier(var('min_pa_qualified'),
                        "case r.window_name when 'std' then d.team_games else d.team_games_last end", 'r.level') }} as qual_min
    from (
        select level, snapshot_date, window_name, batter_id,
               mode(batting_team) as team, count(distinct game_pk) as games, {{ plate_discipline_rates() }}
        from w group by all
    ) r
    join {{ ref('int_snapshot_dates') }} d using (level, snapshot_date)
),

pctl as (
    select s.level, s.snapshot_date, s.window_name, s.batter_id, {{ pool_pctls(hitter_pctl_metrics()) }}
    from base s
    join base q on q.level = s.level and q.snapshot_date = s.snapshot_date and q.window_name = s.window_name
        and q.pa >= q.qual_min
    group by s.level, s.snapshot_date, s.window_name, s.batter_id
)

select b.level, {{ var('season') }} as season, b.snapshot_date, b.window_name, b.batter_id, b.team, b.games, b.pitches, b.pa,
       b.pa >= b.qual_min as qualified, b.qual_min,
       b.chase_pct, b.zone_swing_pct, b.zone_contact_pct, b.whiff_pct, b.k_pct, b.bb_pct,
       b.hard_hit_pct, b.avg_ev, b.ev90, b.xwoba,
       {% for m, _ in hitter_pctl_metrics() %}p.{{ m }}_pctl{{ ',' if not loop.last }}{% endfor %}
from base b
left join pctl p using (level, snapshot_date, window_name, batter_id)
