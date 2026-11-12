-- Pitcher x snapshot date x window ('std' season to date, 'last30'), both levels, current season.
-- The final 'std' snapshot reproduces the season-level marts exactly (tests/test_snapshots.py).
--   role        as of the snapshot, from season-to-date starts (same rule as dim_pitcher_role)
--   fb_velo     primary fastball as of the snapshot (most-thrown FF/SI season to date, ties to FF),
--               measured within the window, same definition as fct_pitcher_report
--   stuff_plus  on the frozen prior-season scale (scale_reference), so 100 means the same all season;
--   loc_plus, pitching_plus  the location and combined models, same frozen scale (kinds 'loc', 'pitching')
--   rv_plus     actual run value on the same kind of frozen scale
--   *_pctl      against that snapshot's qualified pool: BF >= the prorated minimum (macro qualifier)
with pitches as (
    select * from {{ ref('stg_pitches') }} where season = {{ var('season') }} and game_date <= {{ as_of_by_level() }}
),
w as ({{ snapshot_windows('pitches') }}),

rates as (
    select level, snapshot_date, window_name, pitcher_id,
           mode(pitching_team) as team, count(distinct game_pk) as games, {{ plate_discipline_rates() }}
    from w group by all
),

roles as (
    select s.level, s.snapshot_date, a.pitcher_id,
           count(*) as games_std, count(*) filter (where a.started) as starts,
           {{ role_case('count(*) filter (where a.started)', 'count(*)') }} as role
    from {{ ref('int_snapshot_dates') }} s
    join {{ ref('int_pitcher_appearances') }} a
        on a.level = s.level and a.season = {{ var('season') }} and a.game_date <= s.snapshot_date
    group by all
),

fb_types as (
    select level, snapshot_date, window_name, pitcher_id, pitch_type, count(*) as n, avg(release_speed) as velo
    from w where pitch_type in ('FF', 'SI') group by all
),
primary_fb as (
    select level, snapshot_date, pitcher_id, pitch_type
    from fb_types where window_name = 'std'
    qualify row_number() over (partition by level, snapshot_date, pitcher_id order by n desc, pitch_type) = 1
),
fb as (
    select t.level, t.snapshot_date, t.window_name, t.pitcher_id, t.velo as fb_velo
    from fb_types t join primary_fb p using (level, snapshot_date, pitcher_id, pitch_type)
),

preds as (select * from {{ ref('int_pitch_predictions') }}),
pw as ({{ snapshot_windows('preds') }}),
ref as (
    select {% for k in ['stuff', 'loc', 'pitching', 'rv'] %}max(mean) filter (where kind = '{{ k }}') as m_{{ k }},
           max(sd) filter (where kind = '{{ k }}') as sd_{{ k }}{{ ',' if not loop.last }}
           {% endfor %}
    from {{ source('pitch_quality', 'scale_reference') }}
),
stuff as (
    select level, snapshot_date, window_name, pitcher_id, count(*) as graded_pitches,
           {% for k, col in [('stuff', 'xrv_stuff'), ('loc', 'xrv_loc'), ('pitching', 'xrv_pitching'), ('rv', 'rv')] %}
           100 + 10 * (100 * avg({{ col }}) - any_value(ref.m_{{ k }})) / any_value(ref.sd_{{ k }}) as {{ k }}_plus{{ ',' if not loop.last }}
           {% endfor %}
    from pw, ref group by level, snapshot_date, window_name, pitcher_id
),

base as (
    select r.*, r.pa as bf, ro.role, ro.games_std, ro.starts, fb.fb_velo, st.graded_pitches, st.stuff_plus, st.loc_plus, st.pitching_plus, st.rv_plus,
           {{ qualifier(var('min_bf_qualified'),
                        "case r.window_name when 'std' then d.team_games else d.team_games_last end", 'r.level') }} as qual_min
    from rates r
    join {{ ref('int_snapshot_dates') }} d using (level, snapshot_date)
    left join roles ro using (level, snapshot_date, pitcher_id)
    left join fb using (level, snapshot_date, window_name, pitcher_id)
    left join stuff st using (level, snapshot_date, window_name, pitcher_id)
),

pctl as (
    select s.level, s.snapshot_date, s.window_name, s.pitcher_id, {{ pool_pctls(pitcher_pctl_metrics()) }}
    from base s
    join base q on q.level = s.level and q.snapshot_date = s.snapshot_date and q.window_name = s.window_name
        and q.bf >= q.qual_min
    group by s.level, s.snapshot_date, s.window_name, s.pitcher_id
)

select b.level, {{ var('season') }} as season, b.snapshot_date, b.window_name, b.pitcher_id, b.team, b.role,
       b.games_std as games, b.starts, b.games as window_games, b.pitches, b.graded_pitches, b.bf,
       b.bf >= b.qual_min as qualified, b.qual_min,
       b.k_pct, b.bb_pct, b.csw_pct, b.whiff_pct, b.chase_pct, b.zone_pct, b.hard_hit_pct, b.xwoba, b.fb_velo, b.stuff_plus, b.loc_plus, b.pitching_plus, b.rv_plus,
       {% for m, _ in pitcher_pctl_metrics() %}p.{{ m }}_pctl{{ ',' if not loop.last }}{% endfor %}
from base b
left join pctl p using (level, snapshot_date, window_name, pitcher_id)
