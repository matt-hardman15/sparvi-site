-- Pitcher x pitch type x snapshot date, season to date (cumulative only), both levels.
-- Shape and whiff rate from every pitch; Stuff+ from the graded pitches, scaled within the pitch's
-- group on the frozen prior-season scale (scale_reference kind = 'pitch_type').
with pitches as (
    select * from {{ ref('stg_pitches') }}
    where season = {{ var('season') }} and game_date <= {{ as_of_by_level() }} and pitch_type is not null
),
cum as (
    select s.snapshot_date, p.*
    from {{ ref('int_snapshot_dates') }} s
    join pitches p on p.level = s.level and p.game_date <= s.snapshot_date
),
shape as (
    select level, snapshot_date, pitcher_id, pitch_type,
           count(*)                                              as pitches,
           avg(release_speed)                                    as velo,
           avg(ivb_in)                                           as ivb,
           avg(case when p_throws = 'R' then -hb_in else hb_in end) as hb,
           avg(release_spin_rate)                                as spin,
           avg(is_whiff::int) filter (where is_swing)            as whiff_pct
    from cum group by all
),
preds as (
    select s.snapshot_date, x.*
    from {{ ref('int_snapshot_dates') }} s
    join {{ ref('int_pitch_predictions') }} x on x.level = s.level and x.game_date <= s.snapshot_date
),
ref as (select grp, mean, sd from {{ source('pitch_quality', 'scale_reference') }} where kind = 'pitch_type'),
stuff as (
    select p.level, p.snapshot_date, p.pitcher_id, p.pitch_type, any_value(p."group") as pitch_group,
           count(*) as graded,
           100 + 10 * (100 * avg(p.xrv_stuff) - any_value(r.mean)) / any_value(r.sd) as stuff_plus
    from preds p join ref r on r.grp = p."group"
    group by p.level, p.snapshot_date, p.pitcher_id, p.pitch_type
)
select s.level, {{ var('season') }} as season, s.snapshot_date, s.pitcher_id, s.pitch_type, st.pitch_group,
       s.pitches, s.pitches / sum(s.pitches) over (partition by s.level, s.snapshot_date, s.pitcher_id) as usage_pct,
       s.velo, s.ivb, s.hb, s.spin, s.whiff_pct, st.graded, st.stuff_plus
from shape s
left join stuff st using (level, snapshot_date, pitcher_id, pitch_type)
