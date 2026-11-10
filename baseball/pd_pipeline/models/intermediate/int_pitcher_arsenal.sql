-- Pitch-type shape by window, so velocity or movement changes show up as a trend.
with p as ({{ windowed(ref('stg_pitches')) }})

select
    pitcher_id,
    pitch_type,
    window_name,
    count(*)                                       as pitches,
    avg(release_speed)                             as velo,
    avg(ivb_in)                                    as ivb,
    -- arm-side positive for both hands
    avg(case when p_throws = 'R' then -hb_in else hb_in end) as hb,
    avg(release_spin_rate)                         as spin,
    avg(is_whiff::int) filter (where is_swing)     as whiff_pct,
    avg(is_csw::int)                               as csw_pct
from p
where pitch_type is not null
group by 1, 2, 3
