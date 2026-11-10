-- Pitch-by-pitch arsenal with Stuff+ from the pitch-quality model and recent shape change.
with a as (select * from {{ ref('int_pitcher_arsenal') }}),
season as (select * from a where window_name = 'season'),
recent as (select * from a where window_name = 'recent'),
totals as (select pitcher_id, sum(pitches) as total from season group by 1)

select
    s.pitcher_id                     as player_id,
    s.pitch_type,
    s.pitches,
    s.pitches / t.total              as usage_pct,
    s.velo, s.ivb, s.hb, s.spin,
    s.whiff_pct,
    s.csw_pct,
    sq.stuff_plus,
    r.pitches                        as recent_pitches,
    r.velo - s.velo                  as velo_delta,
    r.ivb - s.ivb                    as ivb_delta,
    r.hb - s.hb                      as hb_delta
from season s
join totals t using (pitcher_id)
left join recent r using (pitcher_id, pitch_type)
left join {{ source('pitch_quality', 'pitch_type_aaa_2026') }} sq
    on sq.pitcher = s.pitcher_id and sq.pitch_type = s.pitch_type
where s.pitches >= 20
