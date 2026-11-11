-- Triple-A game-by-game fastball velocity, the most common thing a PD group gets asked to watch.
select
    pitcher_id,
    game_pk,
    game_date,
    pitch_type,
    count(*)                  as pitches,
    avg(release_speed)        as velo,
    max(release_speed)        as max_velo
from {{ ref('stg_pitches') }}
where level = 'AAA'
  and season = {{ var('season') }}
  and game_date <= {{ as_of('AAA') }}
  and pitch_type in ('FF', 'SI')
group by 1, 2, 3, 4
