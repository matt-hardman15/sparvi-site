-- Every Triple-A pitcher's rate profile (batters faced) for the season and trailing window.
with p as ({{ windowed(ref('stg_pitches')) }})

select
    pitcher_id,
    window_name,
    mode(pitching_team)      as team,
    min(game_date)           as first_game,
    max(game_date)           as last_game,
    count(distinct game_pk)  as games,
    {{ plate_discipline_rates() }}
from p
group by 1, 2
