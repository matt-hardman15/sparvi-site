-- Every Triple-A hitter's rate profile for the season to date and the trailing window.
with p as ({{ windowed(ref('stg_pitches')) }})

select
    batter_id,
    window_name,
    mode(batting_team)       as team,
    min(game_date)           as first_game,
    max(game_date)           as last_game,
    count(distinct game_pk)  as games,
    {{ plate_discipline_rates() }}
from p
group by 1, 2
