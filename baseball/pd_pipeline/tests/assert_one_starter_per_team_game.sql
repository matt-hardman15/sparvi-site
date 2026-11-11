-- Every team in every game has exactly one starter.
select level, game_pk, team, count(*) filter (where started) as starters
from {{ ref('int_pitcher_appearances') }}
group by 1, 2, 3
having count(*) filter (where started) != 1
