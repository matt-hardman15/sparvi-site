select
    player_id,
    name,
    cast(birth_date as date) as birth_date,
    position,
    bats,
    throws,
    level
from {{ source('savant', 'milb_players_cur') }}
qualify row_number() over (
    partition by player_id
    order by case level when 'MLB' then 1 when 'AAA' then 2 when 'AA' then 3 else 4 end
) = 1
