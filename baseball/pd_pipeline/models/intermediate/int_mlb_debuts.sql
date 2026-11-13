-- Each player's first MLB game of each season, separately as a hitter and as a pitcher. Splits the
-- within-season Triple-A -> MLB translation cohort: his Triple-A line before this date, his MLB line
-- from it on.
select 'hitter' as side, season, batter_id as player_id, min(game_date) as first_mlb_game
from {{ ref('stg_pitches') }}
where level = 'MLB'
group by all
union all
select 'pitcher' as side, season, pitcher_id as player_id, min(game_date) as first_mlb_game
from {{ ref('stg_pitches') }}
where level = 'MLB'
group by all
