-- Play-by-play vs. CFBD's game results (the F1/F2 foundation check): the final score the play-by-play
-- reaches for each team must equal the game's recorded points. One row per completed game with plays.
with g as (
    select id as game_id, season, homeTeam as home_team, awayTeam as away_team,
           homePoints as home_points, awayPoints as away_points
    from {{ source('cfbd', 'games') }}
    where completed and season between {{ var('cfb_first_season') }} and {{ var('cfb_season') }}
),
p as (
    -- every play (not just efficiency plays) carries the running score from each side's view
    select gameId as game_id, offense as team, offenseScore as score from {{ source('cfbd', 'plays') }}
    union all
    select gameId, defense, defenseScore from {{ source('cfbd', 'plays') }}
),
final as (select game_id, team, max(score) as score from p group by all)
select g.game_id, g.season, g.home_team, g.away_team, g.home_points, g.away_points,
       h.score as pbp_home, a.score as pbp_away,
       (h.score is not distinct from g.home_points and a.score is not distinct from g.away_points) as matches
from g
join final h on h.game_id = g.game_id and h.team = g.home_team
join final a on a.game_id = g.game_id and a.team = g.away_team
