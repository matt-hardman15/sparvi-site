-- One row per level and snapshot date in the current season: the last game date on or before each
-- Sunday, plus the final game date (the level's run date). Weekly, not daily: daily is too much
-- data for the browser and too noisy to read.
-- team_games / team_games_last: median games per team played in the season-to-date and trailing
-- windows; they prorate the qualified-pool minimum (macro qualifier()).
with games as (
    select distinct level, game_pk, pitching_team as team, game_date
    from {{ ref('stg_pitches') }}
    where season = {{ var('season') }} and game_date <= {{ as_of_by_level() }}
),

bounds as (
    select level, min(game_date) as first_day, max(game_date) as last_day from games group by 1
),

sundays as (
    select level, unnest(generate_series(
        first_day + ((7 - isodow(first_day)) % 7)::int, last_day, interval 7 day))::date as sunday
    from bounds
),

game_days as (select distinct level, game_date from games),

snaps as (
    select s.level, max(g.game_date) as snapshot_date
    from sundays s join game_days g on g.level = s.level and g.game_date <= s.sunday
    group by s.level, s.sunday
    union
    select level, last_day from bounds
),

team_games as (
    select
        s.level, s.snapshot_date, g.team,
        count(distinct g.game_pk) as n,
        count(distinct g.game_pk) filter (
            where g.game_date > s.snapshot_date - interval {{ var('last_days') }} day) as n_last
    from snaps s
    join games g on g.level = s.level and g.game_date <= s.snapshot_date
    group by all
)

select
    level,
    {{ var('season') }}                                          as season,
    snapshot_date,
    quantile_cont(n, 0.5)                                        as team_games,
    quantile_cont(n_last, 0.5)                                   as team_games_last,
    snapshot_date = max(snapshot_date) over (partition by level) as is_final
from team_games
group by level, snapshot_date
