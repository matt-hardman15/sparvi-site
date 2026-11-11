-- One row per pitcher per game, both levels, every season in staging.
-- `started` = he threw his team's first pitch of the game (the earliest at-bat and pitch on his
-- team's defensive side). Not "pitched in the 1st inning": an opener and the bulk guy who follows
-- him would both count under that rule.
-- outs_recorded comes from PA-ending events, plus caught-stealing/pickoff events Statcast records;
-- it can undercount outs that happen mid-PA (e.g. a runner thrown out on a wild pitch).
with p as (
    select *
    from {{ ref('stg_pitches') }}
    where game_date <= {{ as_of_by_level() }} or season < {{ var('season') }}
),

team_first_pitch as (
    select
        level,
        game_pk,
        pitching_team,
        arg_min(pitcher_id, at_bat_number * 1000 + pitch_number) as starter_id
    from p
    group by 1, 2, 3
)

select
    p.level,
    p.season,
    p.game_pk,
    min(p.game_date)                            as game_date,
    p.pitcher_id,
    p.pitching_team                             as team,
    bool_or(p.pitcher_id = f.starter_id)        as started,
    count(*)                                    as pitches,
    count(*) filter (where p.ends_pa)           as batters_faced,
    sum(p.outs_on_play)                         as outs_recorded,
    min(p.inning)                               as first_inning
from p
join team_first_pitch f
    on f.level = p.level and f.game_pk = p.game_pk and f.pitching_team = p.pitching_team
group by p.level, p.season, p.game_pk, p.pitcher_id, p.pitching_team
