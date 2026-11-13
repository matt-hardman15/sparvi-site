-- One row per team x game x side (off = the team's offense, def = plays against it) x garbage-time
-- variant ('all' plays, or 'no_gt' = garbage time removed). Additive sums only; rates come later.
with plays as (
    select *, 'all' as gt from {{ ref('stg_cfb_plays') }}
    union all
    select *, 'no_gt' as gt from {{ ref('stg_cfb_plays') }} where not is_garbage
),
off as (
    select season, week, game_id, offense as team, defense as opponent, 'off' as side, gt, {{ team_sums() }}
    from plays group by all
),
def as (
    select season, week, game_id, defense as team, offense as opponent, 'def' as side, gt, {{ team_sums() }}
    from plays group by all
)
select * from off
union all
select * from def
