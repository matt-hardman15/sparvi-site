-- What changed in the last two weeks, per player. affiliate = 'ALL' keeps every Triple-A club.
-- Thresholds are deliberately simple and live here, in one place, so staff can argue with them.
with hitters as (
    select player_id, name, team, 'hitter' as role, recent_pa as recent_n,
        case
            when recent_pa >= 30 and chase_delta >= 0.05
                then 'Chase rate up ' || cast(round(100 * chase_delta) as int) || ' pts over last {{ var("trend_days") }} days'
            when recent_pa >= 30 and chase_delta <= -0.05
                then 'Chase rate down ' || cast(round(-100 * chase_delta) as int) || ' pts over last {{ var("trend_days") }} days'
        end as flag_discipline,
        case
            when recent_pa >= 30 and hard_hit_delta <= -0.10
                then 'Hard-hit rate down ' || cast(round(-100 * hard_hit_delta) as int) || ' pts'
            when recent_pa >= 30 and hard_hit_delta >= 0.10
                then 'Hard-hit rate up ' || cast(round(100 * hard_hit_delta) as int) || ' pts'
        end as flag_contact
    from {{ ref('fct_hitter_report') }}
    where team = '{{ var("affiliate") }}' or '{{ var("affiliate") }}' = 'ALL'
),
pitchers as (
    select player_id, name, team, 'pitcher' as role, recent_bf as recent_n,
        case
            when recent_bf >= 10 and fb_velo_delta <= -1.0
                then 'Fastball velo down ' || round(-fb_velo_delta, 1) || ' mph vs season'
            when recent_bf >= 10 and fb_velo_delta >= 1.0
                then 'Fastball velo up ' || round(fb_velo_delta, 1) || ' mph vs season'
        end as flag_discipline,
        case
            when recent_bf >= 25 and bb_delta >= 0.05
                then 'Walk rate up ' || cast(round(100 * bb_delta) as int) || ' pts'
            when recent_bf >= 25 and whiff_delta >= 0.07
                then 'Whiff rate up ' || cast(round(100 * whiff_delta) as int) || ' pts'
        end as flag_contact
    from {{ ref('fct_pitcher_report') }}
    where team = '{{ var("affiliate") }}' or '{{ var("affiliate") }}' = 'ALL'
)

select date '{{ var("as_of_date") }}' as as_of_date, *
from (
    select * from hitters
    union all
    select * from pitchers
)
where coalesce(flag_discipline, flag_contact) is not null
