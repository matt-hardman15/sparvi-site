-- One row per Triple-A pitcher, same shape as the hitter report but from the mound.
with season as (
    select * from {{ ref('int_pitcher_windows') }} where window_name = 'season'
),
recent as (
    select * from {{ ref('int_pitcher_windows') }} where window_name = 'recent'
),
pool as (
    select * from season where pa >= {{ var('min_bf_qualified') }}
),
fb_velo as (
    -- primary fastball velocity, season vs recent
    select
        pitcher_id,
        max(velo) filter (where window_name = 'season') as fb_velo,
        max(velo) filter (where window_name = 'recent') as recent_fb_velo
    from (
        select *, row_number() over (
            partition by pitcher_id, window_name order by pitches desc) as rn
        from {{ ref('int_pitcher_arsenal') }}
        where pitch_type in ('FF', 'SI')
    )
    where rn = 1
    group by 1
)

select
    s.pitcher_id                                    as player_id,
    pl.name,
    date_diff('year', pl.birth_date, date '{{ var("season") }}-06-30') as age,
    pl.throws,
    s.team,
    s.games,
    s.pa                                            as bf,
    s.pa >= {{ var('min_bf_qualified') }}           as qualified,
    s.last_game,

    s.k_pct, s.bb_pct, s.csw_pct, s.whiff_pct, s.chase_pct, s.zone_pct,
    s.hard_hit_pct, s.avg_ev, s.xwoba,
    v.fb_velo,

    r.pa                                            as recent_bf,
    r.k_pct                                         as recent_k_pct,
    r.bb_pct                                        as recent_bb_pct,
    r.whiff_pct                                     as recent_whiff_pct,
    r.xwoba                                         as recent_xwoba,
    v.recent_fb_velo,
    v.recent_fb_velo - v.fb_velo                    as fb_velo_delta,
    r.whiff_pct - s.whiff_pct                       as whiff_delta,
    r.bb_pct - s.bb_pct                             as bb_delta,

    {{ league_pct('k_pct', 'pool') }},
    {{ league_pct('bb_pct', 'pool', false) }},
    {{ league_pct('csw_pct', 'pool') }},
    {{ league_pct('whiff_pct', 'pool') }},
    {{ league_pct('chase_pct', 'pool') }},
    {{ league_pct('zone_pct', 'pool') }},
    {{ league_pct('hard_hit_pct', 'pool', false) }},
    {{ league_pct('xwoba', 'pool', false) }}
from season s
left join recent r using (pitcher_id)
left join fb_velo v using (pitcher_id)
left join {{ ref('stg_players') }} pl on pl.player_id = s.pitcher_id
