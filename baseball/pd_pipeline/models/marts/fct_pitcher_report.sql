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
    -- Primary fastball = the most-thrown FF/SI over the season (ties go to FF, so the pick is
    -- deterministic). The recent window uses that same pitch type, so the delta compares like
    -- with like instead of, say, a recent sinker against a season four-seamer.
    select
        s.pitcher_id,
        s.velo as fb_velo,
        r.velo as recent_fb_velo
    from (
        select *, row_number() over (partition by pitcher_id order by pitches desc, pitch_type) as rn
        from {{ ref('int_pitcher_arsenal') }}
        where window_name = 'season' and pitch_type in ('FF', 'SI')
    ) s
    left join {{ ref('int_pitcher_arsenal') }} r
        on r.pitcher_id = s.pitcher_id and r.pitch_type = s.pitch_type and r.window_name = 'recent'
    where s.rn = 1
)

select
    s.pitcher_id                                    as player_id,
    pl.name,
    date_diff('year', pl.birth_date, date '{{ var("season") }}-06-30') as age,
    pl.throws,
    s.team,
    s.games,
    d.starts                                        as games_started,
    d.role,
    s.pitches,
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
left join {{ ref('dim_pitcher_role') }} d
    on d.pitcher_id = s.pitcher_id and d.level = 'AAA' and d.season = {{ var('season') }}
