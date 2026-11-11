-- One row per Triple-A hitter: season-to-date line, trailing-window line, change, and
-- percentile vs qualified Triple-A hitters. This is the table the report cards read.
with season as (
    select * from {{ ref('int_hitter_windows') }} where window_name = 'season'
),
recent as (
    select * from {{ ref('int_hitter_windows') }} where window_name = 'recent'
),
qual as (
    select {{ qualifier(var('min_pa_qualified'), 'team_games') }} as qual_min
    from {{ ref('int_snapshot_dates') }} where level = 'AAA' and is_final
),
pool as (
    -- Prorated qualifier at the run date (same rule as the snapshots, so the final snapshot matches).
    select * from season where pa >= (select qual_min from qual)
)

select
    s.batter_id                                     as player_id,
    pl.name,
    date_diff('year', pl.birth_date, date '{{ var("season") }}-06-30') as age,
    pl.bats,
    pl.position,
    s.team,
    s.games,
    s.pa,
    s.pa >= (select qual_min from qual)             as qualified,
    s.last_game,

    s.chase_pct, s.zone_swing_pct, s.zone_contact_pct, s.whiff_pct,
    s.k_pct, s.bb_pct, s.hard_hit_pct, s.avg_ev, s.ev90, s.xwoba,

    r.pa                                            as recent_pa,
    r.chase_pct                                     as recent_chase_pct,
    r.whiff_pct                                     as recent_whiff_pct,
    r.hard_hit_pct                                  as recent_hard_hit_pct,
    r.xwoba                                         as recent_xwoba,
    r.chase_pct - s.chase_pct                       as chase_delta,
    r.whiff_pct - s.whiff_pct                       as whiff_delta,
    r.hard_hit_pct - s.hard_hit_pct                 as hard_hit_delta,

    {{ league_pct('chase_pct', 'pool', false) }},
    {{ league_pct('zone_swing_pct', 'pool') }},
    {{ league_pct('zone_contact_pct', 'pool') }},
    {{ league_pct('whiff_pct', 'pool', false) }},
    {{ league_pct('k_pct', 'pool', false) }},
    {{ league_pct('bb_pct', 'pool') }},
    {{ league_pct('hard_hit_pct', 'pool') }},
    {{ league_pct('ev90', 'pool') }},
    {{ league_pct('xwoba', 'pool') }}
from season s
left join recent r using (batter_id)
left join {{ ref('stg_players') }} pl on pl.player_id = s.batter_id
