-- One row per regular-season pitch at both levels: Triple-A (last season and this one) and
-- MLB (the last three). `level` is 'AAA' or 'MLB'; every downstream model filters or groups on it.
-- Carries the flags every downstream rate is built from.
with raw as (
    select *, 'AAA' as level from {{ source('savant', 'statcast_aaa_prev') }}
    union all by name
    select *, 'AAA' as level from {{ source('savant', 'statcast_aaa_cur') }}
    union all by name
    select *, 'MLB' as level from {{ source('savant', 'statcast_mlb_prev2') }}
    union all by name
    select *, 'MLB' as level from {{ source('savant', 'statcast_mlb_prev') }}
    union all by name
    select *, 'MLB' as level from {{ source('savant', 'statcast_mlb_cur') }}
)

select
    level || '-' || cast(game_pk as varchar) || '-' || cast(at_bat_number as varchar)
        || '-' || cast(pitch_number as varchar)              as pitch_id,
    level,
    game_pk,
    cast(game_date as date)                                  as game_date,
    game_year                                                as season,
    pitcher                                                  as pitcher_id,
    batter                                                   as batter_id,
    p_throws,
    stand,
    case when inning_topbot = 'Top' then home_team else away_team end as pitching_team,
    case when inning_topbot = 'Top' then away_team else home_team end as batting_team,
    at_bat_number,
    pitch_number,
    inning,
    balls,
    strikes,
    pitch_type,
    release_speed,
    pfx_x * 12                                               as hb_in,
    pfx_z * 12                                               as ivb_in,
    release_spin_rate,
    release_extension,
    zone,
    description,
    events,
    launch_speed,
    launch_angle,
    estimated_woba_using_speedangle                          as xwoba_bip,
    woba_value,
    woba_denom,

    -- Swing decisions. Bunt attempts are excluded from swings on purpose.
    zone between 1 and 9                                     as in_zone,
    description in ('foul', 'foul_tip', 'hit_into_play',
                    'swinging_strike', 'swinging_strike_blocked') as is_swing,
    description in ('swinging_strike', 'swinging_strike_blocked',
                    'foul_tip')                              as is_whiff,
    description in ('called_strike', 'swinging_strike',
                    'swinging_strike_blocked', 'foul_tip')   as is_csw,
    description = 'hit_into_play'                            as is_bip,
    events is not null                                       as ends_pa,
    events in ('strikeout', 'strikeout_double_play')         as is_k,
    events in ('walk', 'intent_walk')                        as is_bb,
    case
        when events in ('triple_play') then 3
        when events in ('grounded_into_double_play', 'double_play', 'strikeout_double_play',
                        'sac_fly_double_play', 'sac_bunt_double_play') then 2
        when events in ('strikeout', 'field_out', 'force_out', 'fielders_choice_out',
                        'sac_fly', 'sac_bunt', 'other_out', 'pickoff_1b', 'pickoff_2b', 'pickoff_3b')
          or events like 'caught_stealing%' or events like 'pickoff_caught_stealing%' then 1
        else 0
    end                                                      as outs_on_play,
    description = 'hit_into_play' and launch_speed >= 95     as is_hard_hit
from raw
where game_type = 'R'
  and description not in ('pitchout', 'intent_ball', 'automatic_ball', 'automatic_strike')
