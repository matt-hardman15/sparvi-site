{#- Rate stats shared by the hitter and pitcher marts. Same definitions, opposite sides of the ball. -#}
{% macro plate_discipline_rates() %}
    count(*)                                                        as pitches,
    count(*) filter (where ends_pa)                                 as pa,
    avg(is_swing::int) filter (where not in_zone)                   as chase_pct,
    avg(is_swing::int) filter (where in_zone)                       as zone_swing_pct,
    1 - avg(is_whiff::int) filter (where is_swing and in_zone)      as zone_contact_pct,
    avg(is_whiff::int) filter (where is_swing)                      as whiff_pct,
    avg(is_csw::int)                                                as csw_pct,
    avg(in_zone::int)                                               as zone_pct,
    avg(is_k::int) filter (where ends_pa)                           as k_pct,
    avg(is_bb::int) filter (where ends_pa)                          as bb_pct,
    avg(is_hard_hit::int) filter (where is_bip)                     as hard_hit_pct,
    avg(launch_speed) filter (where is_bip)                         as avg_ev,
    quantile_cont(launch_speed, 0.9) filter (where is_bip)          as ev90,
    sum(case when is_bip and xwoba_bip is not null then xwoba_bip
             else woba_value end) filter (where ends_pa)
        / nullif(sum(woba_denom) filter (where ends_pa), 0)         as xwoba
{% endmacro %}

{%- macro windowed(relation) -%}
{#- Season-to-date and trailing-window copies of the pitch set, as of the run date. -#}
    select *, 'season' as window_name
    from {{ relation }}
    where season = {{ var('season') }}
      and game_date <= date '{{ var("as_of_date") }}'
    union all
    select *, 'recent' as window_name
    from {{ relation }}
    where season = {{ var('season') }}
      and game_date <= date '{{ var("as_of_date") }}'
      and game_date > date '{{ var("as_of_date") }}' - interval {{ var('trend_days') }} day
{%- endmacro %}

{%- macro league_pct(metric, pool, higher_is_better=true) -%}
{#- Percentile of this player's value within the qualified league pool (0-100).
    Works for unqualified players too: they're placed against the qualified distribution. -#}
    round(100 * (
        select avg(({{ 'q.' ~ metric }} {{ '<=' if higher_is_better else '>=' }} s.{{ metric }})::int)
        from {{ pool }} q
    )) as {{ metric }}_pctl
{%- endmacro %}
