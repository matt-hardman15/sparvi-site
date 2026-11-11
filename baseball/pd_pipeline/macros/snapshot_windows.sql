{%- macro snapshot_windows(relation, date_col='game_date') -%}
{#- Copy each current-season row into every snapshot window it belongs to: 'std' (season to date)
    and 'last30' (trailing var('last_days') days). A range join on dates, aggregated by the caller;
    nothing is materialized per day. Adds level-matched snapshot_date and window_name. -#}
    select s.snapshot_date, 'std' as window_name, r.*
    from {{ ref('int_snapshot_dates') }} s
    join {{ relation }} r on r.level = s.level and r.{{ date_col }} <= s.snapshot_date
    union all
    select s.snapshot_date, 'last30' as window_name, r.*
    from {{ ref('int_snapshot_dates') }} s
    join {{ relation }} r on r.level = s.level and r.{{ date_col }} <= s.snapshot_date
        and r.{{ date_col }} > s.snapshot_date - interval {{ var('last_days') }} day
{%- endmacro %}
