{#- One set of football definitions, used by every model (and documented in the glossary). -#}

{%- macro nfl_garbage_time(qtr='qtr', wp='wp') -%}
{#- 4th-quarter play with the offense's win probability outside the band. Missing WP = not garbage.
    Same rule as football/common/filters.py (tested). -#}
    coalesce({{ qtr }} = {{ var('nfl_garbage_qtr') }}
             and ({{ wp }} < {{ var('nfl_garbage_wp_lo') }} or {{ wp }} > {{ var('nfl_garbage_wp_hi') }}), false)
{%- endmacro %}

{%- macro nfl_efficiency_play() -%}
{#- The plays every efficiency rate counts: passes (incl. sacks and scrambles) and runs with an EPA,
    excluding two-point tries, kneel-downs, and spikes. -#}
    (pass = 1 or rush = 1) and epa is not null
    and coalesce(two_point_attempt, 0) = 0 and coalesce(qb_kneel, 0) = 0 and coalesce(qb_spike, 0) = 0
{%- endmacro %}

{%- macro team_sums() -%}
{#- Additive play counts and sums for one team side; every rate is built from these, so a weekly
    window, a season, and a game all use the same arithmetic. -#}
    count(*)                                              as plays,
    sum(epa)                                              as epa_sum,
    sum(is_success::int)                                  as successes,
    count(*) filter (where is_pass)                       as pass_plays,
    coalesce(sum(epa) filter (where is_pass), 0)          as pass_epa_sum,
    count(*) filter (where is_pass and is_success)        as pass_successes,
    count(*) filter (where not is_pass)                   as run_plays,
    coalesce(sum(epa) filter (where not is_pass), 0)      as run_epa_sum,
    count(*) filter (where not is_pass and is_success)    as run_successes,
    count(*) filter (where is_neutral_early)              as early_plays,
    count(*) filter (where is_neutral_early and is_pass)  as early_passes,
    count(*) filter (where xpass is not null)             as xpass_plays,
    coalesce(sum(xpass), 0)                               as xpass_sum,
    count(*) filter (where xpass is not null and is_pass) as xpass_passes,
    count(*) filter (where is_explosive)                  as explosives,
    count(*) filter (where is_turnover)                   as turnovers
{%- endmacro %}

{%- macro sum_cols() -%}
{{ return(['plays', 'epa_sum', 'successes', 'pass_plays', 'pass_epa_sum', 'pass_successes', 'run_plays',
           'run_epa_sum', 'run_successes', 'early_plays', 'early_passes', 'xpass_plays', 'xpass_sum',
           'xpass_passes', 'explosives', 'turnovers']) }}
{%- endmacro %}

{%- macro team_rates(t='') -%}
{#- Rates from the sums above (t = optional table alias with a dot). Null when the denominator is 0. -#}
    {{ t }}epa_sum / nullif({{ t }}plays, 0)                         as epa_play,
    {{ t }}successes::double / nullif({{ t }}plays, 0)               as success_rate,
    {{ t }}pass_epa_sum / nullif({{ t }}pass_plays, 0)               as pass_epa_play,
    {{ t }}pass_successes::double / nullif({{ t }}pass_plays, 0)     as pass_success_rate,
    {{ t }}run_epa_sum / nullif({{ t }}run_plays, 0)                 as run_epa_play,
    {{ t }}run_successes::double / nullif({{ t }}run_plays, 0)       as run_success_rate,
    {{ t }}early_passes::double / nullif({{ t }}early_plays, 0)      as early_pass_rate,
    ({{ t }}xpass_passes - {{ t }}xpass_sum) / nullif({{ t }}xpass_plays, 0) as proe,
    {{ t }}explosives::double / nullif({{ t }}plays, 0)              as explosive_rate,
    {{ t }}turnovers::double / nullif({{ t }}plays, 0)               as turnover_rate
{%- endmacro %}
