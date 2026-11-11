{#- Shared by the season-level marts and the weekly snapshot marts, so both apply the same rules. -#}

{%- macro role_case(starts, games) -%}
{#- SP / RP / Swing from starts and games (thresholds are dbt vars; see dim_pitcher_role). -#}
    case
        when {{ starts }}::double / {{ games }} >= {{ var('role_sp_min_share') }}
             and {{ starts }} >= {{ var('role_sp_min_starts') }} then 'SP'
        when {{ starts }}::double / {{ games }} <= {{ var('role_rp_max_share') }} then 'RP'
        else 'Swing'
    end
{%- endmacro %}

{%- macro full_season_games(level_col='level') -%}
    case {{ level_col }}
    {%- for lvl, g in var('full_season_games').items() %} when '{{ lvl }}' then {{ g }}{% endfor %} end
{%- endmacro %}

{%- macro qualifier(full_min, team_games, level_col='level') -%}
{#- Prorated qualified-pool minimum: full_min x (team games so far / full schedule), floored. -#}
    greatest({{ var('min_qualified_floor') }},
             round({{ full_min }} * least(1.0, {{ team_games }}::double / {{ full_season_games(level_col) }})))
{%- endmacro %}

{%- macro pitcher_pctl_metrics() -%}
    {{ return([('k_pct', true), ('bb_pct', false), ('csw_pct', true), ('whiff_pct', true), ('chase_pct', true),
               ('zone_pct', true), ('hard_hit_pct', false), ('xwoba', false)]) }}
{%- endmacro %}

{%- macro hitter_pctl_metrics() -%}
    {{ return([('chase_pct', false), ('zone_swing_pct', true), ('zone_contact_pct', true), ('whiff_pct', false),
               ('k_pct', false), ('bb_pct', true), ('hard_hit_pct', true), ('ev90', true), ('xwoba', true)]) }}
{%- endmacro %}

{%- macro pool_pctls(metrics, s='s', q='q') -%}
{#- Percentile (0-100) of each metric against the pool rows q, aggregated per s row. Same arithmetic as
    league_pct(), so a snapshot and a season-level mart built from the same pitches agree exactly. -#}
    {%- for m, hib in metrics %}
    round(100 * avg(({{ q }}.{{ m }} {{ '<=' if hib else '>=' }} {{ s }}.{{ m }})::int)) as {{ m }}_pctl{{ ',' if not loop.last }}
    {%- endfor %}
{%- endmacro %}
