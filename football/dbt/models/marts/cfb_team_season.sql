-- Team x season x side x garbage-time variant: the whole season, bowls included (or the season so far).
with s as (
    select season, team, side, gt, count(distinct game_id) as games,
           {%- for c in sum_cols() %}
           sum({{ c }}) as {{ c }}{{ ',' if not loop.last }}
           {%- endfor %}
    from {{ ref('int_cfb_team_games') }}
    group by all
)
select *, {{ team_rates() }}
from s
