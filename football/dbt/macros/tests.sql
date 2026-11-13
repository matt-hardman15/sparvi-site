{% test value_in_range(model, column_name, min_value, max_value) %}
select * from {{ model }}
where {{ column_name }} < {{ min_value }} or {{ column_name }} > {{ max_value }}
{% endtest %}

{% test unique_combination(model, columns) %}
select {{ columns | join(', ') }}, count(*) as n
from {{ model }}
group by {{ columns | join(', ') }}
having count(*) > 1
{% endtest %}

{% test final_week_equals_season(model, season_model, cols) %}
{#- The last 'std' week of each season must equal the season totals exactly. -#}
with last as (
    select w.* from {{ model }} w
    where w.win = 'std'
      and w.week = (select max(week) from {{ model }} m where m.season = w.season)
)
select l.season, l.team, l.side, l.gt
from last l
join {{ season_model }} s using (season, team, side, gt)
where {% for c in cols %}l.{{ c }} is distinct from s.{{ c }}{{ ' or ' if not loop.last }}{% endfor %}
{% endtest %}

{% test every_team_game_scheduled(model) %}
{#- Every team-game in the play-by-play has its schedule row (same game_id and team code), so joins to
    the schedule can't silently drop games (e.g. a relocated team's old code). -#}
select g.game_id, g.team
from {{ model }} g
left join {{ source('nflverse', 'schedules') }} s
  on s.game_id = g.game_id and g.team in (s.home_team, s.away_team)
where s.game_id is null
{% endtest %}

{% test min_share_true(model, column_name, min_share, group_by) %}
{#- Within each group, at least `min_share` of rows must have `column_name` true. -#}
select {{ group_by }}, avg(({{ column_name }})::int) as share, count(*) as n
from {{ model }}
group by {{ group_by }}
having avg(({{ column_name }})::int) < {{ min_share }}
{% endtest %}

{% test cfb_completed_games_have_results(model) %}
{#- Every completed game involving an FBS team has both final scores (CFBD's official result). -#}
select id, season, homeTeam, awayTeam
from {{ source('cfbd', 'games') }}
where completed and (homePoints is null or awayPoints is null)
{% endtest %}
