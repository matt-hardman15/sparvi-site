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

{% test cfb_teams_played_the_game(model) %}
{#- No team's offense or defense is credited in a game it didn't play: each play's offense and defense are the
    game's home and away teams in the games table, one each. -#}
select p.game_id, p.play_id, p.offense, p.defense
from {{ model }} p
left join {{ source('cfbd', 'games') }} g on g.id = p.game_id
where g.id is null or p.offense = p.defense
   or not ((p.offense = g.homeTeam and p.defense = g.awayTeam) or (p.offense = g.awayTeam and p.defense = g.homeTeam))
{% endtest %}

{% test cfb_unattributable_plays_bounded(model, max_share) %}
{#- Efficiency plays that name a team outside the game's two teams are dropped by stg_cfb_plays. A sliver is a
    feed glitch; more than `max_share` of a season means team names stopped matching the games table. -#}
with raw as (
    select p.season, count(*) as n,
           count(*) filter (where not ((p.offense = g.homeTeam and p.defense = g.awayTeam)
                                    or (p.offense = g.awayTeam and p.defense = g.homeTeam))) as bad
    from {{ source('cfbd', 'plays') }} p
    join {{ source('cfbd', 'games') }} g on g.id = p.gameId
    where p.ppa is not null and p.down between 1 and 4 and p.distance > 0
      and (p.playType in {{ cfb_pass_types() }} or p.playType in {{ cfb_run_types() }})
      and p.season between {{ var('cfb_first_season') }} and {{ var('cfb_season') }}
    group by 1
)
select season, n, bad from raw where bad > {{ max_share }} * n
{% endtest %}

{% test cfb_every_gap_game_decided(model) %}
{#- Every game whose play-by-play final disagrees with the official result has a decision the rule defines. -#}
select game_id, pbp_class, decision from {{ model }}
where decision is null
   or decision not in ('keep', 'repair_teams', 'repair_scores', 'keep_no_score', 'exclude')
   or (pbp_class = 'match' and decision <> 'keep')
   or (pbp_class <> 'swapped' and decision in ('repair_teams', 'repair_scores', 'keep_no_score'))
{% endtest %}
