-- Team x season x week x side x garbage-time variant x window, as of the end of that week.
--   window 'std'   = season to date (every game in weeks 1..week)
--   window 'last4' = the trailing cfb_trailing_weeks weeks (weeks week-3..week)
-- The final 'std' week equals cfb_team_season exactly (tested).
with wk as (
    select w.season, w.week, w.team, s.side, g.gt
    from {{ ref('int_cfb_weeks') }} w
    cross join (values ('off'), ('def')) s(side)
    cross join (values ('all'), ('no_gt')) g(gt)
),
windows as (
    select wk.season, wk.week, wk.team, wk.side, wk.gt, x.win,
           count(distinct g.game_id) as games,
           {%- for c in sum_cols() %}
           coalesce(sum(g.{{ c }}), 0) as {{ c }}{{ ',' if not loop.last }}
           {%- endfor %}
    from wk
    cross join (values ('std'), ('last4')) x(win)
    left join {{ ref('int_cfb_team_games') }} g
      on g.season = wk.season and g.team = wk.team and g.side = wk.side and g.gt = wk.gt
     and g.week <= wk.week
     and (x.win = 'std' or g.week > wk.week - {{ var('cfb_trailing_weeks') }})
    group by all
)
select *, {{ team_rates() }}
from windows
