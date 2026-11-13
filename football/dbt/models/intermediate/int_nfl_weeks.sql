-- The slider's steps: every regular-season week with at least one game, per season, and every team
-- that played that season (so a team on a bye still has a row that week).
select distinct w.season, w.week, t.team
from (select distinct season, week from {{ ref('stg_nfl_plays') }}) w
join (select distinct season, offense as team from {{ ref('stg_nfl_plays') }}) t using (season)
