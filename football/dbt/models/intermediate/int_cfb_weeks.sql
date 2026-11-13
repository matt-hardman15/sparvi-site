-- The slider's steps: every week (regular season, then the postseason as one more week) with at least one game, per season, and every team
-- that played that season (so a team on a bye still has a row that week).
select distinct w.season, w.week, t.team
from (select distinct season, week from {{ ref('stg_cfb_plays') }}) w
join (select distinct season, offense as team from {{ ref('stg_cfb_plays') }}) t using (season)
