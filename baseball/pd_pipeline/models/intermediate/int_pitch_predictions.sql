-- Current-season pitch-level Stuff+, Location+, and Pitching+ predictions from train.py, both levels. Exactly the pitches the
-- season-level grades use, so the final snapshot reproduces them.
select 'MLB' as level, pitcher as pitcher_id, cast(game_date as date) as game_date, pitch_type, "group", xrv_stuff, xrv_loc, xrv_pitching, rv
from {{ source('pitch_predictions', 'pred_mlb_cur') }}
union all
select 'AAA', pitcher, cast(game_date as date), pitch_type, "group", xrv_stuff, xrv_loc, xrv_pitching, rv
from {{ source('pitch_predictions', 'pred_aaa_cur') }}
