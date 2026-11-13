-- Season lines behind the Triple-A -> MLB translation (baseball/aaa_translation), for the last two
-- seasons at both levels. One row per side x player x season x level x segment:
--   full        every pitch that season at that level
--   pre_debut   Triple-A only: pitches before the player's first MLB game that season (players who
--               reached MLB that season). His MLB line from that game on is the MLB 'full' line.
-- Hitters are lines as the batter; pitchers as the pitcher. Pitchers also carry Stuff+, Location+,
-- Pitching+, and RV+ on the same frozen scale as the rest of the site.
with p as (
    select * from {{ ref('stg_pitches') }} where season >= {{ var('season') }} - 1
),
debuts as (select * from {{ ref('int_mlb_debuts') }}),

tagged as (
    select 'hitter' as side, batter_id as player_id, batting_team as team, p.*, 'full' as segment from p
    union all
    select 'pitcher', pitcher_id, pitching_team, p.*, 'full' from p
    union all
    select 'hitter', batter_id, batting_team, p.*, 'pre_debut' from p
    join debuts d on d.side = 'hitter' and d.season = p.season and d.player_id = p.batter_id
    where p.level = 'AAA' and p.game_date < d.first_mlb_game
    union all
    select 'pitcher', pitcher_id, pitching_team, p.*, 'pre_debut' from p
    join debuts d on d.side = 'pitcher' and d.season = p.season and d.player_id = p.pitcher_id
    where p.level = 'AAA' and p.game_date < d.first_mlb_game
),

rates as (
    select side, player_id, season, level, segment,
           mode(team)              as team,
           min(game_date)          as first_game,
           max(game_date)          as last_game,
           count(distinct game_pk) as games,
           {{ plate_discipline_rates() }},
           sum(woba_value) filter (where ends_pa) / nullif(sum(woba_denom) filter (where ends_pa), 0) as woba
    from tagged
    group by all
),

preds as (
    {% for lvl, src in [('MLB', 'pred_mlb_prev'), ('MLB', 'pred_mlb_cur'), ('AAA', 'pred_aaa_prev'), ('AAA', 'pred_aaa_cur')] %}
    select '{{ lvl }}' as level, pitcher as player_id, cast(game_date as date) as game_date,
           year(cast(game_date as date)) as season, xrv_stuff, xrv_loc, xrv_pitching, rv
    from {{ source('pitch_predictions', src) }}
    {{ 'union all' if not loop.last }}
    {% endfor %}
),
pred_tagged as (
    select p.*, 'full' as segment from preds p
    union all
    select p.*, 'pre_debut' from preds p
    join debuts d on d.side = 'pitcher' and d.season = p.season and d.player_id = p.player_id
    where p.level = 'AAA' and p.game_date < d.first_mlb_game
),
ref as (
    select {% for k in ['stuff', 'loc', 'pitching', 'rv'] %}max(mean) filter (where kind = '{{ k }}') as m_{{ k }},
           max(sd) filter (where kind = '{{ k }}') as sd_{{ k }}{{ ',' if not loop.last }}
           {% endfor %}
    from {{ source('pitch_quality', 'scale_reference') }}
),
grades as (
    select 'pitcher' as side, player_id, season, level, segment, count(*) as graded_pitches,
           {% for k, col in [('stuff', 'xrv_stuff'), ('loc', 'xrv_loc'), ('pitching', 'xrv_pitching'), ('rv', 'rv')] %}
           100 + 10 * (100 * avg({{ col }}) - any_value(ref.m_{{ k }})) / any_value(ref.sd_{{ k }}) as {{ k }}_plus{{ ',' if not loop.last }}
           {% endfor %}
    from pred_tagged, ref
    group by side, player_id, season, level, segment
)

select r.*, g.graded_pitches, g.stuff_plus, g.loc_plus, g.pitching_plus, g.rv_plus, ro.role
from rates r
left join grades g using (side, player_id, season, level, segment)
left join {{ ref('dim_pitcher_role') }} ro
    on r.side = 'pitcher' and ro.level = r.level and ro.season = r.season and ro.pitcher_id = r.player_id
