-- Season-to-date counts can only grow from one snapshot to the next.
with s as (
    select 'pitcher' as kind, level, pitcher_id as id, '' as sub, snapshot_date, pitches
    from {{ ref('fct_pitcher_snapshots') }} where window_name = 'std'
    union all
    select 'hitter', level, batter_id, '', snapshot_date, pitches
    from {{ ref('fct_hitter_snapshots') }} where window_name = 'std'
    union all
    select 'pitch_type', level, pitcher_id, pitch_type, snapshot_date, pitches
    from {{ ref('fct_pitch_type_snapshots') }}
)
select * from (
    select *, lag(pitches) over (partition by kind, level, id, sub order by snapshot_date) as prev from s
)
where pitches < prev
