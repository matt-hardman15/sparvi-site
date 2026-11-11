"""Which two-week Triple-A numbers carry over to the next two weeks?

Split each configured Triple-A season (sparvi.toml) into consecutive 14-day blocks. For every hitter
(25+ PA) and pitcher (25+ batters faced) in two back-to-back blocks, correlate each
metric in block k with the same metric in block k+1. A high correlation means two
weeks of that number tells you something about the next two; a low one means it's
mostly noise at that sample size.

Reads the stg_pitches view from pd_pipeline (run `dbt build` first).
"""
import json
import os
from pathlib import Path

import duckdb
import pandas as pd

from sparvi.config import RAW, baseball
from sparvi.provenance import provenance

CFG = baseball()
HERE = Path(__file__).resolve().parent
PD = HERE.parent / "pd_pipeline"
OUT = HERE / "output"; OUT.mkdir(exist_ok=True)
BLOCK_DAYS, MIN_PA = 14, 25

os.chdir(PD)  # stg_pitches resolves parquet paths relative to the dbt project
# Single-threaded: parallel float sums vary in the last digit between runs, which would
# make every daily refresh commit "changes" even when no new games were played.
con = duckdb.connect("pd.duckdb", read_only=True, config={"threads": 1})

RATES = """
    count(*) filter (where ends_pa)                                  as pa,
    avg(is_swing::int) filter (where not in_zone)                    as chase_pct,
    avg(is_swing::int) filter (where in_zone)                        as zone_swing_pct,
    avg(is_whiff::int) filter (where is_swing)                       as whiff_pct,
    avg(is_csw::int)                                                 as csw_pct,
    avg(is_k::int) filter (where ends_pa)                            as k_pct,
    avg(is_bb::int) filter (where ends_pa)                           as bb_pct,
    avg(is_hard_hit::int) filter (where is_bip)                      as hard_hit_pct,
    sum(case when is_bip and xwoba_bip is not null then xwoba_bip else woba_value end) filter (where ends_pa)
        / nullif(sum(woba_denom) filter (where ends_pa), 0)          as xwoba,
    sum(woba_value) filter (where ends_pa) / nullif(sum(woba_denom) filter (where ends_pa), 0) as woba
"""

def blocks(side_col: str) -> pd.DataFrame:
    return con.sql(f"""
        with p as (
            select *, floor(date_diff('day', min(game_date) over (partition by season), game_date) / {BLOCK_DAYS}) as blk
            from stg_pitches
            where level = 'AAA'
        )
        select season, blk, {side_col} as player_id, {RATES},
               avg(release_speed) filter (where pitch_type in ('FF', 'SI')) as fb_velo
        from p group by all
    """).df()

def pair_r(df: pd.DataFrame, metrics: list[str]) -> dict:
    df = df[df["pa"] >= MIN_PA]
    nxt = df.assign(blk=df["blk"] - 1)
    m = df.merge(nxt, on=["season", "player_id", "blk"], suffixes=("", "_next"))
    out = {}
    for k in metrics:
        d = m[[k, f"{k}_next"]].dropna()
        out[k] = {"r": round(float(d[k].corr(d[f"{k}_next"])), 3), "n_pairs": int(len(d))}
    return out, m

hit, hit_pairs = pair_r(blocks("batter_id"),
    ["chase_pct", "zone_swing_pct", "whiff_pct", "k_pct", "bb_pct", "hard_hit_pct", "xwoba", "woba"])
pit, pit_pairs = pair_r(blocks("pitcher_id"),
    ["fb_velo", "whiff_pct", "csw_pct", "k_pct", "bb_pct", "chase_pct", "hard_hit_pct", "xwoba", "woba"])
res = {"block_days": BLOCK_DAYS, "min_pa_or_bf": MIN_PA, "seasons": CFG.aaa_seasons, "hitters": hit, "pitchers": pit,
       "provenance": provenance([RAW / f"statcast_aaa_{y}.parquet" for y in CFG.aaa_seasons])}
(OUT / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

# Small sample of pairs for the site's scatter (aggregates per player-fortnight only).
keep = ["season", "blk", "player_id", "pa"]
hp = hit_pairs[keep + [c for c in hit_pairs.columns if c.split("_next")[0] in hit and c not in keep]]
pp = pit_pairs[keep + [c for c in pit_pairs.columns if c.split("_next")[0] in pit and c not in keep]]
# Sorted so reruns on unchanged data produce identical files (DuckDB's group-by order isn't stable).
hp.sort_values(keep).round(4).to_csv(OUT / "hitter_pairs.csv", index=False)
pp.sort_values(keep).round(4).to_csv(OUT / "pitcher_pairs.csv", index=False)
print(json.dumps({k: v for k, v in res.items() if k != "provenance"}, indent=2))
