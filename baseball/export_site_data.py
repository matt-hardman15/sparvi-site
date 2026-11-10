"""Export aggregated, display-ready JSON for the website.

Only aggregates leave this script: pitcher-season and pitch-type summaries, player
report rows, and per-game averages. No pitch-level data is published.

Run after pitch_quality/train.py, pitch_quality/robustness.py,
shape_stability/analysis.py, and `dbt build` in pd_pipeline/.
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "site" / "data" / "baseball"
OUT.mkdir(parents=True, exist_ok=True)
PQ = HERE / "pitch_quality" / "output"
SS = HERE / "shape_stability" / "output"
DB = HERE / "pd_pipeline" / "pd.duckdb"
SEASON = 2026


def first_last(name: str) -> str:
    """Savant uses 'Last, First'; display as 'First Last'."""
    if isinstance(name, str) and ", " in name:
        last, first = name.split(", ", 1)
        return f"{first} {last}"
    return name


def dump(name: str, df: pd.DataFrame, digits: dict | None = None) -> None:
    df = df.copy()
    for col, d in (digits or {}).items():
        if col in df:
            df[col] = df[col].round(d)
    df = df.replace({np.nan: None})
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(df.to_dict(orient="records"), separators=(",", ":"), ensure_ascii=False),
                    encoding="utf-8")  # explicit: Windows would otherwise write cp1252 and break accented names
    print(f"{name}.json  {len(df):>6,} rows  {path.stat().st_size / 1024:>7.0f} KB")


con = duckdb.connect(str(DB), read_only=True)

# ---- Stuff+ ---------------------------------------------------------------
pitchers = []
for lvl in ["mlb", "aaa"]:
    p = pd.read_csv(PQ / f"pitcher_season_{lvl}_{SEASON}.csv")
    p["level"] = lvl.upper()
    pitchers.append(p)
p = pd.concat(pitchers)
p["name"] = p["name"].map(first_last)
dump("stuff_pitchers",
     p.rename(columns={"pitcher": "id"})[["id", "name", "team", "level", "throws", "pitches", "stuff_plus", "rv_plus"]],
     {"stuff_plus": 1, "rv_plus": 1})

pts = []
for lvl in ["mlb", "aaa"]:
    t = pd.read_csv(PQ / f"pitch_type_{lvl}_{SEASON}.csv").query("n >= 30")
    t["level"] = lvl.upper()
    pts.append(t)
t = pd.concat(pts)
t["name"] = t["player_name"].map(first_last)
dump("stuff_pitches",
     t.rename(columns={"pitcher": "id", "pitch_type": "pt"})[
         ["id", "name", "team", "level", "pt", "group", "n", "velo", "ivb", "hb", "stuff_plus"]],
     {"velo": 1, "ivb": 1, "hb": 1, "stuff_plus": 1})

yy = pd.read_csv(PQ / "yoy_pairs.csv")
yy["name"] = yy["name_25"].map(first_last)
dump("stuff_yoy", yy[["pitcher", "name", "team_26", "xrv100_25", "rv100_25", "rv100_26"]]
     .rename(columns={"pitcher": "id", "team_26": "team"}),
     {"xrv100_25": 3, "rv100_25": 3, "rv100_26": 3})

imp = pd.read_csv(PQ / "importance.csv")
dump("stuff_importance", imp, {"gain": 4})

# ---- Shape persistence -----------------------------------------------------
sc = pd.read_csv(SS / "pitch_type_changes.csv")
sc["name"] = sc["player_name"].map(first_last)
cols = ["pitcher", "name", "pitch_type", "pair"]
for m in ["velo", "ivb", "hb"]:
    cols += [f"{m}1", f"{m}2", f"{m}_next", f"{m}_chg", f"{m}_kept"]
dump("shape_changes", sc[cols].rename(columns={"pitcher": "id", "pitch_type": "pt"}),
     {c: 2 for c in cols if c not in ("pitcher", "name", "pitch_type", "pair")})

# ---- Triple-A player reports (dbt marts) ----------------------------------
h = con.sql("select * from fct_hitter_report").df()
dump("aaa_hitters", h.drop(columns=["last_game"]), {c: 4 for c in h.columns if h[c].dtype == float})

pr = con.sql("select * from fct_pitcher_report").df()
dump("aaa_pitchers", pr.drop(columns=["last_game"]), {c: 4 for c in pr.columns if pr[c].dtype == float})

ars = con.sql("select * from fct_pitcher_arsenal").df()
dump("aaa_arsenal", ars, {c: 3 for c in ars.columns if ars[c].dtype == float})

g = con.sql("""
    select pitcher_id as id, strftime(game_date, '%Y-%m-%d') as d, pitch_type as pt, pitches as n,
           round(velo, 1) as velo
    from int_pitcher_games
    qualify row_number() over (partition by pitcher_id, game_pk order by pitches desc) = 1
    order by id, d
""").df()
dump("aaa_fb_velo_by_game", g)

fl = con.sql("select * exclude (as_of_date) from rpt_flags").df()
dump("aaa_flags", fl)

meta = {
    "season": SEASON,
    "as_of": str(con.sql("select max(last_game) from fct_hitter_report").fetchone()[0]),
    "trend_days": 14,
    "min_pa_qualified": 150,
    "min_bf_qualified": 150,
    "stuff_metrics": json.loads((PQ / "metrics.json").read_text()),
    "stuff_robustness": json.loads((PQ / "robustness.json").read_text()),
    "shape_results": json.loads((SS / "results.json").read_text()),
}
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
print("meta.json written")

# ---- Two-week signal --------------------------------------------------------
TW = HERE / "two_week_signal" / "output"
tw = json.loads((TW / "results.json").read_text())
(OUT / "two_week_results.json").write_text(json.dumps(tw, separators=(",", ":")))
for side, cols in [("hitter", ["chase_pct", "whiff_pct", "hard_hit_pct", "xwoba"]),
                   ("pitcher", ["fb_velo", "whiff_pct", "bb_pct", "xwoba"])]:
    d = pd.read_csv(TW / f"{side}_pairs.csv")
    keep = [c for k in cols for c in (k, f"{k}_next")]
    dump(f"two_week_{side}_pairs", d[keep], {c: 4 for c in keep})
