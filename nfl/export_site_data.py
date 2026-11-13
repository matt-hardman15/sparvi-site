"""Write everything the NFL pages read: site/data/nfl/*.json and site/_generated/nfl.yml.

The only writer of NFL site data (see CLAUDE.md). Aggregates only: team x week, team x game, and
team x season rows, never plays. Each output declares the fields the pages read.

Inputs: the football dbt database (nfl_team_week, nfl_team_games), nfl/team_efficiency/output
(adjusted ratings and the lambda search), and data/raw/nfl/teams.parquet.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import pandas as pd

from backtest.gate import require_passing
from football.common.filters import NFL_GARBAGE_WP_HI, NFL_GARBAGE_WP_LO
from sparvi.config import RAW, ROOT, nfl
from sparvi.io import SchemaError, StaleProse, write_json, write_numbers

CFG = nfl()

# Publish gate (PLAN-backtest §6): each model's output ships only with a passing backtest for this
# exact version, or with its card's `experimental` flag (the pages then say so next to the version).
MODEL_STATUS = {m: require_passing(m) for m in ['nfl_team_efficiency']}
OUT = ROOT / "site" / "data" / "nfl"
TE = ROOT / "nfl" / "team_efficiency" / "output"
DB = Path(os.environ.get("FOOTBALL_DB_PATH", ROOT / "football" / "dbt" / "football.duckdb"))
DBT_VARS = __import__("yaml").safe_load((ROOT / "football" / "dbt" / "dbt_project.yml").read_text(encoding="utf-8"))["vars"]

con = duckdb.connect(str(DB), read_only=True)
teams = pd.read_parquet(RAW / "nfl" / "teams.parquet")
TEAM_IDS = sorted(teams["team_abbr"])
if len(TEAM_IDS) != 32:
    raise SchemaError(f"teams.parquet: {len(TEAM_IDS)} teams, expected 32")

# Rates per side, as the pages read them: short key -> (mart column, digits).
RATES = {"epa": ("epa_play", 4), "sr": ("success_rate", 4), "pass_epa": ("pass_epa_play", 4),
         "pass_sr": ("pass_success_rate", 4), "run_epa": ("run_epa_play", 4), "run_sr": ("run_success_rate", 4),
         "edpr": ("early_pass_rate", 4), "proe": ("proe", 4), "expl": ("explosive_rate", 4),
         "to": ("turnover_rate", 4), "plays": ("plays", 0)}
ADJ = {"epa_adj": "all", "pass_epa_adj": "pass", "run_epa_adj": "run"}
FIELDS_STD = ["games"] + [f"{s}_{k}" for s in ("off", "def") for k in [*RATES, *ADJ]]
FIELDS_LAST4 = ["games"] + [f"{s}_{k}" for s in ("off", "def") for k in RATES]


def clean(v, d):
    if v is None or pd.isna(v):
        return None
    return int(round(v)) if d == 0 else round(float(v), d)


OUT.mkdir(parents=True, exist_ok=True)
week = con.sql("select * from nfl_team_week order by season, team, side, gt, win, week").df()
games = con.sql("select * from nfl_team_games order by season, team, week, side, gt").df()
adj = pd.read_csv(TE / "adjusted_team_week.csv")
search = json.loads((TE / "adjustment.json").read_text(encoding="utf-8"))

weeks_by_season: dict[int, list[int]] = {}
for season in CFG.seasons:
    w = week[week["season"] == season]
    weeks = sorted(int(x) for x in w["week"].unique())
    weeks_by_season[season] = weeks
    a = adj[adj["season"] == season]
    for win, fields in (("std", FIELDS_STD), ("last4", FIELDS_LAST4)):
        ww = w[w["win"] == win]
        out_teams = []
        for t in TEAM_IDS:
            series = {}
            for gt in ("all", "no_gt"):
                s = {}
                for side in ("off", "def"):
                    rows = ww[(ww["team"] == t) & (ww["side"] == side) & (ww["gt"] == gt)].set_index("week").reindex(weeks)
                    if side == "off":
                        s["games"] = [clean(v, 0) for v in rows["games"]]
                    for k, (col, d) in RATES.items():
                        s[f"{side}_{k}"] = [clean(v, d) for v in rows[col]]
                    if win == "std":
                        for k, split in ADJ.items():
                            r = a[(a["team"] == t) & (a["gt"] == gt) & (a["split"] == split)].set_index("week").reindex(weeks)
                            s[f"{side}_{k}"] = [clean(v, 4) for v in r[f"{side}_adj"]]
                missing = [f for f in fields if f not in s]
                if missing:
                    raise SchemaError(f"team_week_{season}: missing fields {missing}")
                series[gt] = s
            out_teams.append({"id": t, "series": series})
        suffix = "" if win == "std" else "_last4"
        write_json(OUT / f"team_week_{season}{suffix}.json",
                   {"season": season, "window": win, "weeks": weeks, "fields": fields, "teams": out_teams})

    # Game log per team (team pages): result, score, and each game's rates for both variants.
    g = games[games["season"] == season]
    log = {}
    for t in TEAM_IDS:
        gt_rows = g[g["team"] == t]
        rows = []
        for (wk, _gid), gg in gt_rows.groupby(["week", "game_id"], sort=True):
            first = gg.iloc[0]
            r = {"week": int(wk), "opp": first["opponent"], "home": bool(first["is_home"]),
                 "pf": clean(first["points_for"], 0), "pa": clean(first["points_against"], 0), "res": first["outcome"]}
            for _, x in gg.iterrows():
                p = f"{x['side']}_" + ("" if x["gt"] == "all" else "ng_")
                r[p + "epa"] = clean(x["epa_play"], 4)
                r[p + "sr"] = clean(x["success_rate"], 4)
                r[p + "pass_epa"] = clean(x["pass_epa_play"], 4)
                r[p + "run_epa"] = clean(x["run_epa_play"], 4)
                r[p + "plays"] = clean(x["plays"], 0)
            rows.append(r)
        log[t] = rows
    write_json(OUT / f"team_games_{season}.json", {"season": season, "teams": log})
    print(f"team_week_{season}.json   {len(weeks)} weeks")

info = []
for _, r in teams.sort_values("team_abbr").iterrows():
    info.append({"id": r["team_abbr"], "slug": r["team_abbr"].lower(), "name": r["team_name"], "nick": r["team_nick"],
                 "conf": r["team_conf"], "division": r["team_division"], "color": r["team_color"]})
write_json(OUT / "teams.json", info)

last_game = con.sql(f"""select max(week) w from nfl_team_games
                        where season = {CFG.season} and outcome is not null""").fetchone()[0]
meta = {
    "season": CFG.season, "seasons": CFG.seasons,
    "weeks": {str(s): w for s, w in weeks_by_season.items()},
    "last_week": int(last_game) if last_game is not None else None,
    "trailing_weeks": DBT_VARS["nfl_trailing_weeks"],
    # Definitions the pages quote in tooltips (read here so no page hand-types them).
    "defs": {"explosive_pass_yds": DBT_VARS["explosive_pass_yds"], "explosive_run_yds": DBT_VARS["explosive_run_yds"],
             "neutral_wp": [DBT_VARS["nfl_neutral_wp_lo"], DBT_VARS["nfl_neutral_wp_hi"]],
             "garbage_wp": [NFL_GARBAGE_WP_LO, NFL_GARBAGE_WP_HI]},
    "adjustment": {k: search[k] for k in ("lambda", "n_games", "mae_adjusted", "mae_raw", "mae_ridge", "mae_home_only", "seasons")},
}
write_json(OUT / "meta.json", meta, indent=1)
con.close()

# ---- prose numbers (site/_generated/nfl.yml) --------------------------------------------------
if not (DBT_VARS["nfl_garbage_wp_lo"], DBT_VARS["nfl_garbage_wp_hi"]) == (NFL_GARBAGE_WP_LO, NFL_GARBAGE_WP_HI):
    raise SchemaError("garbage-time thresholds differ between dbt_project.yml and football/common/filters.py")
pct = lambda x: f"{100 * x:.0f}%"  # noqa: E731
N = {
    "season": str(CFG.season),
    "first_season": str(CFG.first_season),
    "last_week": str(meta["last_week"]),
    "garbage_wp_lo": pct(NFL_GARBAGE_WP_LO),
    "garbage_wp_hi": pct(NFL_GARBAGE_WP_HI),
    "neutral_wp_lo": pct(DBT_VARS["nfl_neutral_wp_lo"]),
    "neutral_wp_hi": pct(DBT_VARS["nfl_neutral_wp_hi"]),
    "explosive_pass_yds": str(DBT_VARS["explosive_pass_yds"]),
    "explosive_run_yds": str(DBT_VARS["explosive_run_yds"]),
    "trailing_weeks": str(DBT_VARS["nfl_trailing_weeks"]),
    "adj_lambda": f"{search['lambda']:,}",
    "adj_n_games": f"{search['n_games']:,}",
    "adj_mae": f"{search['mae_adjusted']:.2f}",
    "adj_mae_raw": f"{search['mae_raw']:.2f}",
    "adj_mae_home": f"{search['mae_home_only']:.2f}",
    "adj_first_season": str(min(search["seasons"])),
    "adj_last_season": str(max(search["seasons"])),
    "adj_min_week": str(search["min_week"]),
    "adj_mae_ridge": f"{search['mae_ridge']:.2f}",
}
# The tool's method note says: schedule-adjusted EPA predicted next week's margins about as well as raw
# (within 0.05 points of MAE), shrunk strength ratings predicted better than either, and every rating
# beat home field alone.
if not abs(search["mae_adjusted"] - search["mae_raw"]) < 0.05:
    raise StaleProse("NFL method note says schedule adjustment predicts about as well as raw EPA; it no longer does.")
if not search["mae_ridge"] < min(search["mae_adjusted"], search["mae_raw"]) < max(search["mae_adjusted"], search["mae_raw"]) < search["mae_home_only"]:
    raise StaleProse("NFL method note: shrunk ratings < adjusted/raw < home-field-only MAE no longer holds.")
def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


# ---- "Is Jacksonville the best team in football?" (nfl/jacksonville/analysis.py, frozen at its week) ----
J = json.loads((ROOT / "nfl" / "jacksonville" / "output" / "results.json").read_text(encoding="utf-8"))
jh, jp, jb = J["history"], J["predict"], J["bootstrap"]
names = {t["id"]: t["name"] for t in info}
N.update({
    "jax_season": str(J["season"]), "jax_week": str(J["week"]),
    "jax_net": f"{J['net']:+.3f}", "jax_rank": str(J["rank"]), "jax_off_rank": str(J["off_rank"]), "jax_def_rank": str(J["def_rank"]),
    "jax_record": f"{J['record']['JAX'][0]}–{J['record']['JAX'][1]}", "jax_games": str(J["games"]),
    "jax_rival": names[J["rival"]], "jax_rival_net": f"{J['rival_net']:+.3f}",
    "jax_rival_record": f"{J['record'][J['rival']][0]}–{J['record'][J['rival']][1]}",
    "jax_gap": f"{J['net'] - J['rival_net']:.3f}",
    "jax_p_first": f"{100 * jb['p_first']:.0f}%", "jax_p_ahead": f"{100 * jb['p_ahead_of_rival']:.0f}%", "jax_boot": f"{jb['samples']:,}",
    "jax_hist_n": str(jh["n"]), "jax_hist_first": str(jh["finished_first"]), "jax_hist_top3": str(jh["finished_top3"]),
    "jax_hist_playoffs": str(jh["made_playoffs"]), "jax_hist_sb": str(jh["made_sb"]), "jax_hist_won": str(jh["won_sb"]),
    "jax_hist_first_season": str(jh["seasons"][0]["season"]), "jax_hist_last_season": str(jh["seasons"][-1]["season"]),
    "jax_2017": next(f"{h['final_rank']}" for h in jh["seasons"] if h["season"] == 2017 and h["team"] == "JAX"),
    "jax_r_net": f"{jp['r_net']:.2f}", "jax_r_win": f"{jp['r_win_pct']:.2f}", "jax_pred_n": f"{jp['n']:,}",
    "jax_off_rank_ord": ordinal(J["off_rank"]), "jax_def_rank_ord": ordinal(J["def_rank"]),
    "jax_rival_off_rank_ord": ordinal(J["rival_off_rank"]), "jax_rival_def_rank_ord": ordinal(J["rival_def_rank"]),
    "jax_adj_shift": f"{abs(J['net'] - J['net_raw']):.4f}",
    "jax_rival_games": str(J["record"][J["rival"]][0] + J["record"][J["rival"]][1]),
})
if not abs(J["net"] - J["net_raw"]) < 0.001:
    raise StaleProse("JAX article: 'the opponent adjustment barely moves Jacksonville's number' no longer holds.")
write_json(OUT / "jax_history.json", {"week": J["week"], "seasons": [
    {"season": h["season"], "team": h["team"], "name": names.get(h["team"], h["team"]), "final_rank": h["final_rank"],
     "playoff_games": h["playoff_games"], "made_sb": h["made_sb"], "won_sb": h["won_sb"],
     "champion": names.get(h["champion"], h["champion"])} for h in jh["seasons"]]})
# Claims the article makes, as of its week.
if not (J["rank"] == 1 and J["rival_rank"] == 2):
    raise StaleProse("JAX article: 'Jacksonville ranks first, San Francisco second' no longer holds for its week.")
if not 0.3 <= jb["p_ahead_of_rival"] <= 0.7:
    raise StaleProse("JAX article: 'Jacksonville vs. San Francisco is close to a coin flip' no longer holds.")
if not jb["p_first"] < 0.5:
    raise StaleProse("JAX article: 'less likely than not to be the single best team' no longer holds.")
if not (jh["finished_top3"] > jh["finished_first"] and jh["won_sb"] == 0 and jh["made_playoffs"] == jh["n"]):
    raise StaleProse("JAX article: the history summary (top 3 usually, #1 rarely, every one made the playoffs, none won it) no longer holds.")
if not abs(jp["r_net"] - jp["r_win_pct"]) < 0.05:
    raise StaleProse("JAX article: 'efficiency and record predict the rest of the season about equally' no longer holds.")
write_numbers("nfl", N, "nfl/export_site_data.py")
