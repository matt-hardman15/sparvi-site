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
# ---- Offense or defense (nfl/offense_defense; article site/nfl/offense-or-defense/) ------------------
OD = json.loads((ROOT / "nfl" / "offense_defense" / "output" / "results.json").read_text(encoding="utf-8"))
# This week's unit matchups: each offense's adjusted EPA/play (all, pass, run) against the opposing defense's,
# season to date through the latest week, with ranks. Ratings only: no predicted scores, no lines.
sched = pd.read_parquet(RAW / "nfl" / "schedules.parquet")
todo = sched[(sched["season"] == CFG.season) & (sched["game_type"] == "REG") & sched["result"].isna()]
od_games, od_week, od_through = [], None, None
if len(todo):
    od_week = int(todo["week"].min())
    cur = adj[(adj["season"] == CFG.season) & (adj["gt"] == "no_gt") & (adj["week"] < od_week)]
    od_through = int(cur["week"].max())
    latest = cur.sort_values("week").groupby(["team", "split"]).tail(1).set_index(["team", "split"])
    rank = {}
    for sp in ("all", "pass", "run"):
        z = latest.xs(sp, level="split")
        rank[(sp, "off")] = z["off_adj"].rank(ascending=False, method="min").astype(int)
        rank[(sp, "def")] = z["def_adj"].rank(ascending=True, method="min").astype(int)
    unit = lambda o, d: {sp: {"off": round(float(latest.loc[(o, sp), "off_adj"]), 3), "off_rank": int(rank[(sp, "off")][o]),  # noqa: E731
                              "def": round(float(latest.loc[(d, sp), "def_adj"]), 3), "def_rank": int(rank[(sp, "def")][d])}
                         for sp in ("all", "pass", "run")}
    for g in todo[todo["week"] == od_week].sort_values(["gameday", "game_id"]).itertuples():
        od_games.append({"date": str(g.gameday), "home": g.home_team, "away": g.away_team, "neutral": g.location == "Neutral",
                         "away_off": unit(g.away_team, g.home_team), "home_off": unit(g.home_team, g.away_team)})
write_json(OUT / "offense_defense.json", {"seasons": OD["seasons"], "min_week": OD["min_week"],
                                          "grid": {k: v["grid"] for k, v in OD["splits"].items()},
                                          "week": od_week, "through": od_through, "games": od_games})
oa, op_, orun = OD["splits"]["all"], OD["splits"]["pass"], OD["splits"]["run"]
cell = {(c["off"], c["def"]): c for c in oa["grid"]}
N.update({
    "od_first_season": str(OD["seasons"][0]), "od_last_season": str(OD["seasons"][1]), "od_min_week": str(OD["min_week"]),
    "od_n": f"{oa['n']:,}", "od_n_games": f"{oa['n_games']:,}",
    "od_r_off": f"{oa['r_off']:.2f}", "od_r_def": f"{oa['r_def']:.2f}", "od_ratio": f"{oa['ratio']:.1f}",
    "od_ratio_ci": f"{oa['ci90']['ratio'][0]:.1f} to {oa['ci90']['ratio'][1]:.1f}",
    "od_pass_r_off": f"{op_['r_off']:.2f}", "od_pass_r_def": f"{op_['r_def']:.2f}", "od_pass_ratio": f"{op_['ratio']:.1f}",
    "od_pass_ratio_ci": f"{op_['ci90']['ratio'][0]:.1f} to {op_['ci90']['ratio'][1]:.1f}",
    "od_run_r_off": f"{orun['r_off']:.2f}", "od_run_r_def": f"{orun['r_def']:.2f}", "od_run_ratio": f"{orun['ratio']:.1f}",
    "od_run_ratio_ci": f"{orun['ci90']['ratio'][0]:.1f} to {orun['ci90']['ratio'][1]:.1f}",
    "od_int_ci": f"{oa['ci90']['b_int'][0]:+.3f} to {oa['ci90']['b_int'][1]:+.3f}",
    "od_b_off": f"{oa['b_off']:.3f}", "od_b_def": f"{oa['b_def']:.3f}",
    "od_ss": f"{cell[('strong', 'strong')]['epa']:+.3f}", "od_ww": f"{cell[('weak', 'weak')]['epa']:+.3f}",
    "od_sw": f"{cell[('strong', 'weak')]['epa']:+.3f}", "od_ws": f"{cell[('weak', 'strong')]['epa']:+.3f}",
    "od_cell_n": f"{min(c['n'] for c in oa['grid']):,}",
})
# Claims the article makes. If one stops holding, rewrite the sentence.
if not oa["ci90"]["ratio"][0] > 1.5:
    raise StaleProse("NFL offense/defense article: 'the offense's numbers tell you about twice as much' (interval above 1.5) no longer holds.")
if not op_["ci90"]["ratio"][0] > 1:
    raise StaleProse("NFL offense/defense article: 'the same holds for passing' (interval above 1) no longer holds.")
if not (orun["r_off"] < op_["r_off"] and orun["r_def"] < op_["r_def"]):
    raise StaleProse("NFL offense/defense article: 'run numbers predict less than pass numbers on both sides' no longer holds.")
if not oa["ci90"]["b_int"][0] < 0 < oa["ci90"]["b_int"][1]:
    raise StaleProse("NFL offense/defense article: 'no strength-on-strength effect' (interaction interval spans zero) no longer holds.")
if not cell[("strong", "strong")]["epa"] > cell[("weak", "weak")]["epa"]:
    raise StaleProse("NFL offense/defense article: 'a strong offense against a strong defense did better than a weak one against a weak defense' no longer holds.")
# ---- Fourth-down decisions (nfl/fourth; article site/nfl/fourth-downs/) -----------------------------------------
# Aggregates only: team-season totals, the league by season, the standard-situation chart, and a short list of the
# costliest clear calls. The decision rows themselves stay in nfl/fourth/output (gitignored).
from nfl.fourth import decisions as FD  # noqa: E402
from nfl.fourth import summary as FS  # noqa: E402
from nfl.fourth.decide import PUNT_MIN_YARDLINE  # noqa: E402

FOURTH_MODELS = ["nfl_win_prob", "nfl_fourth_conversion", "nfl_fg_make", "nfl_punt_net"]
MODEL_STATUS |= {m: require_passing(m) for m in FOURTH_MODELS}
fd_meta = json.loads((FS.OUT / "decisions.json").read_text(encoding="utf-8"))["seasons"]
fd_seasons = [s for s in range(FD.FIRST_DECIDED, CFG.season + 1)]
if sorted(map(int, fd_meta)) != fd_seasons:
    raise SchemaError(f"fourth downs: decisions cover {sorted(fd_meta)}, expected {fd_seasons}; run python -m nfl.fourth.decisions")
fd = FS.load(fd_seasons)
fd_late, fd_all = FS.late_share([s for s in fd_seasons if s < CFG.season])
fd_teams, fd_league, fd_costly = FS.team_seasons(fd), FS.league(fd), FS.costliest(fd)
if set(fd_teams["team"]) != set(TEAM_IDS):
    raise SchemaError("fourth downs: team codes don't match teams.parquet")
fd_grid = json.loads((FS.OUT / f"grid_{CFG.season}.json").read_text(encoding="utf-8"))
GRID_MAX_YTG = 10
r4 = lambda v: None if v is None or pd.isna(v) else round(float(v), 4)  # noqa: E731
write_json(OUT / "fourth_downs.json", {
    "season": CFG.season, "seasons": fd_seasons, "last_week": meta["last_week"], "bootstrap": fd_meta[str(CFG.season)]["bootstrap"],
    "engine": {k: fd_meta[str(CFG.season)][k] for k in ("train_seasons", "p_block", "fg_snap", "fg_max_yardline", "kick_start", "runoff")},
    # Standard situation (tie game, 10:00 left in the 2nd quarter, outdoors, even teams): y = yards from the goal
    # line, t = yards to go; win probability of each choice, go minus the best kick (m) with its 90% interval.
    "grid": [{"y": g["yardline_100"], "t": g["ydstogo"], "go": r4(g["wp_go"]), "fg": r4(g["wp_fg"]), "punt": r4(g["wp_punt"]),
              "m": r4(g["margin"]), "lo": r4(g["margin_lo"]), "hi": r4(g["margin_hi"]), "call": g["call"], "kick": g["kick"],
              "pc": r4(g["p_convert"]), "pf": r4(g["p_fg"])} for g in fd_grid if g["ydstogo"] <= GRID_MAX_YTG],
    "teams": [{"season": int(r.season), "team": r.team, "n": int(r.decisions), "went": int(r.went), "clear_go": int(r.clear_go),
               "went_go": int(r.went_on_clear_go), "clear_kick": int(r.clear_kick), "went_kick": int(r.went_on_clear_kick),
               "toss": int(r.toss_up), "cost": r4(r.cost), "cost_passive": r4(r.cost_passive), "cost_aggr": r4(r.cost_aggressive),
               "rate": r4(r.go_rate_clear)} for r in fd_teams.itertuples()],
    "league": [{"season": int(r.season), "n": int(r.decisions), "clear_go": int(r.clear_go), "went_go": int(r.went_on_clear_go),
                "clear_kick": int(r.clear_kick), "went_kick": int(r.went_on_clear_kick), "toss": int(r.toss_up),
                "cost": r4(r.cost), "cost_passive": r4(r.cost_passive), "rate": r4(r.go_rate_clear)} for r in fd_league.itertuples()],
    "costly": [{"season": int(r.season), "week": int(r.week), "post": r.season_type != "REG", "team": r.team, "opp": r.opponent,
                "t": int(r.ydstogo), "y": int(r.yardline_100), "sd": int(r.score_differential), "sec": int(r.game_seconds_remaining),
                "ot": r.game_half == "Overtime",
                "did": r.choice, "go": r4(r.wp_go), "kick": r4(r.wp_kick), "actual": r4(r.wp_actual), "lo": r4(r.margin_lo),
                "hi": r4(r.margin_hi), "cost": r4(r.cost)} for r in fd_costly.itertuples()],
})

# Prose numbers for the article: complete seasons only (the current season is in the explorer, labeled).
LC = json.loads((FS.OUT / "late_calibration.json").read_text(encoding="utf-8"))
lc_b = {b["bucket"]: b for b in LC["buckets"]}
fd_done = fd_league[fd_league["season"] < CFG.season]
fd_tm_done = fd_teams[fd_teams["season"] < CFG.season]
tot = fd_done[["decisions", "clear_go", "went_on_clear_go", "clear_kick", "went_on_clear_kick", "toss_up", "cost", "cost_passive"]].sum()
first, last = fd_done.iloc[0], fd_done.iloc[-1]
# Change in the go rate on clear go calls, first to last complete season, with a 90% interval (two proportions).
p1, p2, n1, n2 = first["go_rate_clear"], last["go_rate_clear"], first["clear_go"], last["clear_go"]
se = (p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) ** 0.5
d_lo, d_hi = 100 * (p2 - p1 - 1.645 * se), 100 * (p2 - p1 + 1.645 * se)
rng_txt = f"90% interval {d_lo:+.0f} to {d_hi:+.0f} points"
trend = (f"That rise is more than chance alone would produce ({rng_txt})." if d_lo > 0 else
         f"That drop is more than chance alone would produce ({rng_txt})." if d_hi < 0 else
         f"A difference that size is within what chance alone produces ({rng_txt}).")


def _bt(model: str) -> dict:
    v = MODEL_STATUS[model]["version"]
    return json.loads((ROOT / "models" / "nfl" / model.removeprefix("nfl_") / "backtest" / f"{v}.json").read_text(encoding="utf-8"))


bt_wp, bt_cv, bt_fg, bt_pt = (_bt(m) for m in FOURTH_MODELS)
eng_now = fd_meta[str(CFG.season)]
N |= {
    "fd_first_season": str(int(first["season"])), "fd_last_complete": str(int(last["season"])),
    "fd_n": f"{int(tot['decisions']):,}", "fd_toss_share": pct(tot["toss_up"] / tot["decisions"]),
    "fd_clear_go_n": f"{int(tot['clear_go']):,}", "fd_go_rate": pct(tot["went_on_clear_go"] / tot["clear_go"]),
    "fd_aggr_rate": pct(tot["went_on_clear_kick"] / tot["clear_kick"]),
    "fd_passive_share": pct(tot["cost_passive"] / tot["cost"]),
    "fd_cost_team": f"{fd_tm_done['cost'].mean():.1f}", "fd_cost_max": f"{fd_tm_done['cost'].max():.1f}",
    "fd_go_rate_first": pct(p1), "fd_go_rate_last": pct(p2), "fd_trend_sentence": trend,
    "fd_late_share": pct(fd_late / fd_all), "fd_late_n": f"{fd_late:,}", "fd_late_minutes": str(FS.LATE_SECONDS // 60),
    "fd_ece_rest": f"{LC['ece_judged']:.3f}", "fd_ece_last_min": f"{lc_b['4th quarter, last 1:00']['ece']:.3f}",
    "fd_ece_ot": f"{lc_b['overtime']['ece']:.3f}", "fd_sure_n": f"{LC['leading_late']['plays']:,}",
    "fd_sure_pred": f"{100 * LC['leading_late']['predicted']:.1f}%", "fd_sure_won": f"{100 * LC['leading_late']['won']:.1f}%",
    "fd_boot": str(eng_now["bootstrap"]), "fd_p_block": f"{100 * eng_now['p_block']:.1f}%",
    "fd_fg_max": str(int(eng_now["fg_max_yardline"] + eng_now["fg_snap"])), "fd_punt_min": str(PUNT_MIN_YARDLINE),
    "fd_wp_brier": f"{bt_wp['evaluation']['point']:.3f}", "fd_wp_base": f"{bt_wp['evaluation']['baselines']['baseline']['value']:.3f}",
    "fd_punt_mae": f"{bt_pt['evaluation']['point']:.2f}", "fd_punt_base": f"{bt_pt['evaluation']['baselines']['baseline']['value']:.2f}",
    "fd_conv_ece": f"{bt_cv['evaluation']['point']:.3f}", "fd_conv_bar": f"{bt_cv['gate']['noise_bar']:.3f}",
    "fd_fg_ece": f"{bt_fg['evaluation']['point']:.3f}", "fd_fg_bar": f"{bt_fg['gate']['noise_bar']:.3f}",
}
# Claims the fourth-down article makes. If one stops holding, rewrite the sentence.
if LC["late_seconds"] != FS.LATE_SECONDS or LC["seasons"] != [s for s in fd_seasons if s < CFG.season]:
    raise SchemaError("fourth downs: late_calibration.json is out of date; run python -m nfl.fourth.late_check")
if not (lc_b["4th quarter, last 1:00"]["ece"] > 1.25 * LC["ece_judged"] and lc_b["overtime"]["ece"] > 2 * LC["ece_judged"]
        and LC["leading_late"]["won"] - LC["leading_late"]["predicted"] > 0.02):
    raise StaleProse("Fourth downs: the reasons given for not judging the end of each half and overtime (worse calibration in the "
                     "last minute and overtime; late leads rated too low) no longer hold.")
if not tot["toss_up"] / tot["decisions"] > 0.5:
    raise StaleProse("Fourth downs: 'most fourth downs are toss-ups' no longer holds.")
if not tot["cost_passive"] / tot["cost"] > 0.9:
    raise StaleProse("Fourth downs: 'the cost is almost all from kicking when going was better' (over 90%) no longer holds.")
if not tot["went_on_clear_kick"] / tot["clear_kick"] < 0.1:
    raise StaleProse("Fourth downs: 'teams rarely go when kicking is clearly better' (under 10%) no longer holds.")
if not tot["went_on_clear_go"] / tot["clear_go"] < 0.75:
    raise StaleProse("Fourth downs: the article is built on teams passing up a real share of clear go calls; that no longer holds.")
if not (bt_wp["passed"] and bt_cv["passed"] and bt_pt["passed"]) or bt_fg["passed"]:
    raise StaleProse("Fourth downs: the method box says win probability, conversion and punts passed and field goals did not; a backtest changed.")
if not bt_fg["evaluation"]["point"] < 0.026:
    raise StaleProse("Fourth downs: 'the field-goal model replaced a first version that missed by more' no longer holds.")

write_numbers("nfl", N, "nfl/export_site_data.py")
