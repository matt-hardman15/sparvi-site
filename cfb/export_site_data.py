"""Write everything the college football pages read: site/data/cfb/*.json and site/_generated/cfb.yml.

The only writer of CFB site data. Aggregates only: team x week, team x game, and team x season rows,
never plays and never raw CFBD responses (the API terms rule out redistributing raw data). Betting
lines appear only as aggregate accuracy numbers in the method text, never per game.

Inputs: the football dbt database (cfb_team_week, cfb_team_games), cfb/adjusted_ratings/output
(adjusted ratings, the power rating, metrics.json), cfb/loss_cost/output, and data/raw/cfb (FBS teams,
polls, this week's schedule).
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from backtest.gate import require_passing
from football.common.filters import CFB_GARBAGE_MARGIN
from sparvi.config import RAW, ROOT, cfb
from sparvi.io import SchemaError, StaleProse, write_json, write_numbers

CFG = cfb()

# Publish gate (PLAN-backtest §6): each model's output ships only with a passing backtest for this
# exact version, or with its card's `experimental` flag (the pages then say so next to the version).
MODEL_STATUS = {m: require_passing(m) for m in ['cfb_power_rating']}
OUT = ROOT / "site" / "data" / "cfb"
AR = ROOT / "cfb" / "adjusted_ratings" / "output"
DB = Path(os.environ.get("FOOTBALL_DB_PATH", ROOT / "football" / "dbt" / "football.duckdb"))
DBT_VARS = __import__("yaml").safe_load((ROOT / "football" / "dbt" / "dbt_project.yml").read_text(encoding="utf-8"))["vars"]
POLLS = {"ap": "AP Top 25", "cfp": "Playoff Committee Rankings"}

# Three decimals: EPA/play to a thousandth and rates to a tenth of a percent, finer than any page shows
# (keeps site/data inside the hosting budget, tests/test_site.py).
RATES = {"epa": ("epa_play", 3), "sr": ("success_rate", 3), "pass_epa": ("pass_epa_play", 3),
         "pass_sr": ("pass_success_rate", 3), "run_epa": ("run_epa_play", 3), "run_sr": ("run_success_rate", 3),
         "expl": ("explosive_rate", 3), "to": ("turnover_rate", 3), "plays": ("plays", 0)}
ADJ = {"epa_adj": "all", "pass_epa_adj": "pass", "run_epa_adj": "run"}
# pw_epa and pw_elo: the power rating's two parts in points (power = pw_epa + pw_elo before rounding).
TOP = ["games", "power", "pw_epa", "pw_elo", "rank", "elo", "ap", "cfp"]
FIELDS_STD = TOP + [f"{s}_{k}" for s in ("off", "def") for k in [*RATES, *ADJ]]
FIELDS_LAST4 = ["games"] + [f"{s}_{k}" for s in ("off", "def") for k in RATES]


def clean(v, d):
    if v is None or pd.isna(v):
        return None
    return int(round(v)) if d == 0 else round(float(v), d)


def slug(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


con = duckdb.connect(str(DB), read_only=True)
OUT.mkdir(parents=True, exist_ok=True)
week = con.sql("select * from cfb_team_week order by season, team, side, gt, win, week").df()
games = con.sql("select * from cfb_team_games order by season, team, week, side, gt").df()
adj = pd.read_csv(AR / "adjusted_team_week.csv")
power = pd.read_csv(AR / "power_team_week.csv")
M = json.loads((AR / "metrics.json").read_text(encoding="utf-8"))
PL = json.loads((AR / "polls.json").read_text(encoding="utf-8"))

# Poll rank as of the end of each week: the poll released for week w+1 (CFBD labels polls by the week
# they apply to); the final postseason poll counts after the bowls (the last week).
ranks = pd.concat([pd.read_parquet(p) for p in sorted((RAW / "cfb").glob("rankings_*.parquet"))], ignore_index=True)
last_reg = week.groupby("season")["week"].max()     # bowls are the last week in the marts

teams_all = []
weeks_by_season: dict[int, list[int]] = {}
seasons_of: dict[str, list[int]] = {}             # the seasons each team played in FBS (team pages)
poll_week: dict[str, dict[str, list[int]]] = {}   # per season and poll: the week whose poll each slider week shows
for season in CFG.seasons:
    fbs = pd.read_parquet(RAW / "cfb" / f"teams_{season}.parquet")
    ids = sorted(fbs["school"])
    conf_of = dict(zip(fbs["school"], fbs["conference"], strict=True))
    teams_all.append(fbs.assign(season=season))
    w = week[(week["season"] == season) & week["team"].isin(ids)]
    weeks = sorted(int(x) for x in w["week"].unique())
    if not weeks:
        continue
    weeks_by_season[season] = weeks
    for t in ids:
        seasons_of.setdefault(t, []).append(season)
    a = adj[adj["season"] == season]
    pw = power[power["season"] == season]
    rk = ranks[ranks["season"] == season].copy()
    rk["as_of"] = np.where(rk["seasonType"] == "postseason", max(weeks), rk["week"] - 1)
    for win, fields in (("std", FIELDS_STD), ("last4", FIELDS_LAST4)):
        ww = w[w["win"] == win]
        out_teams = []
        for t in ids:
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
                            s[f"{side}_{k}"] = [clean(v, 3) for v in r[f"{side}_adj"]]
                if win == "std":
                    p = pw[pw["team"] == t].set_index("week").reindex(weeks)
                    s["power"] = [clean(v, 1) for v in p["power"]]
                    s["pw_epa"] = [clean(v, 1) for v in p["power_epa"]]
                    s["pw_elo"] = [clean(v, 1) for v in p["power_elo"]]
                    s["rank"] = [clean(v, 0) for v in p["rank"]]
                    s["elo"] = [clean(v, 0) for v in p["elo"]]
                    for key, poll in POLLS.items():
                        r = rk[(rk["school"] == t) & (rk["poll"] == poll)].drop_duplicates("as_of", keep="last").set_index("as_of")["rank"]
                        s[key] = [clean(r.get(x), 0) for x in weeks]
                missing = [f for f in fields if f not in s]
                if missing:
                    raise SchemaError(f"team_week_{season}: missing fields {missing}")
                series[gt] = s
            out_teams.append({"id": t, "conf": conf_of.get(t), "series": series})
        if win == "std":
            # The newest week usually has no poll yet (it comes out a day or two after the games):
            # carry the latest poll forward into those trailing weeks, and record which poll week shows.
            for key in POLLS:
                has = [any(t["series"]["no_gt"][key][i] is not None for t in out_teams) for i in range(len(weeks))]
                src = list(range(len(weeks)))
                if any(has):
                    last_i = max(i for i, h in enumerate(has) if h)
                    for i in range(last_i + 1, len(weeks)):
                        src[i] = last_i
                        for t in out_teams:
                            for gt in ("all", "no_gt"):
                                t["series"][gt][key][i] = t["series"][gt][key][last_i]
                poll_week.setdefault(str(season), {})[key] = [weeks[j] for j in src]
        suffix = "" if win == "std" else "_last4"
        write_json(OUT / f"team_week_{season}{suffix}.json",
                   {"season": season, "window": win, "weeks": weeks, "fields": fields, "teams": out_teams})

    g = games[(games["season"] == season) & games["team"].isin(ids)]
    log = {}
    for t in ids:
        rows = []
        for (wk, _gid), gg in g[g["team"] == t].groupby(["week", "game_id"], sort=True):
            first = gg.iloc[0]
            r = {"week": int(wk), "opp": first["opponent"], "home": bool(first["is_home"]),
                 "pf": clean(first["points_for"], 0), "pa": clean(first["points_against"], 0), "res": first["outcome"]}
            for _, x in gg.iterrows():
                p = f"{x['side']}_" + ("" if x["gt"] == "all" else "ng_")
                r[p + "epa"] = clean(x["epa_play"], 3)
                r[p + "sr"] = clean(x["success_rate"], 3)
                r[p + "plays"] = clean(x["plays"], 0)
            rows.append(r)
        log[t] = rows
    write_json(OUT / f"team_games_{season}.json", {"season": season, "teams": log})
    print(f"team_week_{season}.json   {len(ids)} FBS teams, {len(weeks)} weeks")

ta = pd.concat(teams_all).sort_values("season").drop_duplicates("school", keep="last")
info = [{"id": r["school"], "slug": slug(r["school"]), "name": r["school"], "nick": r.get("mascot"),
         "abbr": r.get("abbreviation") or r["school"][:4].upper(), "conf": r["conference"], "color": r.get("color"),
         "seasons": seasons_of[r["school"]]}
        for _, r in ta.sort_values("school").iterrows() if r["school"] in seasons_of]
if len({t["slug"] for t in info}) != len(info):
    raise SchemaError("two CFB teams share a URL slug")
write_json(OUT / "teams.json", info)

done = con.sql(f"select max(week) from cfb_team_games where season = {CFG.season} and outcome is not null").fetchone()[0]
con.close()
bt, wt, pm = M["backtest"], M["wepa_test"], M["power_model"]
meta = {
    "season": CFG.season, "seasons": sorted(weeks_by_season),
    "weeks": {str(s): w for s, w in weeks_by_season.items()},
    "last_week": int(done) if done is not None else None,
    "min_week": M["min_week"], "trailing_weeks": DBT_VARS["cfb_trailing_weeks"],
    "defs": {"explosive_pass_yds": DBT_VARS["explosive_pass_yds"], "explosive_run_yds": DBT_VARS["explosive_run_yds"],
             "garbage_margin": {str(k): v for k, v in CFB_GARBAGE_MARGIN.items()}},
    "power_model": {**{k: pm[k] for k in ("home_points", "points_per_net_epa", "points_per_elo")},
                    # The spread of next-week margins around the power rating's forecast in its backtest: what the matchup
                    # tool's range and win chance use (cfb/adjusted_ratings/matchup.py).
                    "margin_sd": round(bt["power"]["rmse"], 2),
                    "components": pm.get("components", {})},
    "poll_week": poll_week,
}
# The current gaps between the AP poll and the power rating, for the article (as of the latest week).
cur = power[power["season"] == CFG.season]
gap_rows = []
if len(cur):
    lw = int(cur["week"].max())
    pr = cur[cur["week"] == lw].set_index("team")["rank"]
    apc = ranks[(ranks["season"] == CFG.season) & (ranks["poll"] == "AP Top 25") & (ranks["seasonType"] == "regular")]
    apc = apc[apc["week"] == apc["week"].max()].set_index("school")["rank"]
    gap_rows = sorted(({"team": t, "ap": int(a), "power": int(pr[t]), "gap": int(pr[t] - a)} for t, a in apc.items() if t in pr.index),
                      key=lambda r: -r["gap"])
    meta["poll_gaps"] = {"week": lw, "rows": gap_rows}
write_json(OUT / "meta.json", meta, indent=1)

# ---- prose numbers (site/_generated/cfb.yml) --------------------------------------------------
if {2: DBT_VARS["cfb_garbage_q2"], 3: DBT_VARS["cfb_garbage_q3"], 4: DBT_VARS["cfb_garbage_q4"]} != CFB_GARBAGE_MARGIN:
    raise SchemaError("CFB garbage-time margins differ between dbt_project.yml and football/common/filters.py")
f2 = lambda x: f"{x:.2f}"  # noqa: E731
pct = lambda x: f"{100 * x:.0f}%"  # noqa: E731
N = {
    "season": str(CFG.season), "first_season": str(CFG.first_season),
    "last_week": str(meta["last_week"]), "min_week": str(M["min_week"]),
    "garbage_q2": str(CFB_GARBAGE_MARGIN[2]), "garbage_q3": str(CFB_GARBAGE_MARGIN[3]), "garbage_q4": str(CFB_GARBAGE_MARGIN[4]),
    "trailing_weeks": str(DBT_VARS["cfb_trailing_weeks"]),
    "lambda": f"{M['lambda']:g}",
    "bt_first_season": str(min(M["seasons"])), "bt_last_season": str(max(M["seasons"])),
    "bt_n_games": f"{bt['n_games']:,}",
    **{f"bt_mae_{k}": f2(bt[k]["mae"]) for k in ("power", "elo", "ours", "raw_epa", "home_only", "line_open", "line_close")},
    **{f"bt_win_{k}": pct(bt[k]["winner"]) for k in ("power", "elo", "ours", "line_close")},
    "bt_power_vs_elo": f2(-bt["power_minus_elo_mae"]),
    "bt_power_vs_elo_ci": f"{-bt['power_minus_elo_mae_ci90'][1]:.2f} to {-bt['power_minus_elo_mae_ci90'][0]:.2f}",
    "bt_close_vs_power": f2(bt["power"]["mae"] - bt["line_close"]["mae"]),
    "wt_n_games": f"{wt['n_games']:,}", "wt_weeks": str(wt["weeks"]),
    **{f"wt_mae_{k}": f2(wt[k]["mae"]) for k in ("ours", "wepa", "sp_plus", "elo_pregame", "ours_plus_wepa", "home_only")},
    "wt_ours_vs_wepa": f2(-wt["ours_minus_wepa_mae"]),
    "wt_ours_vs_wepa_ci": f"{-wt['ours_minus_wepa_mae_ci90'][1]:.2f} to {-wt['ours_minus_wepa_mae_ci90'][0]:.2f}",
    "pm_home": f"{pm['home_points']:.1f}", "mu_sd": f"{bt['power']['rmse']:.1f}",
    # The blend's fitted weights, as printed on the tool and in the methods text.
    "pm_net": f"{pm['points_per_net_epa']:.1f}", "pm_elo": f"{pm['points_per_elo']:.4f}",
    "pm_net_tenth": f"{0.1 * pm['points_per_net_epa']:.1f}", "pm_elo_100": f"{100 * pm['points_per_elo']:.1f}",
    "pm_n_games": f"{pm['n_games']:,}",
    "pl_n_games": f"{PL['n_games']:,}", "pl_ap_right": f"{100 * PL['ap_right']:.1f}%", "pl_power_right": f"{100 * PL['power_right']:.1f}%",
    "pl_n_disagree": f"{PL['n_disagree']:,}", "pl_disagree_right": f"{100 * PL['power_right_when_disagree']:.0f}%",
    "pl_disagree_ci": f"{100 * PL['power_right_when_disagree_ci90'][0]:.0f}% to {100 * PL['power_right_when_disagree_ci90'][1]:.0f}%",
    "pl_first_season": str(PL["seasons"][0]), "pl_last_season": str(PL["seasons"][1]), "pl_min_week": str(PL["min_week"]),
}
if gap_rows:
    # Team names link to their pages (the prose is markdown, so the link renders in the article).
    fmt = lambda r: f"[{r['team']}](/cfb/teams/{slug(r['team'])}/) (AP {r['ap']}, power rating {r['power']})"  # noqa: E731
    over = [r for r in gap_rows if r["gap"] >= 10][:3]
    under = sorted((r for r in gap_rows if r["gap"] <= -3), key=lambda r: r["gap"])[:3]
    N["gap_week"] = str(meta["poll_gaps"]["week"])
    N["gap_over"] = "; ".join(map(fmt, over)) if over else "none by 10 or more spots"
    N["gap_under"] = "; ".join(map(fmt, under)) if under else "none by 3 or more spots"
    N["gap_n_over10"] = str(sum(r["gap"] >= 10 for r in gap_rows))
# Claims the CFB pages make about the validation. If one stops holding, rewrite the sentence.
if not (bt["power_minus_elo_mae"] < 0 and bt["power_minus_elo_mae_ci90"][1] < 0):
    raise StaleProse("CFB: 'the power rating predicts better than Elo' (interval excluding zero) no longer holds.")
if not bt["ours"]["mae"] < bt["raw_epa"]["mae"] < bt["home_only"]["mae"]:
    raise StaleProse("CFB: 'opponent adjustment beats raw EPA, which beats home field alone' no longer holds.")
if not bt["elo"]["mae"] < bt["ours"]["mae"]:
    raise StaleProse("CFB: 'efficiency alone predicts worse than Elo' no longer holds (good news: rewrite it).")
if not bt["line_close"]["mae"] < bt["line_open"]["mae"] < bt["power"]["mae"]:
    raise StaleProse("CFB: 'the betting market (opening and closing lines) predicts better than the power rating' no longer holds.")
if not (wt["ours_minus_wepa_mae"] < 0 and wt["ours_minus_wepa_mae_ci90"][1] < 0):
    raise StaleProse("CFB: 'our end-of-season rating beats WEPA on next season's games' no longer holds.")
if not wt["elo_pregame"]["mae"] < wt["ours"]["mae"] and wt["sp_plus"]["mae"] < wt["ours"]["mae"]:
    raise StaleProse("CFB: 'preseason Elo and SP+ beat both end-of-season efficiency ratings' no longer holds.")
if not (PL["power_right"] > PL["ap_right"] and PL["power_right_when_disagree_ci90"][0] > 0.5):
    raise StaleProse("CFB article: 'the power rating picks winners more often than the AP poll, and is right more "
                     "often when they disagree (interval above 50%)' no longer holds.")
# ---- What a loss costs (cfb/loss_cost; article site/cfb/what-a-loss-costs/) --------------------
LC = json.loads((ROOT / "cfb" / "loss_cost" / "output" / "results.json").read_text(encoding="utf-8"))
# This week's games involving a team in the latest CFP ranking, with its committee and power ranks: the
# article's table. Kickoff dates only, no predictions and no lines.
cur_games = pd.read_parquet(RAW / "cfb" / f"games_{CFG.season}.parquet")
upcoming = cur_games[~cur_games["completed"].astype(bool) & (cur_games["seasonType"] == "regular")]
cfp_now = ranks[(ranks["season"] == CFG.season) & (ranks["poll"] == POLLS["cfp"]) & (ranks["seasonType"] == "regular")]
lc_games, lc_week, lc_rank_week = [], None, None
if len(upcoming) and len(cfp_now):
    lc_week = int(upcoming["week"].min())
    lc_rank_week = int(cfp_now["week"].max())
    cr = cfp_now[cfp_now["week"] == lc_rank_week].set_index("school")["rank"]
    pr_now = power[power["season"] == CFG.season]
    pr_now = pr_now[pr_now["week"] == pr_now["week"].max()].set_index("team")["rank"]
    band_of = lambda r: next(b for b in LC["by_rank"] if b["lo"] <= r <= b["hi"])  # noqa: E731
    for g in upcoming[upcoming["week"] == lc_week].sort_values(["startDate", "homeTeam"]).itertuples():
        side = lambda t: {"team": t, "cfp": None if t not in cr.index else int(cr[t]),  # noqa: E731
                          "power": None if t not in pr_now.index else int(pr_now[t])}
        h, a_ = side(g.homeTeam), side(g.awayTeam)
        if h["cfp"] is None and a_["cfp"] is None:
            continue
        for x in (h, a_):
            if x["cfp"] is not None:
                b_ = band_of(x["cfp"])
                x["loss_drop"], x["loss_out"] = b_["drop"], b_["fell_out"]
        et = pd.Timestamp(g.startDate).tz_convert("America/New_York")      # kickoff date as listed in the US
        lc_games.append({"date": et.strftime("%Y-%m-%d"), "neutral": bool(g.neutralSite), "home": h, "away": a_})
write_json(OUT / "loss_cost.json", {**{k: LC[k] for k in ("seasons", "top", "out_rank", "by_margin", "by_rank", "points")},
                                    "week": lc_week, "rank_week": lc_rank_week, "games": lc_games})
N.update({
    "lc_first_season": str(LC["seasons"][0]), "lc_last_season": str(LC["seasons"][1]), "lc_top": str(LC["top"]),
    "lc_n_pairs": f"{LC['n_pairs']:,}", "lc_n": str(LC["n_top_losses"]),
    "lc_close_drop": f"{LC['close']['drop']:.1f}", "lc_close_power": f"{LC['close']['power_drop']:.1f}",
    "lc_big_drop": f"{LC['big']['drop']:.1f}", "lc_big_power": f"{LC['big']['power_drop']:.1f}",
    "lc_close_n": str(LC["close"]["n"]), "lc_big_n": str(LC["big"]["n"]),
    "lc_slope_committee": f"{LC['slope_committee']:.2f}", "lc_slope_power": f"{LC['slope_power']:.2f}",
    "lc_slope_committee_ci": f"{LC['slope_committee_ci90'][0]:.2f} to {LC['slope_committee_ci90'][1]:.2f}",
    "lc_slope_power_ci": f"{LC['slope_power_ci90'][0]:.2f} to {LC['slope_power_ci90'][1]:.2f}",
    "lc_per10_committee": f"{10 * LC['slope_committee']:.1f}", "lc_per10_power": f"{10 * LC['slope_power']:.1f}",
    "lc_ratio": f"{LC['slope_power'] / LC['slope_committee']:.0f}",
    "lc_opp_ranked_drop": f"{LC['by_opp']['ranked']['drop']:.1f}", "lc_opp_unranked_drop": f"{LC['by_opp']['unranked']['drop']:.1f}",
    "lc_opp_ranked_n": str(LC["by_opp"]["ranked"]["n"]), "lc_opp_unranked_n": str(LC["by_opp"]["unranked"]["n"]),
    "lc_win_rise": f"{-LC['win']['drop']:.1f}", "lc_win_n": str(LC["win"]["n"]),
    "lc_low_fell_out": f"{100 * LC['by_rank'][-1]['fell_out']:.0f}%", "lc_low_n": str(LC["by_rank"][-1]["n"]),
})
mlo, mhi = LC["by_margin"][0], LC["by_margin"][-1]
N.update({"lc_m1_drop": f"{mlo['drop']:.1f}", "lc_m1_power": f"{mlo['power_drop']:.1f}", "lc_m1_n": str(mlo["n"]),
          "lc_m5_lo": str(mhi["lo"]), "lc_m5_drop": f"{mhi['drop']:.1f}", "lc_m5_power": f"{mhi['power_drop']:.1f}", "lc_m5_n": str(mhi["n"])})
# Claims the article makes. If one stops holding, rewrite the sentence.
if not LC["slope_diff_ci90"][0] > 0:
    raise StaleProse("CFB loss article: 'the power rating weighs the margin more than the committee' (interval above zero) no longer holds.")
if not LC["slope_committee_ci90"][0] > 0:
    raise StaleProse("CFB loss article: 'the committee does weigh the margin a little' (interval above zero) no longer holds.")
if not (LC["close"]["drop"] > 2 * LC["close"]["power_drop"] and abs(LC["big"]["drop"] - LC["big"]["power_drop"]) < 1.5):
    raise StaleProse("CFB loss article: 'a close loss costs far more in the committee's ranking than in the power rating; "
                     "a lopsided one costs about the same in both' no longer holds.")
if not LC["by_opp"]["unranked"]["drop"] > LC["by_opp"]["ranked"]["drop"]:
    raise StaleProse("CFB loss article: 'losing to an unranked team costs more than losing to a ranked one' no longer holds.")
if not all(b["fell_out"] < 0.05 for b in LC["by_rank"] if b["hi"] <= LC["top"]):
    raise StaleProse("CFB loss article: 'a top-12 team almost never falls out of the ranking after one loss' no longer holds.")
# ---- CFB-CAROUSEL: what a head-coaching change does in year 1 (cfb/carousel/analysis.py) ---------------------------------
CA = json.loads((ROOT / "cfb" / "carousel" / "output" / "results.json").read_text(encoding="utf-8"))
write_json(OUT / "carousel.json", {k: CA[k] for k in ("seasons", "current_season", "summary", "changes")})
cs = CA["summary"]
ci = lambda k: f"{cs[k]['ci90'][0]:+.1f} to {cs[k]['ci90'][1]:+.1f}"  # noqa: E731
N.update({
    "ca_first_season": str(CA["seasons"][0]), "ca_last_season": str(CA["seasons"][1]), "ca_n": str(CA["n_changes"]),
    "ca_n_team_seasons": f"{CA['n_team_seasons']:,}", "ca_all": f"{-cs['all']['mean']:.1f}", "ca_all_ci": ci("all"),
    "ca_all_beat": f"{100 * cs['all']['share_beat']:.0f}%",
    "ca_left": f"{-cs['left_for_job']['mean']:.1f}", "ca_left_ci": ci("left_for_job"), "ca_left_n": str(cs["left_for_job"]["n"]),
    "ca_stay": f"{-cs['did_not_leave_for_job']['mean']:.1f}", "ca_stay_ci": ci("did_not_leave_for_job"),
    "ca_stay_n": str(cs["did_not_leave_for_job"]["n"]),
    "ca_bad": f"{cs['bad_last_year']['mean']:+.1f}", "ca_bad_ci": ci("bad_last_year"), "ca_bad_n": str(cs["bad_last_year"]["n"]),
    "ca_bad_beat": f"{100 * cs['bad_last_year']['share_beat']:.0f}%",
    "ca_notbad": f"{-cs['not_bad_last_year']['mean']:.1f}", "ca_notbad_ci": ci("not_bad_last_year"),
    "ca_g5": f"{-cs['g5']['mean']:.1f}", "ca_g5_ci": ci("g5"), "ca_power": f"{-cs['power']['mean']:.1f}", "ca_power_ci": ci("power"),
    "ca_first": f"{-cs['first_time']['mean']:.1f}", "ca_first_ci": ci("first_time"),
    "ca_exp": f"{-cs['experienced']['mean']:.1f}", "ca_exp_ci": ci("experienced"),
    "ca_sd": f"{CA['kept_effect_sd']:.1f}",
    "ca_style_r": f"{CA['style_carryover']['D1']['r_with_his_old_team']:.2f}",
    "ca_style_r_prog": f"{CA['style_carryover']['D1']['r_with_program_last_year']:.2f}",
    "ca_spread_r": f"{CA['style_carryover']['D2']['r_with_his_old_team']:.2f}",
    "ca_style_n": str(CA["style_carryover"]["coaches_who_moved"]),
    "ca_n_current": str(sum(r["in_progress"] for r in CA["changes"])),
})
# Claims the article makes. If one stops holding, rewrite the sentence.
if not cs["all"]["ci90"][1] < 0:
    raise StaleProse("CFB carousel: 'a coaching change costs a team in year 1' (interval below zero) no longer holds.")
if not (cs["left_for_job"]["mean"] < cs["did_not_leave_for_job"]["mean"] and cs["left_for_job"]["ci90"][1] < 0):
    raise StaleProse("CFB carousel: 'the drop is biggest when the coach left for another head job' no longer holds.")
if not cs["bad_last_year"]["ci90"][0] < 0 < cs["bad_last_year"]["ci90"][1]:
    raise StaleProse("CFB carousel: 'after a bottom-third season, the new coach's team lands about where expected' no longer holds.")
if not cs["g5"]["mean"] < cs["power"]["mean"]:
    raise StaleProse("CFB carousel: 'Group of Five teams lose more in year 1 than Power teams' no longer holds.")
if not cs["first_time"]["mean"] < cs["experienced"]["mean"]:
    raise StaleProse("CFB carousel: 'first-time head coaches' teams drop more than experienced coaches'' no longer holds.")
if not CA["style_carryover"]["D1"]["r_with_his_old_team"] > CA["style_carryover"]["D1"]["r_with_program_last_year"]:
    raise StaleProse("CFB carousel: 'a moving coach's pass-heaviness follows him more than it stays with the program' no longer holds.")
write_numbers("cfb", N, "cfb/export_site_data.py")
