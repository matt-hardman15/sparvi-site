"""Write everything the college football pages read: site/data/cfb/*.json and site/_generated/cfb.yml.

The only writer of CFB site data. Aggregates only: team x week, team x game, and team x season rows,
never plays and never raw CFBD responses (the API terms rule out redistributing raw data). Betting
lines appear only as aggregate accuracy numbers in the method text, never per game.

Inputs: the football dbt database (cfb_team_week, cfb_team_games), cfb/adjusted_ratings/output
(adjusted ratings, the power rating, metrics.json), and data/raw/cfb (FBS teams, polls).
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
TOP = ["games", "power", "rank", "elo", "ap", "cfp"]
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
         "abbr": r.get("abbreviation") or r["school"][:4].upper(), "conf": r["conference"], "color": r.get("color")}
        for _, r in ta.sort_values("school").iterrows()]
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
    "power_model": {k: pm[k] for k in ("home_points", "points_per_net_epa", "points_per_elo")},
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
    "pm_home": f"{pm['home_points']:.1f}",
    "pl_n_games": f"{PL['n_games']:,}", "pl_ap_right": f"{100 * PL['ap_right']:.1f}%", "pl_power_right": f"{100 * PL['power_right']:.1f}%",
    "pl_n_disagree": f"{PL['n_disagree']:,}", "pl_disagree_right": f"{100 * PL['power_right_when_disagree']:.0f}%",
    "pl_disagree_ci": f"{100 * PL['power_right_when_disagree_ci90'][0]:.0f}% to {100 * PL['power_right_when_disagree_ci90'][1]:.0f}%",
    "pl_first_season": str(PL["seasons"][0]), "pl_last_season": str(PL["seasons"][1]), "pl_min_week": str(PL["min_week"]),
}
if gap_rows:
    fmt = lambda r: f"{r['team']} (AP {r['ap']}, power rating {r['power']})"  # noqa: E731
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

# ---- "Did the committee pick the best 12?" (cfb/cfp/selection.py, run by hand; frozen at selection day) ----
C = json.loads((ROOT / "cfb" / "cfp" / "output" / "results.json").read_text(encoding="utf-8"))
cs, hist, cg = C["season"], C["history"], C["games"]
write_json(OUT / "cfp_selection.json", {"season": cs["season"], "final": C["final"], "boot": C["boot"], "current": cs,
                                        "bubble": C["bubble"], "history": hist})


def _team_list(rows):
    return ", ".join(f"{r['team']} (power rating #{r['power_rank']})" for r in rows) or "none"


def _wilson(k, n, z=1.645):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


bub = {b["team"]: b for b in C["bubble"]}


def share(x):
    """A resampled share, never rounded to a certainty it doesn't have."""
    return "more than 99%" if x > 0.99 else "less than 1%" if x < 0.01 else pct(x)


best_out = cs["best_left_out"]
lo, hi = _wilson(cg["power_right_when_disagree"], cg["n_disagree"])
h4, h12 = [h for h in hist if h["field_size"] == 4], [h for h in hist if h["field_size"] == 12]
N.update({
    "cfp_season": str(cs["season"]), "cfp_spots": str(cs["at_large_spots"]), "cfp_overlap": str(cs["overlap"]),
    "cfp_n_out": str(len(cs["left_out"])), "cfp_left_out": _team_list(cs["left_out"]), "cfp_instead": _team_list(cs["chosen_instead"]),
    "cfp_best_out": best_out or "none", "cfp_worst_pick": cs["worst_pick"], "cfp_worst_pick_rank": str(cs["worst_pick_power_rank"]),
    "cfp_p_best_out": share(bub[best_out]["p_above_worst_pick"]) if best_out else "n/a",
    "cfp_exp_overlap": f"{C['expected_overlap']:.1f}", "cfp_boot": f"{C['boot']:,}",
    "cfp_hist_first": str(hist[0]["season"]), "cfp_hist_last": str(hist[-1]["season"]), "cfp_hist_n": str(len(hist)),
    "cfp_hist_overlap": str(sum(h["overlap"] for h in hist)), "cfp_hist_spots": str(sum(h["at_large_spots"] for h in hist)),
    "cfp_hist_exact": str(sum(h["overlap"] == h["at_large_spots"] for h in hist)),
    "cfp_h4_n": str(len(h4)), "cfp_h4_overlap": str(sum(h["overlap"] for h in h4)), "cfp_h4_spots": str(sum(h["at_large_spots"] for h in h4)),
    "cfp_h12_n": str(len(h12)), "cfp_h12_overlap": str(sum(h["overlap"] for h in h12)), "cfp_h12_spots": str(sum(h["at_large_spots"] for h in h12)),
    "cfp_champ_power_top4": str(sum(h["champion_power_rank"] <= 4 for h in hist)),
    "cfp_champ_power_worst": str(max(h["champion_power_rank"] for h in hist)),
    "cfp_g_n": str(cg["n"]), "cfp_g_committee": str(cg["committee_right"]), "cfp_g_power": str(cg["power_right"]),
    "cfp_g_dis": str(cg["n_disagree"]), "cfp_g_dis_power": str(cg["power_right_when_disagree"]),
    "cfp_g_dis_ci": f"{100 * lo:.0f}% to {100 * hi:.0f}%",
    "cfp_g_first": str(cg["seasons"][0]), "cfp_g_last": str(cg["seasons"][1]),
    "cfp_overlap_of": f"{cs['overlap']} of {cs['at_large_spots']}",
    "cfp_four_last": str(C["field_12_from"] - 1), "cfp_twelve_first": str(C["field_12_from"]),
    "cfp_auto": str(C["auto_bids"]), "cfp_at_large12": str(C["field_12"] - C["auto_bids"]),
})
# Claims the CFP article makes. If one stops holding at selection, rewrite the sentence.
if not cs["left_out"]:
    raise StaleProse("CFP article: 'the committee left out teams the power rating prefers' no longer holds; rewrite it.")
if not sum(h["overlap"] == h["at_large_spots"] for h in hist) <= 1:
    raise StaleProse("CFP article: 'the committee's picks almost never match the power rating's exactly' no longer holds.")
if not (cg["power_right"] > cg["committee_right"] and lo > 0.5):
    raise StaleProse("CFP article: 'in playoff games the higher-rated team won more often than the higher-ranked one, "
                     "with an interval above 50% when they disagreed' no longer holds.")
if not max(h["champion_power_rank"] for h in hist) <= 5:
    raise StaleProse("CFP article: 'every champion was in the power rating's top 5 at selection' no longer holds.")
write_numbers("cfb", N, "cfb/export_site_data.py")
