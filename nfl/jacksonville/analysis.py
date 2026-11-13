"""Is Jacksonville the best team in football? (NFL article, frozen at PUB_SEASON week PUB_WEEK.)

1. Where the Jaguars rank: opponent-adjusted net EPA/play (offense minus defense, garbage time
   removed) through the publication week, from nfl/team_efficiency/output/adjusted_team_week.csv.
2. How sure is #1? Each team's games are resampled with replacement (2,000 times, seed 7) and net EPA
   per play (raw: the schedule adjustment barely moves NFL numbers, see the team-efficiency tool) is
   recomputed: the share of resamples in which Jacksonville ranks first, and leads San Francisco.
3. What history says (complete seasons 2016 on): the #1 team in adjusted net EPA/play through the same
   week, how it finished the regular season (rank) and the playoffs.
4. What predicts the rest of the season: through-week net EPA/play vs. win percentage, each correlated
   with rest-of-season point differential per game (complete seasons).

Writes output/results.json (numbers quoted by the article via nfl/export_site_data.py).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from sparvi.config import RAW, ROOT, nfl
from sparvi.provenance import provenance

PUB_SEASON, PUB_WEEK = 2026, 10      # the article's moment; the numbers stay as of then
TEAM, RIVAL = "JAX", "SF"
BOOT, SEED = 2000, 7
CFG = nfl()
OUT = Path(__file__).parent / "output"
DB = Path(os.environ.get("FOOTBALL_DB_PATH", ROOT / "football" / "dbt" / "football.duckdb"))
ADJ = ROOT / "nfl" / "team_efficiency" / "output" / "adjusted_team_week.csv"


def adjusted_at(adj: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    s = adj[(adj["season"] == season) & (adj["gt"] == "no_gt") & (adj["split"] == "all") & (adj["week"] == week)].copy()
    s["net"] = s["off_adj"] - s["def_adj"]
    s["rank"] = s["net"].rank(ascending=False, method="min").astype(int)
    s["off_rank"] = s["off_adj"].rank(ascending=False, method="min").astype(int)
    s["def_rank"] = s["def_adj"].rank(ascending=True, method="min").astype(int)
    return s.set_index("team")


def bootstrap_ranks(games: pd.DataFrame) -> dict:
    """games: team x game rows with off/def plays and EPA sums. Resample each team's games."""
    rng = np.random.default_rng(SEED)
    by_team = {t: g.reset_index(drop=True) for t, g in games.groupby("team")}
    first, ahead = 0, 0
    for _ in range(BOOT):
        net = {}
        for t, g in by_team.items():
            s = g.iloc[rng.integers(0, len(g), len(g))]
            net[t] = s["off_epa"].sum() / s["off_plays"].sum() - s["def_epa"].sum() / s["def_plays"].sum()
        best = max(net, key=net.get)
        first += best == TEAM
        ahead += net[TEAM] > net[RIVAL]
    return {"p_first": first / BOOT, "p_ahead_of_rival": ahead / BOOT, "samples": BOOT, "seed": SEED}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    adj = pd.read_csv(ADJ)
    con = duckdb.connect(str(DB), read_only=True)
    tg = con.sql(f"""select season, week, game_id, team, side, plays, epa_sum from int_nfl_team_games
                     where gt = 'no_gt' and season between {CFG.first_season} and {PUB_SEASON}""").df()
    con.close()
    g = tg.pivot_table(index=["season", "week", "game_id", "team"], columns="side", values=["plays", "epa_sum"]).reset_index()
    g.columns = ["season", "week", "game_id", "team", "def_epa", "off_epa", "def_plays", "off_plays"]

    now = adjusted_at(adj, PUB_SEASON, PUB_WEEK)
    me, rv = now.loc[TEAM], now.loc[RIVAL]
    second = now.sort_values("net", ascending=False).iloc[1]
    sched = pd.read_parquet(RAW / "nfl" / "schedules.parquet")
    reg = sched[(sched["season"] == PUB_SEASON) & (sched["game_type"] == "REG") & (sched["week"] <= PUB_WEEK) & sched["result"].notna()]
    rec = pd.concat([reg.assign(team=reg["home_team"], won=reg["result"] > 0, lost=reg["result"] < 0),
                     reg.assign(team=reg["away_team"], won=reg["result"] < 0, lost=reg["result"] > 0)])
    rec = rec.groupby("team")[["won", "lost"]].sum()
    res = {
        "season": PUB_SEASON, "week": PUB_WEEK, "team": TEAM, "rival": RIVAL,
        "net": float(me["net"]), "rank": int(me["rank"]), "off_rank": int(me["off_rank"]), "def_rank": int(me["def_rank"]),
        "net_raw": float(me["off_raw"] - me["def_raw"]), "second": second.name, "second_net": float(second["net"]),
        "rival_net": float(rv["net"]), "rival_rank": int(rv["rank"]),
        "rival_off_rank": int(rv["off_rank"]), "rival_def_rank": int(rv["def_rank"]),
        "record": {t: [int(rec.loc[t, "won"]), int(rec.loc[t, "lost"])] for t in (TEAM, RIVAL)},
        "games": int(g[(g["season"] == PUB_SEASON) & (g["week"] <= PUB_WEEK) & (g["team"] == TEAM)].shape[0]),
        "n_teams": int(len(now)),
    }
    cur = g[(g["season"] == PUB_SEASON) & (g["week"] <= PUB_WEEK)]
    res["bootstrap"] = bootstrap_ranks(cur)

    # History: the #1 team through the same week of each complete season.
    hist = []
    post = sched[sched["game_type"] != "REG"]
    for s in CFG.past_seasons:
        at = adjusted_at(adj, s, PUB_WEEK)
        if at.empty:
            continue
        top = at["net"].idxmax()
        final_week = int(adj[(adj["season"] == s)]["week"].max())
        fin = adjusted_at(adj, s, final_week)
        ps = post[post["season"] == s]
        games_t = ps[(ps["home_team"] == top) | (ps["away_team"] == top)]
        sb = ps[ps["game_type"] == "SB"]
        champ = None if sb.empty else (sb.iloc[0]["home_team"] if sb.iloc[0]["result"] > 0 else sb.iloc[0]["away_team"])
        hist.append({"season": int(s), "team": top, "final_rank": int(fin.loc[top, "rank"]),
                     "playoff_games": int(len(games_t)), "made_sb": bool(((sb["home_team"] == top) | (sb["away_team"] == top)).any()),
                     "won_sb": champ == top, "champion": champ})
    h = pd.DataFrame(hist)
    res["history"] = {"seasons": hist, "n": int(len(h)), "finished_first": int((h["final_rank"] == 1).sum()),
                      "finished_top3": int((h["final_rank"] <= 3).sum()), "made_playoffs": int((h["playoff_games"] > 0).sum()),
                      "made_sb": int(h["made_sb"].sum()), "won_sb": int(h["won_sb"].sum())}

    # Through-week net EPA vs. record, predicting rest-of-season point differential per game.
    rows = []
    for s in CFG.past_seasons:
        at = adjusted_at(adj, s, PUB_WEEK)
        r = sched[(sched["season"] == s) & (sched["game_type"] == "REG") & sched["result"].notna()]
        long = pd.concat([r.assign(team=r["home_team"], m=r["result"]), r.assign(team=r["away_team"], m=-r["result"])])
        before, after = long[long["week"] <= PUB_WEEK], long[long["week"] > PUB_WEEK]
        wp = before.groupby("team")["m"].apply(lambda x: ((x > 0).sum() + 0.5 * (x == 0).sum()) / len(x))
        rest = after.groupby("team")["m"].mean()
        rows.append(pd.DataFrame({"season": s, "net": at["net"], "win_pct": wp, "rest_margin": rest}).dropna())
    pr = pd.concat(rows)
    res["predict"] = {"n": int(len(pr)), "seasons": [int(pr["season"].min()), int(pr["season"].max())],
                      "r_net": float(pr["net"].corr(pr["rest_margin"])), "r_win_pct": float(pr["win_pct"].corr(pr["rest_margin"]))}
    res["provenance"] = provenance(sorted((RAW / "nfl").glob("pbp_*.parquet")), seeds={"bootstrap": SEED})
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "provenance"}, indent=2, default=str))


if __name__ == "__main__":
    main()
