"""Opponent-adjusted offense and defense EPA/play, as of every week, and the choice of ridge lambda.

1. Lambda search (complete seasons only). For every season and week w from MIN_WEEK on, fit the
   ratings on plays through week w and predict each week-(w+1) game's home margin:
       margin = a * home_site + b * (net_home - net_away),   net = off - def
   a and b are fit leave-one-season-out, so no season's games inform their own prediction. Lambda
   sets how much the team-strength estimates (the ridge coefficients) are shrunk; the lambda whose
   ridge ratings predict best wins, and those strengths are what the schedule correction subtracts.
   The same file reports, at that lambda, the schedule-corrected numbers the site shows
   (off_adj/def_adj), raw EPA/play, and home field alone, so the page can say how much the
   adjustment itself adds.
2. Ratings for every season x week x garbage-time variant x split (all plays, passes, runs), with
   the chosen lambda, on season-to-date plays.

Reads the dbt database (stg_nfl_plays) and the schedules; writes
    output/adjusted_team_week.csv   season, week, team, gt, split, off/def adjusted and raw, plays
    output/adjustment.json          lambda grid with MAE, chosen lambda, baselines, n games
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from football.common.adjust import ridge_ratings
from sparvi.config import RAW, ROOT, nfl
from sparvi.provenance import provenance

CFG = nfl()
OUT = Path(__file__).parent / "output"
DB = Path(os.environ.get("FOOTBALL_DB_PATH", ROOT / "football" / "dbt" / "football.duckdb"))
GRID = [25, 50, 100, 200, 400, 800, 1600, 3200]
MIN_WEEK = 3          # first week whose ratings predict the next week in the search
SEARCH_GT = "no_gt"   # the tool's default view: garbage time removed, all plays


def load_plays() -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    df = con.sql("""select season, week, game_id, offense, defense, is_home::int as is_home, is_pass,
                           is_garbage, epa from stg_nfl_plays order by season, week, game_id""").df()
    con.close()
    return df


def variant(df: pd.DataFrame, gt: str, split: str) -> pd.DataFrame:
    d = df if gt == "all" else df[~df["is_garbage"]]
    if split == "pass":
        d = d[d["is_pass"]]
    elif split == "run":
        d = d[~d["is_pass"]]
    return d


def games() -> pd.DataFrame:
    s = pd.read_parquet(RAW / "nfl" / "schedules.parquet")
    s = s[(s["game_type"] == "REG") & s["result"].notna()]
    s = s.assign(home_site=(s["location"] != "Neutral").astype(int), margin=s["home_score"] - s["away_score"])
    return s[["season", "week", "game_id", "home_team", "away_team", "home_site", "margin"]]


def net(r: pd.DataFrame, kind: str) -> pd.Series:
    r = r.set_index("team")
    return r[f"off_{kind}"] - r[f"def_{kind}"]


def predictions(plays: pd.DataFrame, g: pd.DataFrame, lam: float, seasons: list[int]) -> pd.DataFrame:
    """Next-week games with the rating gap from data through the prior week (adjusted and raw)."""
    rows = []
    for s in seasons:
        ps, gs = plays[plays["season"] == s], g[g["season"] == s]
        for w in range(MIN_WEEK, int(gs["week"].max())):
            nxt = gs[gs["week"] == w + 1]
            if nxt.empty:
                continue
            r = ridge_ratings(ps[ps["week"] <= w], lam)
            d = {f"d_{kind}": nxt["home_team"].map(net(r, kind)).to_numpy() - nxt["away_team"].map(net(r, kind)).to_numpy()
                 for kind in ("adj", "raw", "ridge")}
            rows.append(nxt.assign(**d))
    return pd.concat(rows, ignore_index=True)


def loso_mae(p: pd.DataFrame, cols: list[str]) -> float:
    """MAE of margin ~ home_site + cols, each season predicted from a fit on the other seasons."""
    err = []
    for s in sorted(p["season"].unique()):
        tr, te = p[p["season"] != s], p[p["season"] == s]
        X = tr[["home_site", *cols]].to_numpy(float)
        coef, *_ = np.linalg.lstsq(X, tr["margin"].to_numpy(float), rcond=None)
        err.append(np.abs(te["margin"].to_numpy(float) - te[["home_site", *cols]].to_numpy(float) @ coef))
    return float(np.mean(np.concatenate(err)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plays = load_plays()
    g = games()
    search_seasons = CFG.past_seasons
    base = variant(plays, SEARCH_GT, "all")

    grid = []
    for lam in GRID:
        p = predictions(base, g, lam, search_seasons)
        grid.append({"lambda": lam, "mae": loso_mae(p, ["d_ridge"]), "mae_adjusted": loso_mae(p, ["d_adj"]),
                     "n_games": int(len(p))})
        print(f"lambda {lam:>5}: MAE ridge {grid[-1]['mae']:.3f}, schedule-adjusted {grid[-1]['mae_adjusted']:.3f}")
    best = min(grid, key=lambda r: (round(r["mae"], 6), r["lambda"]))
    lam = best["lambda"]
    p = predictions(base, g, lam, search_seasons)
    res = {
        "seasons": search_seasons, "min_week": MIN_WEEK, "variant": {"gt": SEARCH_GT, "split": "all"},
        "grid": grid, "lambda": lam, "n_games": int(len(p)),
        "mae_adjusted": loso_mae(p, ["d_adj"]), "mae_raw": loso_mae(p, ["d_raw"]),
        "mae_home_only": loso_mae(p, []), "mae_ridge": loso_mae(p, ["d_ridge"]),
    }

    rows = []
    for s in CFG.seasons:
        ps = plays[plays["season"] == s]
        for gt in ("all", "no_gt"):
            for split in ("all", "pass", "run"):
                v = variant(ps, gt, split)
                for w in sorted(v["week"].unique()):
                    r = ridge_ratings(v[v["week"] <= w], lam)
                    rows.append(r.assign(season=s, week=int(w), gt=gt, split=split, home=r.attrs["home"]))
    adj = pd.concat(rows, ignore_index=True)
    # Home effect for the search variant on the latest week of each complete season, for the method box.
    h = adj[(adj["gt"] == SEARCH_GT) & (adj["split"] == "all") & adj["season"].isin(search_seasons)]
    res["home_epa_per_play"] = float(h.loc[h.groupby("season")["week"].transform("max") == h["week"], "home"].mean())
    cols = ["season", "week", "team", "gt", "split", "off_plays", "def_plays", "off_raw", "def_raw", "off_adj", "def_adj"]
    adj = adj[cols].sort_values(["season", "week", "gt", "split", "team"])
    adj.to_csv(OUT / "adjusted_team_week.csv", index=False, float_format="%.6f")
    res["provenance"] = provenance(sorted((RAW / "nfl").glob("pbp_*.parquet")), seeds={})
    (OUT / "adjustment.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("grid", "provenance")}, indent=2))


if __name__ == "__main__":
    main()
