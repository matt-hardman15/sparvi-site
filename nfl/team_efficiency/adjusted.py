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
2. A second penalty for the schedule correction alone (NFL-SCHED). The margin search fits a free slope on
   the rating gap, which absorbs uniform shrinkage, so it says little about how large the opponent
   strengths should be, and its heavy penalty (3,200) left a correction of at most 0.008 EPA/play.
   The correction's penalty is chosen on its own question: how well do the coefficients predict a
   held-out game's EPA/play (leave-one-game-out within each complete season)? The same test run
   forward (fit on earlier weeks only) and a simulation with known strengths are reported with it.
3. Ratings for every season x week x garbage-time variant x split (all plays, passes, runs) on
   season-to-date plays: off_ridge/def_ridge at the forecast lambda, off_adj/def_adj corrected with
   the schedule lambda.

Reads the dbt database (stg_nfl_plays) and the schedules; writes
    output/adjusted_team_week.csv   season, week, team, gt, split, off/def adjusted and raw, plays
    output/adjustment.json          lambda grid with MAE, chosen lambda, baselines, n games; the schedule
                                    penalty's search, the size of the correction, and the simulation
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from football.common.adjust import cell_cv, pool_cv, ridge_ratings
from football.common.simulate import recovery
from sparvi.config import RAW, ROOT, nfl
from sparvi.provenance import provenance

CFG = nfl()
OUT = Path(__file__).parent / "output"
DB = Path(os.environ.get("FOOTBALL_DB_PATH", ROOT / "football" / "dbt" / "football.duckdb"))
GRID = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]   # past the minimum on both sides (NFL-LAMBDA)
# Penalties tried for the schedule correction (held-out-game EPA/play), and the (offense, defense) pairs reported as
# an alternative that is not used: defenses differ less than offenses, so a pair fits slightly better (NFL-SCHED).
SCHEDULE_GRID = [50, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 3200, 6400]
PAIR_GRID = [(o, d) for o in (200, 300, 400, 600) for d in (400, 600, 800, 1200, 1600)]
SIM_SEASONS = 40      # simulated seasons with known strengths (football/common/simulate.py)
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


def predictions(plays: pd.DataFrame, g: pd.DataFrame, lam: float, seasons: list[int],
                lam_schedule=None) -> pd.DataFrame:
    """Next-week games with the rating gap from data through the prior week (adjusted and raw)."""
    rows = []
    for s in seasons:
        ps, gs = plays[plays["season"] == s], g[g["season"] == s]
        for w in range(MIN_WEEK, int(gs["week"].max())):
            nxt = gs[gs["week"] == w + 1]
            if nxt.empty:
                continue
            r = ridge_ratings(ps[ps["week"] <= w], lam, lam_schedule=lam_schedule)
            d = {f"d_{kind}": nxt["home_team"].map(net(r, kind)).to_numpy() - nxt["away_team"].map(net(r, kind)).to_numpy()
                 for kind in ("adj", "raw", "ridge")}
            rows.append(nxt.assign(**d))
    return pd.concat(rows, ignore_index=True)


def loso_abs_err(p: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Per-game absolute error of margin ~ home_site + cols, each season predicted from the other seasons
    (in `p`'s season order)."""
    err = []
    for s in sorted(p["season"].unique()):
        tr, te = p[p["season"] != s], p[p["season"] == s]
        X = tr[["home_site", *cols]].to_numpy(float)
        coef, *_ = np.linalg.lstsq(X, tr["margin"].to_numpy(float), rcond=None)
        err.append(np.abs(te["margin"].to_numpy(float) - te[["home_site", *cols]].to_numpy(float) @ coef))
    return np.concatenate(err)


def loso_mae(p: pd.DataFrame, cols: list[str]) -> float:
    """MAE of margin ~ home_site + cols, each season predicted from a fit on the other seasons."""
    return float(np.mean(loso_abs_err(p, cols)))


def mae_gap(p: pd.DataFrame, a: str, b: str, draws: int = 2000, seed: int = 7) -> dict:
    """MAE with rating gap `a` minus MAE with `b` on the same games, with a 90% interval (games resampled)."""
    d = loso_abs_err(p, [a]) - loso_abs_err(p, [b])
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(draws, len(d)))].mean(axis=1)
    return {"diff": float(d.mean()), "lo": float(np.quantile(boot, 0.05)), "hi": float(np.quantile(boot, 0.95)),
            "draws": draws, "seed": seed}


def schedule_search(base: pd.DataFrame, seasons: list[int], lam_forecast: float) -> dict:
    """The schedule correction's penalty: the grid value whose coefficients best predict a held-out game's
    EPA/play (regular season, leave-one-game-out, pooled over complete seasons)."""
    by_season = [base[base["season"] == s] for s in seasons]
    logo = pool_cv([cell_cv(ps, SCHEDULE_GRID) for ps in by_season])
    fwd = pool_cv([cell_cv(ps, SCHEDULE_GRID, forward_from=MIN_WEEK) for ps in by_season])
    grid = [{"lambda": a["lambda"], "skill": a["skill"], "mse": a["mse"], "skill_forward": b["skill"]}
            for a, b in zip(logo, fwd, strict=True)]
    for r in grid:
        print(f"schedule lambda {r['lambda']:>5}: held-out game skill {r['skill']:.4f}, forward {r['skill_forward']:.4f}")
    best = max(grid, key=lambda r: (round(r["skill"], 6), -r["lambda"]))
    pairs = pool_cv([cell_cv(ps, PAIR_GRID) for ps in by_season])
    pair = max(pairs, key=lambda r: r["skill"])
    at = {r["lambda"]: r for r in grid}
    return {
        "objective": "Held-out game EPA/play for each side, predicted as mean + offense + defense + home from the"
                     " season's other games; skill = 1 - squared error / squared error of the league mean",
        "grid": grid, "lambda": best["lambda"], "skill": best["skill"], "n_plays": logo[0]["plays"],
        "lambda_forward": max(grid, key=lambda r: r["skill_forward"])["lambda"],
        "skill_at_forecast_lambda": at[lam_forecast]["skill"] if lam_forecast in at else None,
        "pair_not_used": {"lambda_off": pair["lambda"][0], "lambda_def": pair["lambda"][1], "skill": pair["skill"]},
    }


def correction_size(r: pd.DataFrame) -> dict:
    """How far the correction moves teams, in EPA/play, next to how far teams differ."""
    off, deff = r["off_adj"] - r["off_raw"], r["def_adj"] - r["def_raw"]
    return {"off_max": float(off.abs().max()), "off_sd": float(off.std(ddof=0)), "def_max": float(deff.abs().max()),
            "def_sd": float(deff.std(ddof=0)), "off_raw_sd": float(r["off_raw"].std(ddof=0)),
            "def_raw_sd": float(r["def_raw"].std(ddof=0))}


def rank_changes(r: pd.DataFrame) -> dict:
    """Ranks by raw and by adjusted EPA/play (offense: higher is better; defense: lower is better)."""
    out = {}
    for side, asc in (("off", False), ("def", True)):
        raw = r[f"{side}_raw"].rank(ascending=asc, method="min")
        adj = r[f"{side}_adj"].rank(ascending=asc, method="min")
        move = (raw - adj).astype(int)                                  # > 0: the adjustment moves the team up
        top = move.abs().sort_values(ascending=False, kind="stable").index[:5]
        out[side] = {"moved": int((move != 0).sum()), "max_move": int(move.abs().max()),
                     "mean_abs_move": float(move.abs().mean()),
                     "largest": [{"team": r.loc[i, "team"], "raw_rank": int(raw[i]), "adj_rank": int(adj[i])} for i in top]}
    return out


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
    sched = schedule_search(base, search_seasons, lam)
    lam_s = sched["lambda"]
    p = predictions(base, g, lam, search_seasons, lam_s)
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
                    r = ridge_ratings(v[v["week"] <= w], lam, lam_schedule=lam_s)
                    rows.append(r.assign(season=s, week=int(w), gt=gt, split=split, home=r.attrs["home"]))
    adj = pd.concat(rows, ignore_index=True)
    # Home effect for the search variant on the latest week of each complete season, for the method box.
    h = adj[(adj["gt"] == SEARCH_GT) & (adj["split"] == "all") & adj["season"].isin(search_seasons)]
    res["home_epa_per_play"] = float(h.loc[h.groupby("season")["week"].transform("max") == h["week"], "home"].mean())
    # The correction's size in the current season's latest week (the tool's default view), at the schedule penalty
    # and, for comparison, at the forecast penalty it replaced; and what it does to the rankings.
    cur = variant(plays[plays["season"] == CFG.season], SEARCH_GT, "all")
    if cur.empty:        # before week 1: the last complete season
        cur = variant(plays[plays["season"] == search_seasons[-1]], SEARCH_GT, "all")
    now = ridge_ratings(cur, lam, lam_schedule=lam_s)
    res["lambda_schedule"] = lam_s
    res["schedule_search"] = sched
    res["mae_adjusted_minus_raw"] = mae_gap(p, "d_adj", "d_raw")
    res["mae_adjusted_forecast_lambda"] = loso_mae(predictions(base, g, lam, search_seasons), ["d_adj"])
    res["correction"] = {"season": int(cur["season"].iloc[0]), "week": int(cur["week"].max()),
                         **correction_size(now), "at_forecast_lambda": correction_size(ridge_ratings(cur, lam)),
                         "ranks": rank_changes(now)}
    # Known-truth check: NFL-sized simulated seasons with unequal schedules. Share of each team's true schedule
    # effect the correction recovers, at a full season and at the current week.
    weeks = res["correction"]["week"]
    res["simulation"] = {
        "full_season": recovery(lam, lam_s, seasons=SIM_SEASONS, seed=7),
        "through_week": {"week": weeks, **recovery(lam, lam_s, seasons=SIM_SEASONS, seed=7, through_week=weeks)},
        "full_season_at_forecast_lambda": recovery(lam, None, seasons=SIM_SEASONS, seed=7),
    }
    cols = ["season", "week", "team", "gt", "split", "off_plays", "def_plays", "off_raw", "def_raw", "off_adj", "def_adj"]
    adj = adj[cols].sort_values(["season", "week", "gt", "split", "team"])
    adj.to_csv(OUT / "adjusted_team_week.csv", index=False, float_format="%.6f")
    res["provenance"] = provenance(sorted((RAW / "nfl").glob("pbp_*.parquet")), seeds={})
    (OUT / "adjustment.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("grid", "provenance", "schedule_search")}, indent=2))


if __name__ == "__main__":
    main()
