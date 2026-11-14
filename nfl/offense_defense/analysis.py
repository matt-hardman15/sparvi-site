"""Offense or defense: when an offense meets a defense, which side's season-to-date numbers tell you more
about how that matchup goes?

Every regular-season game from week MIN_WEEK on, completed seasons (first season through the season
before the current one), garbage time removed. Each game gives two matchups (each offense against the other
defense). The outcome is the offense's EPA/play in that game; the predictors are the offense's
opponent-adjusted EPA/play and the defense's opponent-adjusted EPA/play allowed, both season to date
through the previous week (nothing from the game itself or later). For all plays, passes, and runs:

- correlation of each side with the game result, and their ratio;
- a linear fit with both sides (standardized), and the same fit with an offense x defense interaction
  (does a strength-vs-strength matchup behave differently from the sum of its parts?);
- a 3 x 3 grid: offense tercile x defense tercile, average game EPA/play.

Intervals: 2,000 bootstrap resamples of games (both matchups in a game move together).
Writes nfl/offense_defense/output/results.json. Deterministic. ~20 seconds.

    python nfl/offense_defense/analysis.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from sparvi.config import ROOT, nfl

CFG = nfl()
OUT = ROOT / "nfl" / "offense_defense" / "output"
DB = Path(os.environ.get("FOOTBALL_DB_PATH", ROOT / "football" / "dbt" / "football.duckdb"))
SEASONS = [s for s in CFG.seasons if s < CFG.season]
MIN_WEEK = 6          # ratings through week 5 or later: at least four or five games behind them
BOOT, SEED = 2000, 5
SPLITS = {"all": "epa_play", "pass": "pass_epa_play", "run": "run_epa_play"}


def matchups() -> pd.DataFrame:
    adj = pd.read_csv(ROOT / "nfl" / "team_efficiency" / "output" / "adjusted_team_week.csv")
    adj = adj[(adj["gt"] == "no_gt") & adj["season"].isin(SEASONS)]
    con = duckdb.connect(str(DB), read_only=True)
    g = con.sql(f"""select season, week, game_id, team, opponent, {', '.join(SPLITS.values())}
                    from nfl_team_games where side = 'off' and gt = 'no_gt'
                      and season between {SEASONS[0]} and {SEASONS[-1]} and week >= {MIN_WEEK}""").df()
    con.close()
    g = g.astype({"season": "int64", "week": "int64"})
    out = []
    for split, col in SPLITS.items():
        a = adj[adj["split"] == split][["season", "week", "team", "off_adj", "def_adj"]].sort_values("week")
        x = g[["season", "week", "game_id", "team", "opponent", col]].rename(columns={col: "y"}).dropna()
        x = x.assign(rw=x["week"] - 1).sort_values("rw")
        # Latest rating before the game (a bye leaves no row for that week).
        x = pd.merge_asof(x, a.rename(columns={"week": "rw", "off_adj": "off"})[["season", "rw", "team", "off"]],
                          on="rw", by=["season", "team"], allow_exact_matches=True)
        x = pd.merge_asof(x.sort_values("rw"), a.rename(columns={"week": "rw", "team": "opponent", "def_adj": "dfn"})[["season", "rw", "opponent", "dfn"]],
                          on="rw", by=["season", "opponent"], allow_exact_matches=True)
        out.append(x.dropna(subset=["off", "dfn"]).assign(split=split))
    return pd.concat(out, ignore_index=True).sort_values(["split", "season", "week", "game_id", "team"]).reset_index(drop=True)


def stats(x: pd.DataFrame) -> dict:
    off, dfn, y = x["off"].to_numpy(), x["dfn"].to_numpy(), x["y"].to_numpy()
    z = lambda v: (v - v.mean()) / v.std()  # noqa: E731
    X = np.column_stack([np.ones(len(y)), z(off), z(dfn)])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    bi = np.linalg.lstsq(np.column_stack([X, z(off) * z(dfn)]), y, rcond=None)[0]
    r_off, r_def = np.corrcoef(off, y)[0, 1], np.corrcoef(dfn, y)[0, 1]
    return {"r_off": r_off, "r_def": r_def, "ratio": r_off / r_def, "b_off": b[1], "b_def": b[2], "b_int": bi[3]}


def main() -> None:
    d = matchups()
    rng = np.random.default_rng(SEED)
    res = {"seasons": [SEASONS[0], SEASONS[-1]], "min_week": MIN_WEEK, "boot": BOOT, "seed": SEED, "splits": {}}
    for split, x in d.groupby("split", sort=False):
        x = x.reset_index(drop=True)
        point = stats(x)
        gids = x["game_id"].unique()
        idx = {gid: np.flatnonzero(x["game_id"].to_numpy() == gid) for gid in gids}
        bs = []
        for _ in range(BOOT):
            pick = rng.choice(gids, len(gids))
            bs.append(stats(x.iloc[np.concatenate([idx[gid] for gid in pick])]))
        bs = pd.DataFrame(bs)
        ci = {k: [round(float(v), 4) for v in np.percentile(bs[k], [5, 95])] for k in bs.columns}
        # 3 x 3 grid: offense tercile (rows) x defense tercile (columns), average game EPA/play.
        qo = pd.qcut(x["off"], 3, labels=["weak", "middle", "strong"])
        qd = pd.qcut(-x["dfn"], 3, labels=["weak", "middle", "strong"])      # strong defense = low EPA allowed
        grid = x.groupby([qo, qd], observed=True)["y"].agg(["mean", "size"]).reset_index()
        res["splits"][split] = {
            "n": int(len(x)), "n_games": int(len(gids)),
            **{k: round(float(v), 4) for k, v in point.items()}, "ci90": ci,
            "sd_off": round(float(x["off"].std()), 4), "sd_def": round(float(x["dfn"].std()), 4),
            "mean_y": round(float(x["y"].mean()), 4),
            "grid": [{"off": str(r["off"]), "def": str(r["dfn"]), "epa": round(float(r["mean"]), 3), "n": int(r["size"])}
                     for _, r in grid.rename(columns={"off": "off", "dfn": "dfn"}).iterrows()],
        }
        print(split, len(x), {k: round(float(v), 3) for k, v in point.items()}, ci["ratio"], ci["b_int"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
