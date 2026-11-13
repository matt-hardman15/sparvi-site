"""Ranked vs. actually good (F2 article): how well the AP poll and the power rating, each as of the
end of the previous week, pick the winners of games between FBS teams, and who is right when they
disagree. Unranked teams count as rank 26 (below every ranked team), so a game counts for the poll
only if at least one team is ranked. Seasons 2014 to last season; weeks 5 on, bowls included.

Reads power_team_week.csv (adjusted.py) and data/raw/cfb games and rankings; writes
    output/polls.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sparvi.config import RAW, cfb

CFG = cfb()
OUT = Path(__file__).parent / "output"
MIN_WEEK = 5
UNRANKED = 26


def wilson(p: float, n: int, z: float = 1.645) -> list[float]:
    """90% Wilson interval for a proportion."""
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def main() -> None:
    power = pd.read_csv(OUT / "power_team_week.csv")
    g = pd.concat([pd.read_parquet(p) for p in sorted((RAW / "cfb").glob("games_*.parquet"))], ignore_index=True)
    last = g[g["seasonType"] == "regular"].groupby("season")["week"].max()
    g["week"] = np.where(g["seasonType"] == "postseason", g["season"].map(last) + 1, g["week"])
    g = g[g["completed"].fillna(False) & (g["homeClassification"] == "fbs") & (g["awayClassification"] == "fbs")
          & g["season"].isin(CFG.past_seasons) & (g["week"] >= MIN_WEEK) & (g["homePoints"] != g["awayPoints"])]
    r = pd.concat([pd.read_parquet(p) for p in sorted((RAW / "cfb").glob("rankings_*.parquet"))], ignore_index=True)
    ap = r[(r["poll"] == "AP Top 25") & (r["seasonType"] == "regular")][["season", "week", "school", "rank"]]

    def asof(df, key, week_shift):
        # rating as of the end of week w-1 for a week-w game
        return df.assign(week=df["week"] + week_shift).rename(columns={key: "v"})

    pw = asof(power[["season", "week", "team", "power"]], "power", 1).rename(columns={"team": "school"})
    # AP polls are labeled by the week they apply to, which is already "as of the end of the week before".
    rows = []
    for side in ("home", "away"):
        t = g[["id", "season", "week", f"{side}Team"]].rename(columns={f"{side}Team": "school"})
        t = t.merge(pw, on=["season", "week", "school"], how="left").rename(columns={"v": f"{side}_power"})
        t = t.merge(ap.rename(columns={"rank": f"{side}_ap"}), on=["season", "week", "school"], how="left")
        rows.append(t.drop(columns="school").set_index(["id", "season", "week"]))
    d = g[["id", "season", "week", "homePoints", "awayPoints"]].set_index(["id", "season", "week"]).join(rows[0]).join(rows[1]).reset_index()
    d = d.dropna(subset=["home_power", "away_power"])
    d = d[d["home_ap"].notna() | d["away_ap"].notna()]
    d["home_ap"] = d["home_ap"].fillna(UNRANKED)
    d["away_ap"] = d["away_ap"].fillna(UNRANKED)
    home_won = d["homePoints"] > d["awayPoints"]
    ap_pick = np.where(d["home_ap"] == d["away_ap"], np.nan, (d["home_ap"] < d["away_ap"]).astype(float))
    pw_pick = (d["home_power"] > d["away_power"]).astype(float)
    ok = ~np.isnan(ap_pick)
    ap_right = (ap_pick[ok] == home_won[ok]).mean()
    pw_right = (pw_pick[ok] == home_won[ok]).mean()
    disagree = ok & (ap_pick != pw_pick)
    res = {
        "seasons": [int(d["season"].min()), int(d["season"].max())], "min_week": MIN_WEEK,
        "n_games": int(ok.sum()), "ap_right": float(ap_right), "power_right": float(pw_right),
        "n_disagree": int(disagree.sum()),
        "power_right_when_disagree": float((pw_pick[disagree] == home_won[disagree]).mean()),
        "n_both_ranked": int((ok & (d["home_ap"] < UNRANKED) & (d["away_ap"] < UNRANKED)).sum()),
    }
    res["power_right_when_disagree_ci90"] = wilson(res["power_right_when_disagree"], res["n_disagree"])
    # Note: the power rating's two weights (efficiency vs. Elo) were fit on these seasons' games
    # (adjusted.py); its leave-one-season-out backtest is in metrics.json.
    (OUT / "polls.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
