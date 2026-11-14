"""What a loss costs: how far a ranked team falls in the College Football Playoff committee's next ranking
after a loss, against how far the power rating drops it, by margin of defeat.

Every pair of consecutive regular-season CFP rankings, completed seasons only (first season through the
season before the current one). A team counts when it was ranked in the first release and played exactly
one FBS or FCS game before the next (byes and double weeks are left out). CFBD labels a ranking by the
week it applies to, so release `a` is followed by the games of weeks a .. b-1; the power rating's rank is
taken through week a-1 (before those games) and through week b-1 (after them).

Main sample: teams ranked 1-12 that lost (a top-12 team almost never falls out of the 25, so its new rank
is observed). Teams ranked 13-25 are summarized separately with the share that fell out.

Writes cfb/loss_cost/output/results.json. Deterministic (fixed bootstrap seed). ~10 seconds.

    python cfb/loss_cost/analysis.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sparvi.config import RAW, ROOT, cfb

CFG = cfb()
OUT = ROOT / "cfb" / "loss_cost" / "output"
SEASONS = [s for s in CFG.seasons if s < CFG.season]
TOP = 12                        # main sample: ranked 1..TOP before the loss
BANDS = [(1, 4), (5, 8), (9, 12), (13, 25)]
MARGINS = [(1, 3), (4, 7), (8, 14), (15, 21), (22, 99)]
BOOT, SEED = 2000, 7
OUT_RANK = 26                   # a team that falls out of the 25 counts as 26 (a lower bound on its fall)


def rows() -> pd.DataFrame:
    ranks = pd.concat([pd.read_parquet(RAW / "cfb" / f"rankings_{s}.parquet") for s in SEASONS], ignore_index=True)
    cfp = ranks[(ranks["poll"] == "Playoff Committee Rankings") & (ranks["seasonType"] == "regular")]
    games = pd.concat([pd.read_parquet(RAW / "cfb" / f"games_{s}.parquet") for s in SEASONS], ignore_index=True)
    games = games[games["completed"].astype(bool) & (games["seasonType"] == "regular")]
    power = pd.read_csv(ROOT / "cfb" / "adjusted_ratings" / "output" / "power_team_week.csv").set_index(["season", "week", "team"])["rank"]
    out = []
    for season, cs in cfp.groupby("season"):
        weeks = sorted(cs["week"].unique())
        for a, b in zip(weeks, weeks[1:], strict=False):
            ra = cs[cs["week"] == a].set_index("school")["rank"]
            rb = cs[cs["week"] == b].set_index("school")["rank"]
            gg = games[(games["season"] == season) & games["week"].between(a, b - 1)]
            for team, r0 in ra.items():
                m = gg[(gg["homeTeam"] == team) | (gg["awayTeam"] == team)]
                if len(m) != 1:
                    continue
                g = m.iloc[0]
                home = g["homeTeam"] == team
                pf, pa = (g["homePoints"], g["awayPoints"]) if home else (g["awayPoints"], g["homePoints"])
                opp = g["awayTeam"] if home else g["homeTeam"]
                p0, p1 = power.get((season, a - 1, team)), power.get((season, b - 1, team))
                if p0 is None or p1 is None or pd.isna(pf) or pd.isna(pa) or pf == pa:
                    continue
                out.append({"season": int(season), "week": int(a), "team": team, "opp": opp, "rank0": int(r0),
                            "rank1": None if team not in rb.index else int(rb[team]), "opp_ranked": opp in ra.index,
                            "margin": int(pf - pa), "power0": int(p0), "power1": int(p1)})
    d = pd.DataFrame(out)
    d["fell_out"] = d["rank1"].isna()
    d["drop"] = d["rank1"].fillna(OUT_RANK) - d["rank0"]
    d["power_drop"] = d["power1"] - d["power0"]
    d["lost_by"] = -d["margin"]
    return d


def slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[0])


def main() -> None:
    d = rows()
    loss = d[d["margin"] < 0]
    top = loss[loss["rank0"] <= TOP].reset_index(drop=True)
    x, yc, yp = top["lost_by"].to_numpy(float), top["drop"].to_numpy(float), top["power_drop"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(BOOT):
        i = rng.integers(0, len(top), len(top))
        sc, sp = slope(x[i], yc[i]), slope(x[i], yp[i])
        close, big = x[i] <= 7, x[i] >= 15
        bs.append((sc, sp, sp - sc, yc[i][close].mean() - yc[i][big].mean(), yp[i][close].mean() - yp[i][big].mean()))
    bs = np.array(bs)
    ci = lambda k: [round(float(v), 3) for v in np.percentile(bs[:, k], [5, 95])]  # noqa: E731
    close, big = top[top["lost_by"] <= 7], top[top["lost_by"] >= 15]
    by_margin = [{"lo": lo, "hi": hi, "n": int(len(z)), "drop": round(float(z["drop"].mean()), 2),
                  "power_drop": round(float(z["power_drop"].mean()), 2)}
                 for lo, hi in MARGINS for z in [top[top["lost_by"].between(lo, hi)]]]
    by_rank = [{"lo": lo, "hi": hi, "n": int(len(z)), "drop": round(float(z["drop"].mean()), 2),
                "drop_median": float(z["drop"].median()), "fell_out": round(float(z["fell_out"].mean()), 3),
                "power_drop": round(float(z["power_drop"].mean()), 2)}
               for lo, hi in BANDS for z in [loss[loss["rank0"].between(lo, hi)]]]
    wins = d[(d["margin"] > 0) & (d["rank0"] <= TOP)]
    opp = {k: {"n": int(len(z)), "drop": round(float(z["drop"].mean()), 2), "power_drop": round(float(z["power_drop"].mean()), 2)}
           for k, z in (("ranked", top[top["opp_ranked"]]), ("unranked", top[~top["opp_ranked"]]))}
    points = [{"season": int(r.season), "week": int(r.week), "team": r.team, "opp": r.opp, "rank0": int(r.rank0),
               "rank1": None if pd.isna(r.rank1) else int(r.rank1), "lost_by": int(r.lost_by), "drop": int(r.drop),
               "power0": int(r.power0), "power1": int(r.power1), "power_drop": int(r.power_drop)}
              for r in top.sort_values(["season", "week", "rank0"]).itertuples()]
    res = {
        "seasons": [SEASONS[0], SEASONS[-1]], "top": TOP, "boot": BOOT, "seed": SEED, "out_rank": OUT_RANK,
        "n_pairs": int(len(d)), "n_losses": int(len(loss)), "n_top_losses": int(len(top)),
        "slope_committee": round(slope(x, yc), 3), "slope_committee_ci90": ci(0),
        "slope_power": round(slope(x, yp), 3), "slope_power_ci90": ci(1), "slope_diff_ci90": ci(2),
        "close": {"n": int(len(close)), "drop": round(float(close["drop"].mean()), 2), "power_drop": round(float(close["power_drop"].mean()), 2)},
        "big": {"n": int(len(big)), "drop": round(float(big["drop"].mean()), 2), "power_drop": round(float(big["power_drop"].mean()), 2)},
        "close_minus_big_committee_ci90": ci(3), "close_minus_big_power_ci90": ci(4),
        "win": {"n": int(len(wins)), "drop": round(float(wins["drop"].mean()), 2), "power_drop": round(float(wins["power_drop"].mean()), 2)},
        "by_margin": by_margin, "by_rank": by_rank, "by_opp": opp, "points": points,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(top)} top-{TOP} losses, {SEASONS[0]}-{SEASONS[-1]}: committee {res['slope_committee']} spots/point "
          f"{res['slope_committee_ci90']}, power {res['slope_power']} {res['slope_power_ci90']}")
    print("close", res["close"], "big", res["big"], "win", res["win"])
    for r in by_margin:
        print(r)
    for r in by_rank:
        print(r)


if __name__ == "__main__":
    main()
