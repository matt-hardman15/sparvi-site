"""College football opponent-adjusted ratings (F2): the choice of ridge lambda, the validation
against public benchmarks, and the ratings as of every week.

1. Weekly backtest (seasons with complete data, 2014 on). For each season and week w >= MIN_WEEK,
   ratings fit on plays through week w predict each week-(w+1) game between FBS teams (bowls count
   as the week after the last regular-season week):
       home margin = a * home_site + b * (net_home - net_away),   net = offense - defense EPA/play
   a and b are fit leave-one-season-out, so no season's games inform their own predictions. Lambda
   is chosen on the ridge ratings' MAE. On the same games, the benchmarks are: raw EPA/play, home
   field alone, CFBD's pre-game Elo (same leave-one-season-out fit on the Elo gap), and the betting
   market's median opening and closing spreads across sportsbooks (used as given, in points).
   Lines are an accuracy benchmark only, reported in aggregate after the fact; no game-level edges.
2. WEPA test. CFBD's WEPA and SP+ exist only as season totals, so inside a season they'd see the
   future. They're compared where nothing leaks: each season's final numbers (ours fit on all of that
   season's plays, WEPA, SP+) predict the next season's weeks 1-4 games between FBS teams.
3. Ratings for every season x week x garbage-time variant x split, with the chosen lambda.
4. The power rating (Matt's decision, Nov 13 2026): our efficiency rating alone predicts worse than
   CFBD's Elo, but the two combined predict better than Elo (the backtest reports both, with a
   bootstrap interval). The power rating is that combination, in points against an average FBS team
   on a neutral field: b_net * net_ridge + b_elo * Elo, centered on the FBS average each week, with
   b fit on every complete season's backtest games. Elo as of a week is the team's post-game Elo
   after its latest game (pre-game Elo before its first).

Reads the football dbt database (stg_cfb_plays) and data/raw/cfb; writes
    output/adjusted_team_week.csv   season, week, team, gt, split, off/def adjusted and raw, plays
    output/power_team_week.csv      season, week, team, net (efficiency), elo, power, FBS rank
    output/metrics.json             lambda grid, the weekly backtest table, the WEPA test, the power model
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from football.common.adjust import ridge_ratings
from sparvi.config import RAW, ROOT, cfb
from sparvi.provenance import provenance

CFG = cfb()
OUT = Path(__file__).parent / "output"
DB = Path(os.environ.get("FOOTBALL_DB_PATH", ROOT / "football" / "dbt" / "football.duckdb"))
GRID = [12.5, 25, 50, 100, 200, 400, 800, 1600]
MIN_WEEK = 4           # first week whose ratings predict the next week (the plan's "from week 4 on")
SEARCH_GT = "no_gt"    # the tool's default view: garbage time removed, all plays
PRESEASON_WEEKS = 4    # the WEPA test predicts the next season's weeks 1..4


def load_plays() -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    df = con.sql("""select season, week, game_id, offense, defense, is_home::int as is_home, is_pass,
                           is_garbage, epa from stg_cfb_plays order by season, week, game_id, play_id""").df()
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
    g = pd.concat([pd.read_parquet(p) for p in sorted((RAW / "cfb").glob("games_*.parquet"))], ignore_index=True)
    last = g[g["seasonType"] == "regular"].groupby("season")["week"].max()
    g["week"] = np.where(g["seasonType"] == "postseason", g["season"].map(last) + 1, g["week"])
    g = g[g["completed"].fillna(False) & (g["homeClassification"] == "fbs") & (g["awayClassification"] == "fbs")]
    g = g.assign(home_site=(~g["neutralSite"].fillna(False)).astype(int), margin=g["homePoints"] - g["awayPoints"],
                 d_elo=g["homePregameElo"] - g["awayPregameElo"])
    lines = pd.concat([pd.read_parquet(p) for p in sorted((RAW / "cfb").glob("lines_*.parquet"))], ignore_index=True)
    med = lines.groupby("game_id")[["spread", "spreadOpen"]].median()
    g = g.join(-med.rename(columns={"spread": "line_close", "spreadOpen": "line_open"}), on="id")   # home margin
    return g.rename(columns={"id": "game_id", "homeTeam": "home_team", "awayTeam": "away_team"})[
        ["season", "week", "game_id", "home_team", "away_team", "home_site", "margin", "d_elo", "line_close", "line_open"]]


def net(r: pd.DataFrame, kind: str) -> pd.Series:
    r = r.set_index("team")
    return r[f"off_{kind}"] - r[f"def_{kind}"]


def predictions(plays: pd.DataFrame, g: pd.DataFrame, lam: float, seasons: list[int]) -> pd.DataFrame:
    rows = []
    for s in seasons:
        ps, gs = plays[plays["season"] == s], g[g["season"] == s]
        for w in range(MIN_WEEK, int(gs["week"].max())):
            nxt = gs[gs["week"] == w + 1]
            if nxt.empty:
                continue
            r = ridge_ratings(ps[ps["week"] <= w], lam)
            d = {f"d_{k}": nxt["home_team"].map(net(r, k)).to_numpy() - nxt["away_team"].map(net(r, k)).to_numpy()
                 for k in ("adj", "raw", "ridge")}
            rows.append(nxt.assign(**d))
    return pd.concat(rows, ignore_index=True).dropna(subset=["d_ridge", "d_raw"])


def loso_pred(p: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Margin predicted from home_site + cols, each season from a fit on the other seasons."""
    out = np.full(len(p), np.nan)
    for s in p["season"].unique():
        tr, te = p["season"] != s, p["season"] == s
        X = p.loc[tr, ["home_site", *cols]].to_numpy(float)
        coef, *_ = np.linalg.lstsq(X, p.loc[tr, "margin"].to_numpy(float), rcond=None)
        out[te.to_numpy()] = p.loc[te, ["home_site", *cols]].to_numpy(float) @ coef
    return out


def score(y: np.ndarray, pred: np.ndarray) -> dict:
    e = y - pred
    called = pred != 0
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "winner": float(np.mean(np.sign(pred[called]) == np.sign(y[called])))}


def paired_ci(y, a, b, seasons, n=2000) -> tuple[float, list[float]]:
    """Mean of |y-a| - |y-b| and its 90% interval, resampling games within each season."""
    rng = np.random.default_rng(0)
    diff = np.abs(y - a) - np.abs(y - b)
    groups = [np.flatnonzero(seasons == s) for s in np.unique(seasons)]
    boots = [diff[np.concatenate([gi[rng.integers(0, len(gi), len(gi))] for gi in groups])].mean() for _ in range(n)]
    return float(diff.mean()), [float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))]


def elo_by_week(seasons: list[int]) -> pd.DataFrame:
    """Each team's Elo as of the end of every week: post-game Elo of its latest game, else the
    pre-game Elo of its first game."""
    g = pd.concat([pd.read_parquet(p) for p in sorted((RAW / "cfb").glob("games_*.parquet"))], ignore_index=True)
    last = g[g["seasonType"] == "regular"].groupby("season")["week"].max()
    g["week"] = np.where(g["seasonType"] == "postseason", g["season"].map(last) + 1, g["week"])
    long = pd.concat([g[["season", "week", "homeTeam", "homePregameElo", "homePostgameElo", "completed"]].set_axis(
                          ["season", "week", "team", "pre", "post", "completed"], axis=1),
                      g[["season", "week", "awayTeam", "awayPregameElo", "awayPostgameElo", "completed"]].set_axis(
                          ["season", "week", "team", "pre", "post", "completed"], axis=1)], ignore_index=True)
    out = []
    for s in seasons:
        ls = long[long["season"] == s].sort_values("week")
        first = ls.dropna(subset=["pre"]).groupby("team")["pre"].first()
        for w in range(1, int(ls["week"].max()) + 1):
            done = ls[(ls["week"] <= w) & ls["completed"].fillna(False)].dropna(subset=["post"])
            elo = done.groupby("team")["post"].last().combine_first(first)
            out.append(pd.DataFrame({"season": s, "week": w, "team": elo.index, "elo": elo.to_numpy()}))
    return pd.concat(out, ignore_index=True)


def backtest(p: pd.DataFrame) -> dict:
    """Every method on the same games: those with Elo and both lines available."""
    c = p.dropna(subset=["d_elo", "line_close", "line_open"]).copy()
    y = c["margin"].to_numpy(float)
    preds = {"ours": loso_pred(c, ["d_ridge"]), "schedule_adjusted": loso_pred(c, ["d_adj"]), "raw_epa": loso_pred(c, ["d_raw"]),
             "home_only": loso_pred(c, []), "elo": loso_pred(c, ["d_elo"]), "power": loso_pred(c, ["d_ridge", "d_elo"]),
             "line_open": c["line_open"].to_numpy(float), "line_close": c["line_close"].to_numpy(float)}
    res = {k: score(y, v) for k, v in preds.items()}
    res["ours_closer_than_open_share"] = float(np.mean(np.abs(y - preds["ours"]) < np.abs(y - preds["line_open"])))
    res["ours_closer_than_close_share"] = float(np.mean(np.abs(y - preds["ours"]) < np.abs(y - preds["line_close"])))
    res["power_minus_elo_mae"], res["power_minus_elo_mae_ci90"] = paired_ci(y, preds["power"], preds["elo"], c["season"].to_numpy())
    res["n_games"] = int(len(c))
    res["n_by_season"] = {int(k): int(v) for k, v in c.groupby("season").size().items()}
    res["mae_by_season"] = {int(s): {k: float(np.mean(np.abs(y[m] - v[m]))) for k, v in preds.items()}
                            for s in sorted(c["season"].unique()) for m in [(c["season"] == s).to_numpy()]}
    return res


def wepa_test(plays: pd.DataFrame, g: pd.DataFrame, lam: float) -> dict:
    rows = []
    for s in range(CFG.first_season + 1, CFG.season):
        prev = s - 1
        r = ridge_ratings(variant(plays[plays["season"] == prev], SEARCH_GT, "all"), lam)
        ours = net(r, "ridge")
        w = pd.read_parquet(RAW / "cfb" / f"wepa_{prev}.parquet").set_index("team")
        wepa = w["epa"] - w["epa_allowed"]
        sp = pd.read_parquet(RAW / "cfb" / f"sp_{prev}.parquet").set_index("team")["rating"]
        nxt = g[(g["season"] == s) & (g["week"] <= PRESEASON_WEEKS)]
        rows.append(nxt.assign(d_ours=nxt["home_team"].map(ours).to_numpy() - nxt["away_team"].map(ours).to_numpy(),
                               d_wepa=nxt["home_team"].map(wepa).to_numpy() - nxt["away_team"].map(wepa).to_numpy(),
                               d_sp=nxt["home_team"].map(sp).to_numpy() - nxt["away_team"].map(sp).to_numpy()))
    p = pd.concat(rows, ignore_index=True).dropna(subset=["d_ours", "d_wepa", "d_sp", "d_elo"])
    y = p["margin"].to_numpy(float)
    preds = {"ours": loso_pred(p, ["d_ours"]), "wepa": loso_pred(p, ["d_wepa"]), "sp_plus": loso_pred(p, ["d_sp"]),
             "elo_pregame": loso_pred(p, ["d_elo"]), "home_only": loso_pred(p, []),
             "ours_plus_wepa": loso_pred(p, ["d_ours", "d_wepa"])}
    res = {k: score(y, v) for k, v in preds.items()}
    # Paired bootstrap (games resampled within season blocks): ours minus WEPA, MAE.
    rng = np.random.default_rng(0)
    diff = np.abs(y - preds["ours"]) - np.abs(y - preds["wepa"])
    seasons = p["season"].to_numpy()
    groups = [np.flatnonzero(seasons == s) for s in np.unique(seasons)]
    boots = []
    for _ in range(2000):
        idx = np.concatenate([gi[rng.integers(0, len(gi), len(gi))] for gi in groups])
        boots.append(diff[idx].mean())
    res["ours_minus_wepa_mae"] = float(diff.mean())
    res["ours_minus_wepa_mae_ci90"] = [float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))]
    res["n_games"] = int(len(p))
    res["weeks"] = PRESEASON_WEEKS
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plays = load_plays()
    g = games()
    ok = g.dropna(subset=["line_close"])
    sign = float(np.corrcoef(ok["line_close"], ok["margin"])[0, 1])
    assert sign > 0.3, f"closing line should agree in sign with the home margin (r = {sign:.2f}); check CFBD's spread convention"
    search = CFG.past_seasons
    base = variant(plays, SEARCH_GT, "all")

    grid = []
    for lam in GRID:
        p = predictions(base, g, lam, search)
        grid.append({"lambda": lam, "mae": float(np.mean(np.abs(p["margin"] - loso_pred(p, ["d_ridge"])))), "n_games": int(len(p))})
        print(f"lambda {lam:>5}: MAE {grid[-1]['mae']:.3f} on {grid[-1]['n_games']:,} games")
    lam = min(grid, key=lambda r: (round(r["mae"], 6), r["lambda"]))["lambda"]
    p = predictions(base, g, lam, search)
    res = {"seasons": search, "min_week": MIN_WEEK, "variant": {"gt": SEARCH_GT, "split": "all"}, "grid": grid,
           "lambda": lam, "line_sign_check_r": sign, "backtest": backtest(p), "wepa_test": wepa_test(plays, g, lam)}

    # Power model: the backtest's combination, fit on every complete season's games.
    c = p.dropna(subset=["d_elo"])
    X = c[["home_site", "d_ridge", "d_elo"]].to_numpy(float)
    b_home, b_net, b_elo = np.linalg.lstsq(X, c["margin"].to_numpy(float), rcond=None)[0]
    res["power_model"] = {"home_points": float(b_home), "points_per_net_epa": float(b_net), "points_per_elo": float(b_elo),
                          "n_games": int(len(c)), "variant": {"gt": SEARCH_GT, "split": "all"}}

    rows = []
    for s in CFG.seasons:
        ps = plays[plays["season"] == s]
        for gt in ("all", "no_gt"):
            for split in ("all", "pass", "run"):
                v = variant(ps, gt, split)
                for w in sorted(v["week"].unique()):
                    r = ridge_ratings(v[v["week"] <= w], lam)
                    rows.append(r.assign(season=s, week=int(w), gt=gt, split=split))
    adj = pd.concat(rows, ignore_index=True)
    fbs = pd.concat([pd.read_parquet(f, columns=["school"]).assign(season=int(f.stem.split("_")[1]))
                     for f in sorted((RAW / "cfb").glob("teams_*.parquet"))]).rename(columns={"school": "team"})
    base_rows = adj[(adj["gt"] == SEARCH_GT) & (adj["split"] == "all")].assign(net=lambda d: d["off_ridge"] - d["def_ridge"])
    pw = base_rows[["season", "week", "team", "net"]].merge(fbs, on=["season", "team"]).merge(
        elo_by_week(CFG.seasons), on=["season", "week", "team"], how="left")
    raw_power = b_net * pw["net"] + b_elo * pw["elo"]
    pw["power"] = raw_power - raw_power.groupby([pw["season"], pw["week"]]).transform("mean")
    pw["rank"] = pw.groupby(["season", "week"])["power"].rank(ascending=False, method="min")
    pw.sort_values(["season", "week", "rank", "team"]).to_csv(OUT / "power_team_week.csv", index=False, float_format="%.6f")
    cols = ["season", "week", "team", "gt", "split", "off_plays", "def_plays", "off_raw", "def_raw", "off_adj", "def_adj"]
    adj[cols].sort_values(["season", "week", "gt", "split", "team"]).to_csv(OUT / "adjusted_team_week.csv", index=False, float_format="%.6f")
    res["provenance"] = provenance(sorted((RAW / "cfb").glob("games_*.parquet")), seeds={"bootstrap": 0},
                                   date_expr="startDate::timestamp")
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    show = {k: v for k, v in res.items() if k not in ("grid", "provenance")}
    show["backtest"] = {k: v for k, v in res["backtest"].items() if k not in ("mae_by_season", "n_by_season")}
    print(json.dumps(show, indent=2))


if __name__ == "__main__":
    main()
