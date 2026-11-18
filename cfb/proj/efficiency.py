"""CFB-PROJ P3: efficiency. Per-opportunity production for season t as of May 1 of t (PLAN-fit, CFB-PROJ).
DEVELOPMENT ONLY: evaluated on seasons 2016-2020, leave one season out; the test seasons (2021-2025) are never scored
here, and 2026 is the vault.

Player efficiency (the player's own, wherever he played; level from his team's classification that season):
  QB  passing yards per dropback (sacks count with their negative yards)
  RB  rushing yards per carry
  Projection = position-level mean + w * (his last two seasons' rate - that mean) + level shift for movers, with
  w = n / (n + K) (n = his opportunities, t-2 counted at half weight) and K chosen on the development seasons (leave one
  season out). Players with no history get the mean of first-year players at the position.
Team context (for assembly: receivers' yards come from the team's passing, RB carries from its run plays):
  team passing yards per dropback, and the share of run plays carried by RBs; each from the team's last two seasons,
  pulled toward the league mean by a slope fit on the other development seasons.
Baselines: carry (last season's value; the position mean when missing) and mean (position mean).

    python cfb/proj/efficiency.py            # --cached reuses the private season tables
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from backtest.asof import Loader
from cfb.fit import data as D
from cfb.fit import labels as LB
from cfb.profile import engine as G
from sparvi.config import RAW, ROOT

OUT = ROOT / "cfb" / "proj" / "output"
CACHE_P, CACHE_T = RAW / "cfb" / "proj_eff_players.parquet", RAW / "cfb" / "proj_eff_teams.parquet"   # private
SEASONS = range(2014, 2021)
DEV = range(2016, 2021)
K_GRID = (25, 50, 100, 150, 200, 300, 400, 600, 800)
MIN_EVAL = {"QB": 50, "RB": 30}                        # the backup minimum (labels.BACKUP_MIN)
LEVEL = {"FCS": 0, "G5": 1, "Power": 2}
SEED, N_BOOT = 7, 1000


def seasons() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Player-seasons (QB dropbacks / RB carries: n, yards, team, level) and team-seasons (dropbacks, pass yards,
    run plays, RB carries)."""
    L = Loader()
    pr, tm = [], []
    for s in SEASONS:
        ev, pos = D.season_events(L, s, pd.Timestamp(f"{s + 1}-02-01"))
        conf = dict(pd.read_parquet(RAW / "cfb" / f"teams_{s}.parquet")[["school", "conference"]].values)
        grp = ev["player_id"].map(pos).map(G.POSITION_GROUP)
        comp = (ev["opp"] == "target") & (ev["complete"] == 1)
        for g, sel in (("QB", (ev["opp"] == "dropback") & (grp == "QB")), ("RB", (ev["opp"] == "carry") & (grp == "RB")),
                       ("REC", comp & grp.isin(["WR", "TE", "RB"]))):        # REC: receptions and receiving yards
            r = ev[sel]
            a = r.groupby("player_id").agg(n=("yards", "size"), yards=("yards", "sum"),
                                           team=("offense", lambda x: x.mode().iloc[0])).reset_index()
            a["level"] = [LEVEL[LB.tier(t, s, conf)] for t in a["team"]]
            pr.append(a.assign(season=s, group=g))
        db = ev[ev["opp"] == "dropback"].drop_duplicates("play_id")
        runs = ev[ev["kind"] == "run"].drop_duplicates("play_id")
        rb = ev[(ev["opp"] == "carry") & (grp == "RB")].drop_duplicates("play_id")
        rec = ev[comp].drop_duplicates("play_id")
        t = pd.DataFrame({"dropbacks": db.groupby("offense").size(), "pass_yards": db.groupby("offense")["yards"].sum(),
                          "runs": runs.groupby("offense").size(), "rb_carries": rb.groupby("offense").size(),
                          "rb_yards": rb.groupby("offense")["yards"].sum(), "rec_yards": rec.groupby("offense")["yards"].sum()}).fillna(0)
        t = t[t.index.isin(conf)]
        tm.append(t.rename_axis("team").reset_index().assign(season=s))
        print(f"  efficiency {s}", flush=True)
    p, t = pd.concat(pr, ignore_index=True), pd.concat(tm, ignore_index=True)
    p["rate"] = p["yards"] / p["n"]
    t["ypd"] = t["pass_yards"] / t["dropbacks"].clip(lower=1)
    t["rb_frac"] = t["rb_carries"] / t["runs"].clip(lower=1)
    t["rec_ypd"] = t["rec_yards"] / t["dropbacks"].clip(lower=1)
    return p, t


def player_rows(p: pd.DataFrame) -> pd.DataFrame:
    """Season-t outcome rows (players at or above MIN_EVAL in t) with their history."""
    k = p.set_index(["season", "player_id", "group"])
    rows = []
    for t in DEV:
        for g in ("QB", "RB"):
            cur = p[(p["season"] == t) & (p["group"] == g) & (p["n"] >= MIN_EVAL[g])]
            for r in cur.itertuples():
                h1 = k.loc[(t - 1, r.player_id, g)] if (t - 1, r.player_id, g) in k.index else None
                h2 = k.loc[(t - 2, r.player_id, g)] if (t - 2, r.player_id, g) in k.index else None
                rows.append({"season": t, "player_id": r.player_id, "group": g, "actual": r.rate, "n_t": r.n, "level_t": r.level,
                             "team_t": r.team,
                             "n1": 0 if h1 is None else h1["n"], "y1": 0.0 if h1 is None else h1["yards"],
                             "lvl1": np.nan if h1 is None else h1["level"], "team1": None if h1 is None else h1["team"],
                             "n2": 0 if h2 is None else h2["n"], "y2": 0.0 if h2 is None else h2["yards"]})
    return pd.DataFrame(rows)


def project_players(d: pd.DataFrame, p: pd.DataFrame, K: float, train_seasons) -> np.ndarray:
    """Shrunk projection for rows d, with means and the level shift estimated on train_seasons only."""
    tr = p[p["season"].isin(train_seasons)]
    out = np.full(len(d), np.nan)
    for g in ("QB", "RB"):
        q = tr[tr["group"] == g]
        mean = q["yards"].sum() / q["n"].sum()
        # Level shift: the change in rate per level step among players who moved levels between consecutive training
        # seasons (weighted by opportunities), from the training seasons only.
        a = q.set_index(["season", "player_id"])
        b = q.assign(season=q["season"] - 1).set_index(["season", "player_id"])
        j = a.join(b, rsuffix="_n", how="inner")
        j = j[(j["level"] != j["level_n"]) & (j["n"] >= 20) & (j["n_n"] >= 20)]
        step = float(np.average((j["rate_n"] - j["rate"]) / (j["level_n"] - j["level"]), weights=np.minimum(j["n"], j["n_n"]))) if len(j) > 10 else 0.0
        first = q[q["season"] == q["season"]]  # every training player-season; first-timers below
        seen = set()
        newbie = []
        for s in sorted(first["season"].unique()):
            qs = first[first["season"] == s]
            newbie.append(qs[~qs["player_id"].isin(seen)])
            seen |= set(qs["player_id"])
        nb = pd.concat(newbie)
        nb = nb[nb["season"] > min(SEASONS)]
        new_mean = nb["yards"].sum() / max(nb["n"].sum(), 1)
        m = (d["group"] == g).to_numpy()
        x = d[m]
        n = x["n1"] + 0.5 * x["n2"]
        rate = (x["y1"] + 0.5 * x["y2"]) / n.where(n > 0, np.nan)
        w = n / (n + K)
        shift = step * (x["level_t"] - x["lvl1"]).fillna(0)
        out[m] = np.where(n > 0, mean + w * (rate - mean) + shift, new_mean)
    return out


def _boot(d: pd.DataFrame, a: np.ndarray, b: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    keys = d["player_id"].to_numpy()
    uniq = np.unique(keys)
    pos = {k: np.flatnonzero(keys == k) for k in uniq}
    y = d["actual"].to_numpy()
    r = lambda i, v: np.corrcoef(v[i], y[i])[0, 1]  # noqa: E731
    draws = []
    for _ in range(N_BOOT):
        i = np.concatenate([pos[k] for k in rng.choice(uniq, len(uniq))])
        draws.append(r(i, a) - r(i, b))
    allidx = np.arange(len(d))
    return {"est": round(float(r(allidx, a) - r(allidx, b)), 4), "ci90": [round(float(np.percentile(draws, 5)), 4), round(float(np.percentile(draws, 95)), 4)]}


def team_context(t: pd.DataFrame) -> dict:
    k = t.set_index(["season", "team"])
    out = {}
    for col in ("ypd", "rb_frac"):
        rows = []
        for s in DEV:
            for team in t.loc[t["season"] == s, "team"]:
                if (s - 1, team) not in k.index:
                    continue
                x1 = k.loc[(s - 1, team), col]
                x2 = k.loc[(s - 2, team), col] if (s - 2, team) in k.index else x1
                rows.append({"season": s, "team": team, "x1": x1, "x2": x2, "y": k.loc[(s, team), col]})
        d = pd.DataFrame(rows)
        pred = np.full(len(d), np.nan)
        for s in DEV:
            tr, te = d["season"] != s, d["season"] == s
            A = np.c_[np.ones(tr.sum()), d.loc[tr, ["x1", "x2"]].to_numpy()]
            beta = np.linalg.lstsq(A, d.loc[tr, "y"].to_numpy(), rcond=None)[0]
            pred[te.to_numpy()] = np.c_[np.ones(te.sum()), d.loc[te, ["x1", "x2"]].to_numpy()] @ beta
        y = d["y"].to_numpy()
        out[col] = {"team_seasons": int(len(d)), "mean": round(float(y.mean()), 4),
                    "model": {"r": round(float(np.corrcoef(pred, y)[0, 1]), 4), "mae": round(float(np.abs(pred - y).mean()), 4)},
                    "carry": {"r": round(float(np.corrcoef(d["x1"], y)[0, 1]), 4), "mae": round(float(np.abs(d["x1"] - y).mean()), 4)},
                    "mean_only_mae": round(float(np.abs(y.mean() - y).mean()), 4)}
    return out


def load(cached: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cached and CACHE_P.exists() and CACHE_T.exists():
        return pd.read_parquet(CACHE_P), pd.read_parquet(CACHE_T)
    p, t = seasons()
    p.to_parquet(CACHE_P, index=False)
    t.to_parquet(CACHE_T, index=False)
    return p, t


def team_dev(p: pd.DataFrame, season, team, group) -> np.ndarray:
    """The team's rate at the position in the season before, minus the league's (0 when unknown)."""
    tr = p.groupby(["season", "team", "group"]).agg(y=("yards", "sum"), n=("n", "sum"))
    tr = (tr["y"] / tr["n"]).rename("rate")
    lg = p.groupby(["season", "group"]).apply(lambda z: z["yards"].sum() / z["n"].sum(), include_groups=False)
    out = [tr.get((s - 1, tm, g), np.nan) - lg.get((s - 1, g), np.nan) for s, tm, g in zip(season, team, group, strict=True)]
    return np.nan_to_num(np.array(out, float))


def main() -> None:
    p, t = load("--cached" in sys.argv)
    p = p[p["group"].isin(["QB", "RB"])]
    d = player_rows(p)
    assert d["season"].max() <= 2020, "development seasons only"
    res = {"status": "DEVELOPMENT (CFB-PROJ P3): seasons 2016-2020 only; test seasons 2021-2025 not scored", "players": {}}
    # K by leave-one-season-out over the development seasons (the season scored is never used to pick its own K).
    pred = np.full(len(d), np.nan)
    chosen = {}
    for s in DEV:
        te = (d["season"] == s).to_numpy()
        inner = [x for x in DEV if x != s]
        train_p = [x for x in SEASONS if x != s]
        best = {}
        for g in ("QB", "RB"):
            errs = {}
            for K in K_GRID:
                v = np.concatenate([project_players(d[(d["season"] == x) & (d["group"] == g)], p, K, [y for y in train_p if y != x])
                                    for x in inner])
                a = np.concatenate([d.loc[(d["season"] == x) & (d["group"] == g), "actual"].to_numpy() for x in inner])
                errs[K] = float(np.mean(np.abs(v - a)))
            best[g] = min(errs, key=errs.get)
        chosen[s] = best
        for g in ("QB", "RB"):
            m = te & (d["group"] == g).to_numpy()
            pred[m] = project_players(d[m], p, best[g], train_p)
    d["base"] = pred
    # Supporting cast: the destination team's rate at the position last season (QB yards per dropback, RB yards per
    # carry, everyone at the position), relative to the league; weight fit on the other development seasons.
    team_rate = p.groupby(["season", "team", "group"]).agg(y=("yards", "sum"), n=("n", "sum"))
    team_rate = (team_rate["y"] / team_rate["n"]).rename("rate")
    league = p.groupby(["season", "group"]).apply(lambda z: z["yards"].sum() / z["n"].sum(), include_groups=False)
    d["team_dev"] = [team_rate.get((s - 1, tm, g), np.nan) - league.get((s - 1, g), np.nan)
                     for s, tm, g in zip(d["season"], d["team_t"], d["group"], strict=True)]
    d["team_dev"] = d["team_dev"].fillna(0.0)
    d["pred"] = np.nan
    lam = {}
    for s in DEV:
        for g in ("QB", "RB"):
            tr = (d["season"] != s) & (d["group"] == g)
            te = (d["season"] == s) & (d["group"] == g)
            x, r = d.loc[tr, "team_dev"], d.loc[tr, "actual"] - d.loc[tr, "base"]
            lam.setdefault(g, []).append(float((x * r).sum() / max((x * x).sum(), 1e-9)))
            d.loc[te, "pred"] = d.loc[te, "base"] + lam[g][-1] * d.loc[te, "team_dev"]
    for g in ("QB", "RB"):
        q = d[d["group"] == g].reset_index(drop=True)
        mean = np.full(len(q), np.nan)
        for s in DEV:
            tr = p[(p["group"] == g) & (p["season"] != s)]
            mean[(q["season"] == s).to_numpy()] = tr["yards"].sum() / tr["n"].sum()
        carry = np.where(q["n1"] > 0, q["y1"] / q["n1"].where(q["n1"] > 0, 1), mean)
        y = q["actual"].to_numpy()
        res.setdefault("fitted", {})[g] = {"K": {s: v[g] for s, v in chosen.items()}, "team_weight": dict(zip(DEV, lam[g], strict=True))}
        res["players"][g] = {
            "rows": int(len(q)), "with_history": int((q["n1"] + q["n2"] > 0).sum()), "K_chosen": sorted({v[g] for v in chosen.values()}),
            "model": {"r": round(float(np.corrcoef(q["pred"], y)[0, 1]), 4), "mae": round(float(np.abs(q["pred"] - y).mean()), 4)},
            "carry": {"r": round(float(np.corrcoef(carry, y)[0, 1]), 4), "mae": round(float(np.abs(carry - y).mean()), 4)},
            "mean": {"mae": round(float(np.abs(mean - y).mean()), 4)},
            "model_minus_carry_r": _boot(q, q["pred"].to_numpy(), carry),
            "without_team": {"r": round(float(np.corrcoef(q["base"], y)[0, 1]), 4), "mae": round(float(np.abs(q["base"] - y).mean()), 4)},
            "team_weight": [round(v, 3) for v in lam[g]],
            "team_minus_without_r": _boot(q, q["pred"].to_numpy(), q["base"].to_numpy()),
        }
    res["teams"] = team_context(t)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "efficiency_dev.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
