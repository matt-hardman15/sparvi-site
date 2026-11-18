"""CFB-PROJ P4: assembly. Projected season-t yards for every QB, RB, WR, TE on a projected FBS roster, as of May 1 of t:
team volume (P1) x share (P2) x efficiency (P3), with 80% intervals (PLAN-fit, CFB-PROJ). DEVELOPMENT ONLY: seasons
2016-2020, every piece leave-one-season-out; the test seasons (2021-2025) are never scored here, and 2026 is the vault.

  QB passing yards        = share of dropbacks x team dropbacks x his yards per dropback
  RB rushing yards        = share of RB carries x team runs x RB share of runs x his yards per carry
  WR/TE (and RB) receiving = share of receiving yards x team dropbacks x team receiving yards per dropback, where the
                             team's rate comes from its last two seasons and its projected QBs (share-weighted P3 rates)
Rows: the projected rosters (P2 candidates). Outcome: his season-t yards in that category (0 if he didn't record any).
Baselines (pre-registered in PLAN-fit): carry = his own last-season yards wherever he played (0 for freshmen);
share_only = his last share x the destination team's last-season yards in the category (0 for freshmen).
Intervals: 80%, from the other development seasons' residuals in bins of projected yards, per category.

    python cfb/proj/assemble.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backtest.asof import Loader
from cfb.fit import labels as LB
from cfb.proj import efficiency as E
from cfb.proj import shares as S
from cfb.proj import volume as V
from sparvi.config import RAW, ROOT

OUT = ROOT / "cfb" / "proj" / "output"
DEV = range(2016, 2021)
COVID = 2020
N_BINS, SEED, N_BOOT = 5, 7, 1000


def _loso_lstsq(d: pd.DataFrame, cols: list[str], y: str) -> np.ndarray:
    pred = np.full(len(d), np.nan)
    for s in DEV:
        tr, te = (d["season"] != s).to_numpy(), (d["season"] == s).to_numpy()
        A = np.c_[np.ones(tr.sum()), d.loc[tr, cols].to_numpy(float)]
        beta = np.linalg.lstsq(A, d.loc[tr, y].to_numpy(float), rcond=None)[0]
        pred[te] = np.c_[np.ones(te.sum()), d.loc[te, cols].to_numpy(float)] @ beta
    return pred


def player_rates(c: pd.DataFrame, p: pd.DataFrame, fitted: dict) -> np.ndarray:
    """P3 projection (QB yards per dropback, RB yards per carry) for each QB- or RB-pool candidate."""
    k = p.set_index(["season", "player_id", "group"])
    conf = {s: dict(pd.read_parquet(RAW / "cfb" / f"teams_{s}.parquet")[["school", "conference"]].values) for s in DEV}
    rows = []
    for r in c.itertuples():
        h1 = k.loc[(r.season - 1, r.player_id, r.group)] if r.player_id and (r.season - 1, r.player_id, r.group) in k.index else None
        h2 = k.loc[(r.season - 2, r.player_id, r.group)] if r.player_id and (r.season - 2, r.player_id, r.group) in k.index else None
        rows.append({"season": r.season, "group": r.group, "team_t": r.team, "level_t": E.LEVEL[LB.tier(r.team, r.season, conf[r.season])],
                     "n1": 0 if h1 is None else h1["n"], "y1": 0.0 if h1 is None else h1["yards"], "lvl1": np.nan if h1 is None else h1["level"],
                     "n2": 0 if h2 is None else h2["n"], "y2": 0.0 if h2 is None else h2["yards"]})
    d = pd.DataFrame(rows, index=c.index)
    out = pd.Series(np.nan, index=c.index)
    for s in DEV:
        for g in ("QB", "RB"):
            m = (d["season"] == s) & (d["group"] == g)
            if not m.any():
                continue
            K, lam = fitted[g]["K"][str(s)], fitted[g]["team_weight"][str(s)]
            base = E.project_players(d[m], p, K, [x for x in E.SEASONS if x != s])
            out[m] = base + lam * E.team_dev(p, d.loc[m, "season"], d.loc[m, "team_t"], d.loc[m, "group"])
    return out.to_numpy()


def build() -> pd.DataFrame:
    L = Loader()
    # P1: team volume.
    vd = V.frame(pd.read_parquet(V.CACHE), L)
    vp = {tg: V.predict(vd, tg)["ridge"] for tg in ("pass_pg", "run_pg", "games")}
    vol = vd[["season", "team"]].assign(pass_pg=vp["pass_pg"], run_pg=vp["run_pg"], games=vp["games"])
    vol["games"] = vol["games"].fillna(vd.loc[vd["season"] != COVID, "y_games"].mean())     # 2020 isn't projected for games
    vol = vol.set_index(["season", "team"])
    # P2: shares.
    c = S.load(True)
    c["share"], _ = S.loso(c)
    # P3: efficiency.
    p, t = E.load(True)
    fitted = json.loads((OUT / "efficiency_dev.json").read_text(encoding="utf-8"))["fitted"]
    pq = p[p["group"].isin(["QB", "RB"])]
    c["rate"] = np.nan
    m = c["group"].isin(["QB", "RB"])
    c.loc[m, "rate"] = player_rates(c[m], pq, fitted)
    # Team context: RB share of runs, and receiving yards per dropback (with the projected QBs).
    tk = t.set_index(["season", "team"])
    tc = []
    for s in DEV:
        for team in t.loc[t["season"] == s, "team"]:
            if (s - 1, team) not in tk.index:
                continue
            row = {"season": s, "team": team}
            for col in ("rb_frac", "rec_ypd"):
                row[f"{col}_1"] = tk.loc[(s - 1, team), col]
                row[f"{col}_2"] = tk.loc[(s - 2, team), col] if (s - 2, team) in tk.index else row[f"{col}_1"]
                row[col] = tk.loc[(s, team), col]
            tc.append(row)
    tc = pd.DataFrame(tc)
    q = c[c["group"] == "QB"]
    qb = (q["share"] * q["rate"]).groupby([q["season"], q["team"]]).sum() / q["share"].groupby([q["season"], q["team"]]).sum()
    tc["qb_proj"] = [qb.get((s, tm), np.nan) for s, tm in zip(tc["season"], tc["team"], strict=True)]
    tc["qb_proj"] = (tc["qb_proj"] - tc.groupby("season")["qb_proj"].transform("mean")).fillna(0.0)
    tc["rb_frac_hat"] = _loso_lstsq(tc, ["rb_frac_1", "rb_frac_2"], "rb_frac")
    tc["rec_ypd_hat"] = _loso_lstsq(tc, ["rec_ypd_1", "rec_ypd_2", "qb_proj"], "rec_ypd")
    tc = tc.set_index(["season", "team"])
    # Assembly.
    key = list(zip(c["season"], c["team"], strict=True))
    games = vol["games"].reindex(key).to_numpy()
    dropbacks = vol["pass_pg"].reindex(key).to_numpy() * games
    runs = vol["run_pg"].reindex(key).to_numpy() * games
    rb_frac = tc["rb_frac_hat"].reindex(key).to_numpy()
    rec_ypd = tc["rec_ypd_hat"].reindex(key).to_numpy()
    c["proj"] = np.select([c["group"] == "QB", c["group"] == "RB"],
                          [c["share"] * dropbacks * c["rate"], c["share"] * runs * rb_frac * c["rate"]],
                          c["share"] * dropbacks * rec_ypd)
    # Outcomes and baselines.
    yards = p.set_index(["season", "player_id", "group"])["yards"]
    c["actual"] = [yards.get((s, o, g), 0.0) for s, o, g in zip(c["season"], c["out_id"], c["group"], strict=True)]
    c["carry"] = [0.0 if not pid else yards.get((s - 1, pid, g), 0.0) for s, pid, g in zip(c["season"], c["player_id"], c["group"], strict=True)]
    col = {"QB": "pass_yards", "RB": "rb_yards", "REC": "rec_yards"}
    team_prev = [tk[col[g]].get((s - 1, tm), np.nan) for s, tm, g in zip(c["season"], c["team"], c["group"], strict=True)]
    c["share_only"] = np.where(c["status"] == "freshman", 0.0, c["last_share"] * np.nan_to_num(np.array(team_prev, float)))
    c["category"] = np.where(c["group"] == "REC", c["pos"] + "_rec", c["pos"])
    return c[c["proj"].notna()].reset_index(drop=True)


def intervals(c: pd.DataFrame) -> pd.DataFrame:
    """80% intervals: for each held-out season, residual quantiles from the other seasons in projected-yards bins."""
    c = c.copy()
    c["lo80"], c["hi80"] = np.nan, np.nan
    for _cat, d in c.groupby("category"):
        for s in DEV:
            tr, te = d[d["season"] != s], d[d["season"] == s]
            edges = np.unique(np.quantile(tr["proj"], np.linspace(0, 1, N_BINS + 1)))
            bt, be = np.clip(np.searchsorted(edges, tr["proj"], side="right") - 1, 0, len(edges) - 2), \
                np.clip(np.searchsorted(edges, te["proj"], side="right") - 1, 0, len(edges) - 2)
            res = (tr["actual"] - tr["proj"]).to_numpy()
            for b in np.unique(be):
                r = res[bt == b] if (bt == b).sum() >= 20 else res
                idx = te.index[be == b]
                c.loc[idx, "lo80"] = (c.loc[idx, "proj"] + np.quantile(r, 0.10)).clip(lower=0)
                c.loc[idx, "hi80"] = c.loc[idx, "proj"] + np.quantile(r, 0.90)
    return c


def interval_bins(c: pd.DataFrame) -> dict:
    """For the backtest and the board: per category, projected-yards bin edges and the 10th / 90th residual percentiles
    over every development season's leave-one-season-out projections."""
    out = {}
    for cat, d in c.groupby("category"):
        edges = np.unique(np.quantile(d["proj"], np.linspace(0, 1, N_BINS + 1)))
        b = np.clip(np.searchsorted(edges, d["proj"], side="right") - 1, 0, len(edges) - 2)
        res = (d["actual"] - d["proj"]).to_numpy()
        out[cat] = {"edges": [round(float(e), 3) for e in edges[:-1]],
                    "q10": [round(float(np.quantile(res[b == i], 0.10)), 3) for i in range(len(edges) - 1)],
                    "q90": [round(float(np.quantile(res[b == i], 0.90)), 3) for i in range(len(edges) - 1)]}
    return out


def _boot(d: pd.DataFrame, a: str, b: str) -> dict:
    rng = np.random.default_rng(SEED)
    keys = d["out_id"].to_numpy()
    uniq = np.unique(keys)
    pos = {k: np.flatnonzero(keys == k) for k in uniq}
    y = d["actual"].to_numpy()
    va, vb = d[a].to_numpy(), d[b].to_numpy()
    r = lambda i, v: np.corrcoef(v[i], y[i])[0, 1]  # noqa: E731
    draws = [r(i, va) - r(i, vb) for i in (np.concatenate([pos[k] for k in rng.choice(uniq, len(uniq))]) for _ in range(N_BOOT))]
    allidx = np.arange(len(d))
    return {"est": round(float(r(allidx, va) - r(allidx, vb)), 4), "ci90": [round(float(np.percentile(draws, 5)), 4), round(float(np.percentile(draws, 95)), 4)]}


def main() -> None:
    c = intervals(build())
    assert c["season"].max() <= 2020, "development seasons only"
    res = {"status": "DEVELOPMENT (CFB-PROJ P4): seasons 2016-2020 only; test seasons 2021-2025 not scored", "categories": {}}
    for cat in ("QB", "RB", "WR_rec", "TE_rec", "RB_rec"):
        d = c[c["category"] == cat].reset_index(drop=True)
        out = {"rows": int(len(d)), "actual_mean": round(float(d["actual"].mean()), 1)}
        for m in ("proj", "carry", "share_only"):
            out[m] = {"r": round(float(np.corrcoef(d[m], d["actual"])[0, 1]), 4), "mae": round(float((d[m] - d["actual"]).abs().mean()), 1)}
        out["proj_minus_carry_r"] = _boot(d, "proj", "carry")
        out["proj_minus_share_only_r"] = _boot(d, "proj", "share_only")
        out["coverage80"] = round(float(((d["actual"] >= d["lo80"]) & (d["actual"] <= d["hi80"])).mean()), 4)
        out["by_status"] = {st: {"n": int(len(q)), **{m: round(float(np.corrcoef(q[m], q["actual"])[0, 1]), 4) if q[m].std() > 0 else None
                                                      for m in ("proj", "carry", "share_only")}} for st, q in d.groupby("status")}
        res["categories"][cat] = out
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "interval_bins.json").write_text(json.dumps(interval_bins(c), indent=1) + "\n", encoding="utf-8", newline="\n")
    (OUT / "assemble_dev.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    for cat, v in res["categories"].items():
        print(f"{cat:<7} n={v['rows']:<6} proj r {v['proj']['r']:.3f}  carry {v['carry']['r']:.3f}  share_only {v['share_only']['r']:.3f}  "
              f"| vs carry {v['proj_minus_carry_r']}  vs share_only {v['proj_minus_share_only_r']}  cover80 {v['coverage80']}")
        print(f"         by status {v['by_status']}")


if __name__ == "__main__":
    main()
