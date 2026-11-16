"""FIT-STABLE (PLAN-fit Phase 1b, exploratory): which profile components are stable from one season to the next, and
which ones at a player's old school predict how he does at the new one? FIT-AUDIT found EPA per play noisy year to year
(WR r 0.16), so this looks for measures worth building the outcome and the inputs on. Seen data only; no 2026 outcome is
read. Writes cfb/fit/output/stability.json (aggregates only).

  stability   per component and position: r(shrunk value in season s, s+1), both seasons at or above the component's
              min_n; players who stayed at the same school, and transfers (old school -> new school)
  predicts    per component, measured at the old school: r with year-1 value over replacement at the new school, and
              AUC for winning the role in year 1 and for F3 success within two years (cohorts 2018-2024)
  composite   per position, fixed before looking at transfers: the performance components (better = higher/lower) whose
              stability among players who STAYED is at least STABLE_R, each as a within-position z-score oriented so
              higher is better, averaged. Compared on transfers with the current efficiency composite (EPA + success
              rate, model.efficiency) and last season's share

    python cfb/fit/stability.py            # --cached reuses the private profile table
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from backtest.asof import Loader
from cfb.fit import data as D
from cfb.fit import labels as LB
from cfb.profile import engine as G
from sparvi.config import RAW, ROOT

OUT = ROOT / "cfb" / "fit" / "output"
CACHE = RAW / "cfb" / "fit_profiles.parquet"            # private (built from CFBD plays)
SEASONS = range(2017, 2026)
STABLE_R = 0.30
GROUPS = ("QB", "RB", "WR", "TE")
FEATS = {f["id"]: f for f in G.load_features()}


def profiles() -> pd.DataFrame:
    L = Loader()
    out = []
    for s in SEASONS:
        ev, pos = D.season_events(L, s, pd.Timestamp(f"{s + 1}-02-01"))
        out.append(G.season_profile(ev, pos, s))
        print(f"  profile {s}", flush=True)
    return pd.concat(out, ignore_index=True)


def _ci(r: float, n: int) -> list[float]:
    if n < 4 or not np.isfinite(r):
        return [np.nan, np.nan]
    z, se = np.arctanh(np.clip(r, -0.999, 0.999)), 1.645 / np.sqrt(n - 3)
    return [round(float(np.tanh(z - se)), 4), round(float(np.tanh(z + se)), 4)]


def _r(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    r = float(np.corrcoef(a[ok], b[ok])[0, 1]) if n > 3 and a[ok].std() > 0 and b[ok].std() > 0 else np.nan
    return {"n": n, "r": None if not np.isfinite(r) else round(r, 4), "ci90": _ci(r, n)}


def _auc(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30 or len(np.unique(y[ok])) < 2:
        return {"n": int(ok.sum()), "auc": None}
    return {"n": int(ok.sum()), "auc": round(float(roc_auc_score(y[ok], x[ok])), 4)}


def pairs(prof: pd.DataFrame, moves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide (season, player) x feature tables of qualified shrunk values; stayers' and transfers' consecutive pairs."""
    q = prof[prof["n"] >= prof["feature"].map(lambda f: FEATS[f]["min_n"])]
    wide = q.pivot_table(index=["season", "player_id", "group", "team"], columns="feature", values="shrunk").reset_index()
    wide = wide.sort_values("team").drop_duplicates(["season", "player_id", "group"])
    w = wide.set_index(["season", "player_id"])
    nxt = wide.assign(season=wide["season"] - 1).set_index(["season", "player_id"])
    stay = w.join(nxt, rsuffix="__next", how="inner")
    stay = stay[(stay["team"] == stay["team__next"]) & (stay["group"] == stay["group__next"])].reset_index()
    a = w.reindex(list(zip(moves["transfer_season"] - 1, moves["prev_id"], strict=True))).reset_index(drop=True)
    b = w.reindex(list(zip(moves["transfer_season"], moves["player_id"], strict=True))).add_suffix("__next").reset_index(drop=True)
    moved = pd.concat([moves.reset_index(drop=True), a.drop(columns=["group"]), b], axis=1)
    return stay, moved


def composite(frame: pd.DataFrame, feats: list[str], ref: pd.DataFrame, suffix: str = "") -> pd.Series:
    """Mean of oriented z-scores (mean and sd from `ref`, the stayers' first seasons at the position)."""
    zs = []
    for f in feats:
        z = (frame[f + suffix] - ref[f].mean()) / ref[f].std()
        zs.append(-z if FEATS[f]["better"] == "lower" else z)
    return pd.concat(zs, axis=1).mean(axis=1) if zs else pd.Series(np.nan, index=frame.index)


def main() -> None:
    prof = pd.read_parquet(CACHE) if "--cached" in sys.argv and CACHE.exists() else profiles()
    prof.to_parquet(CACHE, index=False)
    ex = pd.read_parquet(RAW / "cfb" / "fit_explore.parquet")
    ex["prev_id"], ex["player_id"] = ex["prev_id"].astype(str), ex["player_id"].astype(str)
    ex = ex.drop_duplicates(["transfer_season", "prev_id", "group"])
    ys = pd.read_parquet(RAW / "cfb" / "fit_audit_years.parquet")
    ys = ys[ys["transfer_season"] <= 2024]
    s2 = ys[ys["year"] <= 2].groupby(["player_id", "group", "transfer_season"])["success"].any()
    ex["success2"] = s2.reindex(list(zip(ex["player_id"], ex["group"], ex["transfer_season"], strict=True))).astype(float).to_numpy()
    ex["won1"] = ex["won_role"].astype(float)
    stay, moved = pairs(prof, ex[["transfer_season", "prev_id", "player_id", "group", "A_vor", "won1", "success2", "p_share"]])
    res = {"status": "EXPLORATORY (FIT-STABLE): seen data only, no 2026 outcome", "stable_r": STABLE_R, "by_position": {}}
    for g in GROUPS:
        sg, mg = stay[stay["group"] == g], moved[moved["group"] == g]
        rows = []
        for f, spec in FEATS.items():
            if g not in spec["group"] or f not in stay.columns:
                continue
            sgn = -1.0 if spec["better"] == "lower" else 1.0
            rows.append({"feature": f, "label": spec["label"], "better": spec["better"],
                         "stayers": _r(sg[f], sg[f + "__next"]), "transfers": _r(mg[f], mg[f + "__next"]),
                         "predicts_A_vor": _r(sgn * mg[f], mg["A_vor"]), "predicts_won_role": _auc(sgn * mg[f], mg["won1"]),
                         "predicts_success2": _auc(sgn * mg[f], mg["success2"])})
        rows.sort(key=lambda r: -(r["stayers"]["r"] if r["stayers"]["r"] is not None else -9))
        chosen = [r["feature"] for r in rows if r["better"] != "style" and (r["stayers"]["r"] or 0) >= STABLE_R]
        ref = sg
        cur = [f for f in LB.EFFICIENCY[g] if f in stay.columns]
        comp = {}
        for name, feats in (("stable", chosen), ("current_efficiency", cur)):
            st_now, st_next = composite(sg, feats, ref), composite(sg, feats, ref, "__next")
            mv_now, mv_next = composite(mg, feats, ref), composite(mg, feats, ref, "__next")
            comp[name] = {"features": feats, "stayers": _r(st_now, st_next), "transfers": _r(mv_now, mv_next),
                          "predicts_A_vor": _r(mv_now, mg["A_vor"]), "predicts_won_role": _auc(mv_now, mg["won1"]),
                          "predicts_success2": _auc(mv_now, mg["success2"])}
        comp["share_last_season"] = {"predicts_A_vor": _r(mg["p_share"], mg["A_vor"]), "predicts_won_role": _auc(mg["p_share"], mg["won1"]),
                                     "predicts_success2": _auc(mg["p_share"], mg["success2"])}
        res["by_position"][g] = {"stayer_pairs": int(len(sg)), "transfer_pairs": int(len(mg)), "components": rows, "composites": comp}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stability.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    for g, v in res["by_position"].items():
        print(f"\n== {g}: {v['stayer_pairs']} stayer pairs, {v['transfer_pairs']} transfers")
        print(f"  {'component':<22}{'better':<7}{'stay r':>8}{'xfer r':>8}{'->VOR r':>9}{'->role':>8}{'->succ':>8}")
        for r in v["components"]:
            fmt = lambda x: "    -  " if x is None else f"{x:+.3f}"  # noqa: E731
            print(f"  {r['feature']:<22}{r['better']:<7}{fmt(r['stayers']['r']):>8}{fmt(r['transfers']['r']):>8}"
                  f"{fmt(r['predicts_A_vor']['r']):>9}{fmt(r['predicts_won_role']['auc']):>8}{fmt(r['predicts_success2']['auc']):>8}")
        for name, c in v["composites"].items():
            print(f"  [{name}] {c.get('features', '')}")
            print(f"     stay {c.get('stayers', {}).get('r')}  xfer {c.get('transfers', {}).get('r')}  ->VOR {c['predicts_A_vor']['r']} "
                  f"->role {c['predicts_won_role']['auc']}  ->succ {c['predicts_success2']['auc']}")


if __name__ == "__main__":
    main()
