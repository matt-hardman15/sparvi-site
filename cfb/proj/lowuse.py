"""Low-usage veterans in the portal (EXPLORATORY research, Matt Nov 16; seen data, 2021-2025, the 2026 vault untouched).
The portal board's counterfactual put fourth-year backups near the top of program lists (Cole Gonzales, 1.9% of Pitt's
dropbacks, ranked 9th at Boise State). Questions:
  1. Do low-usage veterans who transfer (3+ seasons, under the backup minimum last season) win roles, by direction of
     the move (down, same level, up) and position, against transfers who had real roles?
  2. Is the projection miscalibrated for them on moves that happened (projected vs. actual share), or only in the
     counterfactual (where he could go anywhere)?
  3. Within the group, what separates the ones who win a role: recruiting rating, height and weight, level drop,
     experience, the projection itself? (AUC for winning the role; r with actual share.)
Writes cfb/proj/output/lowuse.json (aggregates only).

    python cfb/proj/lowuse.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from backtest.asof import Loader
from cfb.fit import labels as LB
from cfb.fit import model as M
from cfb.proj import pit
from sparvi.config import ROOT

OUT = ROOT / "cfb" / "proj" / "output"
SEASONS = range(2021, 2026)
PRIMARY = {"QB": "QB", "RB": "RB", "WR": "WR_rec", "TE": "TE_rec"}
ROLE = {"QB": 0.5, "RB": 0.35, "WR": 0.12, "TE": 0.08}     # share of the pool that counts as winning the role
SIZE = {"QB": "height", "RB": "weight", "WR": "height", "TE": "weight"}


def transfers(L, t: int) -> pd.DataFrame:
    c = pit.project(L, t, pd.Timestamp(f"{t}-05-01"))
    c = c[(c["status"] == "transfer") & (c["category"] == c["pos"].map(PRIMARY))].copy()
    cur, prev = pit.season(L, t), pit.season(L, t - 1)
    moved = LB.transfers({t - 1: prev["roster"], t: cur["roster"]}, t, M.portal(L, t, M.done(t)).assign(season=t),
                         {t - 1: prev["played"], t: cur["played"]})
    new_id = dict(zip(moved["prev_id"].dropna().astype(str), moved.loc[moved["prev_id"].notna(), "player_id"].astype(str), strict=True))
    work = cur["work"].set_index(["player_id", "group"])["share"]
    c["actual_share"] = [work.get((new_id.get(str(k), str(k)), g), 0.0) for k, g in zip(c["key"], c["group"], strict=True)]
    vol = prev["p"].set_index(["player_id", "group"])["n"]
    c["last_volume"] = [vol.get((p, g), 0.0) if p else 0.0 for p, g in zip(c["player_id"], c["group"], strict=True)]
    ros = prev["roster"].assign(pid=lambda d: d["id"].astype(str)).drop_duplicates("pid").set_index("pid")
    c["height"] = c["player_id"].map(ros["height"]).where(lambda v: v.between(62, 84))
    c["weight"] = c["player_id"].map(ros["weight"]).where(lambda v: v.between(140, 400))
    c["won_role"] = c["actual_share"] >= c["pos"].map(ROLE)
    c["low_vet"] = (c["exp"] >= 3) & (c["last_volume"] < c["pos"].map(LB.BACKUP_MIN))
    c["direction"] = np.select([c["dlvl"] < 0, c["dlvl"] > 0], ["down", "up"], "same")
    c["season"] = t
    print(f"  {t}: {len(c)} transfers, {int(c['low_vet'].sum())} low-usage veterans", flush=True)
    return c


def _auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return None if ok.sum() < 30 or len(np.unique(y[ok])) < 2 else round(float(roc_auc_score(y[ok], x[ok])), 3)


def _r(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return None if ok.sum() < 30 or x[ok].std() == 0 else round(float(np.corrcoef(x[ok], y[ok])[0, 1]), 3)


def summary(d: pd.DataFrame) -> dict:
    return {"n": int(len(d)), "won_role": round(float(d["won_role"].mean()), 3) if len(d) else None,
            "actual_share": round(float(d["actual_share"].mean()), 3) if len(d) else None,
            "projected_share": round(float(d["share"].mean()), 3) if len(d) else None,
            "projected_over_actual": round(float(d["share"].mean() / max(d["actual_share"].mean(), 1e-9)), 2) if len(d) else None}


def main() -> None:
    L = Loader()
    c = pd.concat([transfers(L, t) for t in SEASONS], ignore_index=True)
    res = {"status": "EXPLORATORY: seasons 2021-2025 have been seen; the 2026 vault is untouched", "by_position": {}}
    for pos in ("QB", "RB", "WR", "TE"):
        d = c[c["pos"] == pos]
        lv, rest = d[d["low_vet"]], d[~d["low_vet"]]
        out = {"all_transfers": summary(d), "low_usage_veterans": summary(lv), "other_transfers": summary(rest),
               "low_usage_veterans_by_direction": {k: summary(lv[lv["direction"] == k]) for k in ("down", "same", "up")},
               "other_transfers_by_direction": {k: summary(rest[rest["direction"] == k]) for k in ("down", "same", "up")}}
        size = SIZE[pos]
        out["within_low_usage_veterans"] = {
            "won_role_auc": {"projection": _auc(lv["share"], lv["won_role"]), "recruiting_rating": _auc(lv["hs"], lv["won_role"]),
                             size: _auc(lv[size], lv["won_role"]), "level_drop": _auc(-lv["dlvl"], lv["won_role"]),
                             "experience": _auc(lv["exp"], lv["won_role"]), "last_volume": _auc(lv["last_volume"], lv["won_role"]),
                             "last_efficiency": _auc(lv["eff"], lv["won_role"])},
            "share_r": {"projection": _r(lv["share"], lv["actual_share"]), "recruiting_rating": _r(lv["hs"], lv["actual_share"]),
                        size: _r(lv[size], lv["actual_share"]), "level_drop": _r(-lv["dlvl"], lv["actual_share"]),
                        "experience": _r(lv["exp"], lv["actual_share"]), "last_volume": _r(lv["last_volume"], lv["actual_share"])}}
        res["by_position"][pos] = out
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lowuse.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
