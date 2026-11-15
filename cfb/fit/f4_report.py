"""SYSTEM-FIT F4/F5 report: the pre-registered secondary results of the system-fit backtest (models/cfb/system_fit/
card.yml, `secondary`), plus the harness's primary result, written to cfb/fit/output/f4.json (the card's validation file).
Run after `python -m backtest.runner cfb_system_fit`; nothing here can change the gate.

    python cfb/fit/f4_report.py

Windows 2021-2024 use the two-year label; the 2025 window uses the year-1 label (its second year is the 2026 vault).
Methods: fit (the model), transfer_rating, raw_production, profile_only, random; plus fit_x_need (probability x the
program's open share at the position, F0).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backtest import metrics
from backtest.asof import Loader
from backtest.splits import season_folds
from cfb.fit import model as M
from sparvi import registry
from sparvi.config import ROOT

OUT = ROOT / "cfb" / "fit" / "output"
WINDOWS = (2021, 2022, 2023, 2024)
YEAR1_WINDOW = 2025
SCORES = {"fit": "p", "transfer_rating": "rating_key", "raw_production": "yards", "profile_only": "eff_key"}


def _keys(b: pd.DataFrame) -> pd.DataFrame:
    return b.assign(rating_key=b["stars"].fillna(0) + b["rating"].fillna(0) / 10, eff_key=b["eff"].fillna(-1e9))


def _credits(b: pd.DataFrame, top: int) -> pd.DataFrame:
    by = [b["team"], b["group"]]
    out = pd.DataFrame({"entity_id": b["season"].astype(str) + "|" + b["player"] + "|" + b["team"], "player": b["player"],
                        "group": b["group"]})
    for name, col in {**SCORES, "fit_x_need": "p_need"}.items():
        out[name] = M.credit(b[col], by, top)
    out["random"] = np.minimum(1.0, top / b["board_size"])
    return out


def _recall(df: pd.DataFrame) -> dict:
    df = df.reset_index(drop=True).assign(y=1.0)
    res = {"n": int(len(df))}
    for m in ("fit", "fit_x_need", "transfer_rating", "raw_production", "profile_only", "random"):
        res[m] = round(metrics.recall(df[m], df["y"]), 4)
    for b in ("transfer_rating", "raw_production", "profile_only"):
        p = metrics.paired_bootstrap(df, "recall", "fit", b, "y", "player", n=2000, seed=7) if len(df) > 1 else None
        res[f"fit_minus_{b}"] = None if p is None else {"est": round(p["est"], 4), "ci90": [round(x, 4) for x in p["ci90"]]}
    return res


def _auc(sig: pd.DataFrame, col: str) -> dict:
    """Within program, position, and window: P(a success outscores a miss), ties half."""
    wins = pairs = 0.0
    for _, g in sig.groupby(["season", "team_to", "group_b"]):
        s, f = g.loc[g["success2"], col].to_numpy(float), g.loc[~g["success2"], col].to_numpy(float)
        if len(s) and len(f):
            d = s[:, None] - f[None, :]
            wins += (d > 0).sum() + 0.5 * (d == 0).sum()
            pairs += d.size
    return {"auc": round(wins / pairs, 4) if pairs else None, "pairs": int(pairs)}


def window(L: Loader, t: int) -> dict:
    fold = season_folds(t, t, 1, "05-01", 1, "05-01")[0]
    f = M.fit_fold(L, t, fold.train_until)
    b = _keys(M.board(f, L, t, fold.decision_date))
    need = M.availability(L, t, fold.decision_date).set_index(["team", "group"])["open_share"]
    b["open_share"] = [need.get((tm, g), np.nan) for tm, g in zip(b["team"], b["group"], strict=True)]
    b["p_need"] = b["p"] * b["open_share"].fillna(need.median())
    last = t + 1 if t < YEAR1_WINDOW else t
    lab = M.cohort_outcomes(L, t, last)
    fbs = lab[lab["fbs"]].drop_duplicates("entity_id")
    c10, c25 = _credits(b, 10), _credits(b, 25)
    succ = fbs[fbs["success2"]]
    out = {"training": {g: {"n": m["n"], "success_rate": round(m["rate"], 4), "C": m["C"],
                            "coef": {k: round(v, 4) for k, v in m["coef"].items()}} for g, m in f["models"].items()},
           "board": {"entrants": int(b["player"].nunique()), "programs": int(b["team"].nunique()),
                     "size_by_position": b.groupby("group")["board_size"].first().astype(int).to_dict()},
           "successes_fbs": int(len(succ)), "successes_on_board": int(succ["entity_id"].isin(set(c10["entity_id"])).sum())}
    rows = {"r10": c10.merge(succ[["entity_id", "dest_tier"]], on="entity_id"),
            "r25": c25.merge(succ[["entity_id"]], on="entity_id"),
            "eff10": c10.merge(fbs.loc[fbs["efficient2"], ["entity_id"]], on="entity_id")}
    # Signees on a board: their scores at the program they joined.
    sig = fbs.merge(b.assign(entity_id=b["season"].astype(str) + "|" + b["player"] + "|" + b["team"])
                    [["entity_id", "p", "p_need", "rating_key", "yards", "eff_key", "group"]].rename(columns={"group": "group_b"}), on="entity_id")
    sig["season"] = t
    fcs = None
    if t < YEAR1_WINDOW:
        bf = _keys(M.board(f, L, t, fold.decision_date, programs="fcs"))
        bf["p_need"] = bf["p"]
        if len(bf):
            fs = lab[~lab["fbs"] & lab["success2"]].drop_duplicates("entity_id")
            fcs = _credits(bf, 10).merge(fs[["entity_id"]], on="entity_id")
    return {"summary": out, "rows": rows, "signees": sig, "fcs": fcs}


def main() -> None:
    m = registry.get("cfb_system_fit")
    rep = json.loads((m.dir / "backtest" / f"{m.version}.json").read_text(encoding="utf-8"))
    ev = rep["evaluation"]
    res = {"primary": {"metric": ev["metric"], "n": ev["n"], "point": round(ev["point"], 4),
                       **{b: round(v["value"], 4) for b, v in ev["baselines"].items()},
                       "edges": {b: {"est": round(v["paired"]["est"], 4), "ci90": [round(x, 4) for x in v["paired"]["ci90"]]}
                                 for b, v in ev["baselines"].items()},
                       "gate_passed": rep["gate"]["passed"], "run_id": rep["run_id"], "kind": rep["kind"]},
           "windows": {}, "secondary": {}}
    L = Loader()
    w = {t: window(L, t) for t in (*WINDOWS, YEAR1_WINDOW)}
    for t, x in w.items():
        res["windows"][t] = x["summary"]
    two = [w[t] for t in WINDOWS]
    r10 = pd.concat([x["rows"]["r10"] for x in two], ignore_index=True)
    sig = pd.concat([x["signees"] for x in two], ignore_index=True)
    sec = res["secondary"]
    sec["recall_at_10_check"] = _recall(r10)
    sec["recall_at_25"] = _recall(pd.concat([x["rows"]["r25"] for x in two], ignore_index=True))
    sec["efficient_label_recall_at_10"] = _recall(pd.concat([x["rows"]["eff10"] for x in two], ignore_index=True))
    sec["by_position"] = {g: _recall(d) for g, d in r10.groupby("group")}
    sec["by_destination_level"] = {g: _recall(d) for g, d in r10.groupby("dest_tier")}
    sec["by_window"] = {t: _recall(w[t]["rows"]["r10"]) for t in WINDOWS}
    sec["window_2025_year1_label"] = _recall(w[YEAR1_WINDOW]["rows"]["r10"])
    fcs = [x["fcs"] for x in two if x["fcs"] is not None and len(x["fcs"])]
    sec["fcs_destinations"] = _recall(pd.concat(fcs, ignore_index=True)) if fcs else {"n": 0}
    sec["auc_among_signees"] = {name: _auc(sig, col) for name, col in {**SCORES, "fit_x_need": "p_need"}.items()}
    s = sig.dropna(subset=["p"])
    y = s["success2"].astype(float)
    dec = pd.qcut(s["p"].rank(method="first"), 10, labels=False)
    sec["calibration_signees"] = {
        "n": int(len(s)), "success_rate": round(float(y.mean()), 4),
        "brier": round(float(((s["p"] - y) ** 2).mean()), 4),
        "brier_base_rate": round(float(((y.mean() - y) ** 2).mean()), 4),
        "deciles": [{"predicted": round(float(s.loc[dec == k, "p"].mean()), 4), "actual": round(float(y[dec == k].mean()), 4),
                     "n": int((dec == k).sum())} for k in range(10)]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "f4.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: res[k] for k in ("primary", "secondary")}, indent=1, default=str))


if __name__ == "__main__":
    main()
