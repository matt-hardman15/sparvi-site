"""RECV-MEASURES follow-up (EXPLORATORY; every transfer here has been seen, the 2026 vault is not read): Phase 1's WR/TE
style-match leads and FIT-AUDIT's agreement test, rerun with receiver value built on EPA per team pass play (profile
1.1.0) instead of EPA per target, which 2021-2024's missing incompletions bias. Writes cfb/fit/output/recv_rerun.json.

Receiver value over replacement, per season and competition level:
  rate          his opponent-adjusted, shrunk receiving EPA per team pass play (rec_epa_tp)
  replacement   the team-pass-play-weighted mean rate of backups: outside his team's top 3 (WR) / top 2 (TE) in catches
                per team pass play, with at least 150 team pass plays
  value         (rate - replacement) x his team pass plays
Outcomes (as Phase 1): A = value in year 1 at the new school; B = the best of years 2-3 there (cohorts through 2023).
Quality-only baseline: explore.BASE with the previous-season efficiency and value on the same per-team-play basis.

    python cfb/fit/recv_rerun.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cfb.fit import audit as AU
from cfb.fit import explore as X
from cfb.fit import labels as LB
from sparvi.config import RAW, ROOT

OUT = ROOT / "cfb" / "fit" / "output"
TOP = {"WR": 3, "TE": 2}
MIN_N = 150


def receiver_values() -> pd.DataFrame:
    prof = pd.read_parquet(RAW / "cfb" / "fit_profiles.parquet")
    p = prof[prof["group"].isin(["WR", "TE"]) & prof["feature"].isin(["rec_epa_tp", "rec_catch_tp"])]
    w = p.pivot_table(index=["season", "player_id", "group", "team"], columns="feature", values="shrunk").reset_index()
    n = p[p["feature"] == "rec_epa_tp"].set_index(["season", "player_id", "group"])[["n", "adj"]]
    w = w.join(n, on=["season", "player_id", "group"]).dropna(subset=["rec_epa_tp", "rec_catch_tp"])
    conf = {s: dict(pd.read_parquet(RAW / "cfb" / f"teams_{s}.parquet")[["school", "conference"]].values) for s in w["season"].unique()}
    w["tier"] = [LB.tier(t, s, conf[s]) for t, s in zip(w["team"], w["season"], strict=True)]
    w["rank"] = w.groupby(["season", "team", "group"])["rec_catch_tp"].rank(ascending=False, method="min")
    backup = (w["rank"] > w["group"].map(TOP)) & (w["n"] >= MIN_N)
    rep = w[backup].groupby(["season", "group", "tier"]).apply(lambda x: float(np.average(x["adj"], weights=x["n"])), include_groups=False)
    w["replacement"] = [rep.get(k, np.nan) for k in zip(w["season"], w["group"], w["tier"], strict=True)]
    w["value"] = ((w["rec_epa_tp"] - w["replacement"]) * w["n"]).fillna(0.0)
    return w


def frame(vals: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_parquet(RAW / "cfb" / "fit_explore.parquet")
    df = df[df["group"].isin(["WR", "TE"])].copy()
    df["prev_id"], df["player_id"] = df["prev_id"].astype(str), df["player_id"].astype(str)
    v = vals.assign(player_id=vals["player_id"].astype(str)).sort_values("n", ascending=False).drop_duplicates(["season", "player_id"])
    by = v.set_index(["season", "player_id"])
    at = v.set_index(["season", "player_id", "team"])["value"]
    df["A_tp"] = [at.get((s, p, t), 0.0) for s, p, t in zip(df["transfer_season"], df["player_id"], df["team_to"], strict=True)]
    later = [max(at.get((s + k, p, t), 0.0) for k in (1, 2)) for s, p, t in zip(df["transfer_season"], df["player_id"], df["team_to"], strict=True)]
    df["B_tp"] = np.where(df["transfer_season"] <= X.B_LAST_COHORT, later, np.nan)
    k = list(zip(df["transfer_season"] - 1, df["prev_id"], strict=True))
    df["p_epa"] = by["rec_epa_tp"].reindex(k).to_numpy()
    df["p_vor"] = by["value"].reindex(k).to_numpy()
    df["no_prior"] = pd.isna(df["p_epa"]).astype(float)
    df["p_epa"] = df["p_epa"].fillna(df.groupby("group")["p_epa"].transform("median"))
    df["p_vor"] = df["p_vor"].fillna(0.0)
    return df


def analyses(df: pd.DataFrame) -> list[dict]:
    log = []

    def add(aid, desc, d, stat):
        est, ci = X._boot(d.reset_index(drop=True), stat)
        log.append({"id": aid, "analysis": desc, "n": int(len(d)), "est": round(est, 4), "ci90": [round(ci[0], 4), round(ci[1], 4)],
                    "excludes_zero": bool(ci[0] > 0 or ci[1] < 0)})

    sty = AU.style_scores(df)
    df["rank_style"] = AU.destination_rank(df, sty, "style")
    for g in ("WR", "TE"):
        for out in ("A_tp", "B_tp"):
            d = df[(df["group"] == g) & df[out].notna()].copy()
            d["resid"] = d[out] - X._loco(d, X.BASE, out)
            q = d[d["M1_distance"].notna()]
            add(f"{g}.{out}.M1.all", "corr(style distance, residual); closer is better if negative", q, X._corr("M1_distance", "resid"))
            k = q[q["same_coach"]]
            add(f"{g}.{out}.M1.kept_coach", "same, destinations that kept their head coach", k, X._corr("M1_distance", "resid"))
            r = d[d["rank_style"].notna()].copy()
            r["third"] = AU._third(r["rank_style"])
            r["cat"] = np.where(r[out] == 0, "didnt_play", np.where(r["resid"] > 0, "beat", "fell_short"))
            add(f"{g}.{out}.agree", "agreement rate minus chance (style rank thirds vs. beat / fell short or didn't play)", r,
                lambda z: AU._agree(z, "cat"))
            rk = r[r["same_coach"]]
            add(f"{g}.{out}.agree.kept_coach", "same, destinations that kept their head coach", rk, lambda z: AU._agree(z, "cat"))
    return log


def main() -> None:
    vals = receiver_values()
    df = frame(vals)
    log = analyses(df)
    res = {"status": "EXPLORATORY (RECV-MEASURES follow-up): seen data only; the 2026 vault is not read",
           "receiver_value_definition": "rate = receiving EPA per team pass play (shrunk); replacement = backups' weighted mean",
           "transfers": {g: int((df["group"] == g).sum()) for g in ("WR", "TE")}, "analyses_run": len(log),
           "expected_by_chance_at_90pct": round(0.10 * len(log), 1), "excluding_zero": sum(a["excludes_zero"] for a in log),
           "analyses": log}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "recv_rerun.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    for a in log:
        print(f"{'*' if a['excludes_zero'] else ' '} {a['id']:<28} n={a['n']:<5} est={a['est']:+.4f} ci90=[{a['ci90'][0]:+.4f}, {a['ci90'][1]:+.4f}]")


if __name__ == "__main__":
    main()
