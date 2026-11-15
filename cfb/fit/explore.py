"""SYSTEM-FIT Phase 1 (SYSTEM-FIT-EXPLORE): where does style match add to a quality-only model? EXPLORATORY: every
transfer 2018-2025 has been seen (the F5 backtest used 2021-2024), so nothing here is a test. It produces hypotheses
for Phase 2, which is confirmed once on the 2026 vault (PLAN-fit, decided Nov 15). Every analysis run is listed in
cfb/fit/output/explore.json with an id, so the number of attempts is visible next to how many would clear a 90%
interval by chance.

Outcomes (value over replacement, labels.value_over_replacement; Matt, Nov 15):
  A_vor    model A, immediate impact: value over replacement in year 1 at the new school (0 if he didn't play)
  A_rate   efficiency only: year-1 shrunk adjusted EPA per play minus replacement, among those who reached the backup
           minimum (isolates fit from opportunity)
  B_vor    model B, development: the best value over replacement in years 2-3 at the school (0 if he left or didn't
           play); cohorts 2018-2023, whose third year is observed
Quality-only baseline (per position, ridge, leave-one-cohort-out predictions): his previous season's shrunk EPA per
play, value over replacement, share, a no-prior flag, recruiting rating and a no-rating flag, experience, origin and
destination level, and the level change. Analyses test each fit measure against the baseline's residual.

Fit measures (destination and origin systems are F1 dimensions in the season before the move):
  M1 style distance        distance between his old and new team's systems (D1-D2); hypothesis: closer is better
  M2 type x system         his F2 type dimensions x the new system's (4 products): cross-validated R^2 gain
  M3 positional usage      how much the new system uses his position minus the old one (target share for WR/TE/RB,
                           QB run share for QB); hypothesis: more is better (opportunity)
  M4 coaching continuity   M1 split by whether the destination kept its head coach (F1: systems persist under the
                           same coach, r 0.75 vs. 0.34)

    python cfb/fit/explore.py            # --cached reuses the private frame from the last run
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from cfb.fit import data as D
from cfb.fit import f3_report
from cfb.fit import labels as LB
from sparvi.config import RAW, ROOT

OUT = ROOT / "cfb" / "fit" / "output"
P = RAW / "cfb" / "players"
LEVEL = {"FCS": 0, "G5": 1, "Power": 2}
USAGE = {"QB": "qb_run_share", "RB": "rb_target_share", "WR": "wr_target_share", "TE": "te_target_share"}
BASE = ["p_epa", "p_vor", "p_share", "no_prior", "hs", "no_hs", "exp", "lvl_from", "lvl_to", "dlvl"]
B_LAST_COHORT = 2023
N_BOOT, SEED = 1000, 7


def frame() -> pd.DataFrame:
    """One row per transfer with prior-season features, the systems involved, and outcomes A and B."""
    d, y, outcomes = f3_report.build(with_outcomes=True)
    d = d[d["prev_id"].notna()].copy()
    key = ["player_id", "group", "transfer_season"]
    y1 = y[y["year"] == 1].set_index(key)
    later = y[y["year"].isin([2, 3])].groupby(key)["vor"].max()
    df = y1.assign(B_vor=later.reindex(y1.index).fillna(0.0)).reset_index()
    df = df[df["prev_id"].notna()].copy()                   # previous-season features need his previous id
    df["A_vor"] = df["vor"]
    df["A_rate"] = df["vor_rate"].where(df["volume"] >= df["group"].map(LB.BACKUP_MIN))
    df.loc[df["transfer_season"] > B_LAST_COHORT, "B_vor"] = np.nan
    # Previous season (the player's own record, by his previous id).
    prev = pd.concat([o.assign(season=s) for s, o in outcomes.items()], ignore_index=True)
    prev = prev.sort_values("volume", ascending=False).drop_duplicates(["season", "player_id"])
    pk = prev.set_index(["season", "player_id"])
    idx = list(zip(df["transfer_season"] - 1, df["prev_id"].astype(str), strict=True))
    df["p_epa"] = pk["epa_shrunk"].reindex(idx).to_numpy()
    df["p_vor"] = pk["vor"].reindex(idx).to_numpy()
    df["p_share"] = pk["share"].reindex(idx).to_numpy()
    df["no_prior"] = pd.isna(df["p_epa"]).astype(float)
    for c in ("p_epa", "p_vor", "p_share"):
        df[c] = df[c].fillna(df.groupby("group")[c].transform("median") if c == "p_epa" else 0.0)
    rosters = {s: pd.read_parquet(P / f"roster_{s}.parquet") for s in range(2014, 2026)}
    recruits = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(P / "recruits_*.parquet")))], ignore_index=True)
    hs = (recruits[recruits["athleteId"].notna()].assign(pid=lambda r: r["athleteId"].astype(str))
          .sort_values(["year", "rating"], ascending=[True, False]).drop_duplicates("pid").set_index("pid")["rating"])
    df["hs"] = df["prev_id"].astype(str).map(hs)
    df["no_hs"] = df["hs"].isna().astype(float)
    df["hs"] = df["hs"].fillna(hs.min())
    df["exp"] = _experience(df, rosters, recruits)
    conf = {s: dict(pd.read_parquet(RAW / "cfb" / f"teams_{s}.parquet")[["school", "conference"]].values) for s in range(2017, 2026)}
    df["lvl_from"] = [LEVEL[LB.tier(t, s - 1, conf[s - 1])] for t, s in zip(df["team_from"], df["transfer_season"], strict=True)]
    df["lvl_to"] = [LEVEL[LB.tier(t, s, conf[s])] for t, s in zip(df["team_to"], df["transfer_season"], strict=True)]
    df["dlvl"] = df["lvl_to"] - df["lvl_from"]
    # Systems (F1 output: dimensions fit on 2014-2025 team-seasons) in the season before the move.
    st = pd.read_csv(OUT / "team_styles.csv").set_index(["season", "team"])
    for side, team in (("from", "team_from"), ("to", "team_to")):
        k = list(zip(df["transfer_season"] - 1, df[team], strict=True))
        for c in ("D1", "D2", *USAGE.values()):
            df[f"{side}_{c}"] = st[c].reindex(k).to_numpy()
    df["M1_distance"] = np.hypot(df["to_D1"] - df["from_D1"], df["to_D2"] - df["from_D2"])
    df["M3_usage_change"] = [r[f"to_{USAGE[g]}"] - r[f"from_{USAGE[g]}"] for g, (_, r) in zip(df["group"], df.iterrows(), strict=True)]
    # Player types (F2 output: qualified player-seasons only).
    ty = pd.read_csv(OUT / "player_types.csv", dtype={"player_id": str}).set_index(["season", "player_id"])
    k = list(zip(df["transfer_season"] - 1, df["prev_id"].astype(str), strict=True))
    df["ty1"], df["ty2"] = ty["D1"].reindex(k).to_numpy(), ty["D2"].reindex(k).to_numpy()
    for i in (1, 2):
        for j in (1, 2):
            df[f"ty{i}_x_sys{j}"] = df[f"ty{i}"] * df[f"to_D{j}"]
    # Head coach of the destination: the same in season t as in t-1?
    co = pd.read_parquet(RAW / "cfb" / "coaches.parquet")
    head = co.sort_values("games").groupby(["season", "school"])["coach_id"].last()
    df["same_coach"] = [head.get((s, t)) is not None and head.get((s, t)) == head.get((s - 1, t))
                        for s, t in zip(df["transfer_season"], df["team_to"], strict=True)]
    return df


def _experience(df: pd.DataFrame, rosters: dict, recruits: pd.DataFrame) -> np.ndarray:
    out = np.ones(len(df))
    for s in sorted(df["transfer_season"].unique()):
        e = D.experience(s - 1, {k: v for k, v in rosters.items() if k <= s - 1}, recruits)
        m = (df["transfer_season"] == s).to_numpy()
        out[m] = df.loc[m, "prev_id"].astype(str).map(e).fillna(1).to_numpy() + 1
    return out


def _loco(d: pd.DataFrame, cols: list[str], y: str) -> np.ndarray:
    """Leave-one-cohort-out ridge predictions (features standardized on the training cohorts)."""
    pred = np.full(len(d), np.nan)
    coh = d["transfer_season"].to_numpy()
    X, Y = d[cols].to_numpy(float), d[y].to_numpy(float)
    for c in np.unique(coh):
        tr, te = coh != c, coh == c
        mu, sd = np.nanmean(X[tr], 0), np.nanstd(X[tr], 0)
        sd[sd == 0] = 1
        Z = np.nan_to_num((X - mu) / sd)
        pred[te] = Ridge(alpha=1.0).fit(Z[tr], Y[tr]).predict(Z[te])
    return pred


def _boot(d: pd.DataFrame, stat) -> tuple[float, list[float]]:
    rng = np.random.default_rng(SEED)
    players = d["prev_id"].astype(str).to_numpy()
    uniq = np.unique(players)
    pos = {p: np.flatnonzero(players == p) for p in uniq}
    est = stat(d)
    draws = []
    for _ in range(N_BOOT):
        idx = np.concatenate([pos[p] for p in rng.choice(uniq, len(uniq))])
        draws.append(stat(d.iloc[idx]))
    return float(est), [float(np.nanpercentile(draws, 5)), float(np.nanpercentile(draws, 95))]


def _corr(x: str, r: str):
    return lambda d: float(np.corrcoef(d[x], d[r])[0, 1]) if d[x].std() > 0 else np.nan


def analyses(df: pd.DataFrame) -> list[dict]:
    log = []

    def add(aid, desc, d, est_ci, extra=None):
        est, ci = est_ci
        log.append({"id": aid, "analysis": desc, "n": int(len(d)), "est": round(est, 4), "ci90": [round(ci[0], 4), round(ci[1], 4)],
                    "excludes_zero": bool(ci[0] > 0 or ci[1] < 0), **(extra or {})})

    for g in ("QB", "RB", "WR", "TE"):
        for out in ("A_vor", "A_rate", "B_vor"):
            d = df[(df["group"] == g) & df[out].notna()].copy()
            d["base_pred"] = _loco(d, BASE, out)
            d["resid"] = d[out] - d["base_pred"]
            q = d[d["M1_distance"].notna()]
            add(f"{g}.{out}.M1", "corr(style distance, residual); closer is better if negative", q, _boot(q, _corr("M1_distance", "resid")))
            q = d[d["M3_usage_change"].notna()]
            add(f"{g}.{out}.M3", "corr(change in positional usage, residual); more usage better if positive", q,
                _boot(q, _corr("M3_usage_change", "resid")))
            q = d[d["ty1"].notna() & d["to_D1"].notna()].copy()
            inter = ["ty1", "ty2", "to_D1", "to_D2", "ty1_x_sys1", "ty1_x_sys2", "ty2_x_sys1", "ty2_x_sys2"]
            if len(q) > 50:
                q["p_base"], q["p_fit"] = _loco(q, BASE, out), _loco(q, BASE + inter, out)

                def gain(z, out=out):
                    sse = lambda c: float(((z[out] - z[c]) ** 2).sum())  # noqa: E731
                    return (sse("p_base") - sse("p_fit")) / float(((z[out] - z[out].mean()) ** 2).sum())
                add(f"{g}.{out}.M2", "cross-validated R^2 gain from type x system terms (qualified players)", q, _boot(q, gain))
            for same in (True, False):
                q = d[d["M1_distance"].notna() & (d["same_coach"] == same)]
                if len(q) > 30:
                    add(f"{g}.{out}.M4.{'same' if same else 'new'}_coach", "M1 within destinations that kept / changed head coach",
                        q, _boot(q, _corr("M1_distance", "resid")))
    return log


def main() -> None:
    import sys
    cache = RAW / "cfb" / "fit_explore.parquet"                             # private (built from CFBD rosters)
    if "--cached" in sys.argv and cache.exists():
        df = pd.read_parquet(cache)
    else:
        df = frame()
        df.to_parquet(cache, index=False)
    log = analyses(df)
    n = len(log)
    res = {"status": "EXPLORATORY: every transfer here has been seen; nothing is a test (Phase 2 confirms on the 2026 vault)",
           "transfers": int(len(df)), "by_group": df["group"].value_counts().to_dict(),
           "outcome_rates": {o: {"n": int(df[o].notna().sum()), "positive": round(float((df[o] > 0).mean()), 4)} for o in ("A_vor", "B_vor")},
           "replacement_level_note": "per position, level, season: volume-weighted adjusted EPA/play of backups",
           "analyses_run": n, "expected_by_chance_at_90pct": round(0.10 * n, 1),
           "excluding_zero": sum(a["excludes_zero"] for a in log), "analyses": log}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "explore.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in res.items() if k != "analyses"}, indent=1, default=str))
    for a in log:
        flag = "*" if a["excludes_zero"] else " "
        print(f"{flag} {a['id']:<28} n={a['n']:<5} est={a['est']:+.4f} ci90=[{a['ci90'][0]:+.4f}, {a['ci90'][1]:+.4f}]")


if __name__ == "__main__":
    main()
