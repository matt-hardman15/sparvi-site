"""Which Triple-A numbers travel to the majors, and by how much do they change?

Cohorts (lines from fct_translation_lines; run `dbt build` first):
  across   last season's Triple-A line (>= 150 PA / BF) -> this season's MLB line (>= 100)
  within   for each of the two seasons, a player's Triple-A line before his first MLB game
           that season (>= 150) -> his MLB line from that game on (>= 100)
  pooled   both (a player can appear once in each); every pair is weighted by the smaller of its
           two samples, and resampling is by player so a player in both cohorts counts once.
Pitchers carry their MLB role (SP/RP/Swing) for the role split.

Analyses, each per side and cohort (pitchers also by role):
  1. carryover: weighted correlation of a Triple-A metric with the same metric in MLB
  2. level adjustment: weighted median of (MLB - Triple-A), with a bootstrap 90% interval
  3. what predicts MLB production (hitters: MLB xwOBA; pitchers: MLB RV+): each Triple-A metric's
     correlation with it, and a ridge regression on all of them, cross-validated by player,
     against Triple-A results alone (wOBA for hitters, RV+ for pitchers)
  4. projection: every Triple-A player over the minimum this season, translated metric by
     metric (the weighted MLB-on-Triple-A line from the pooled cohort), each with an 80% band
     from the cohort's residuals around that line

Writes output/results.json (everything above), output/pairs.csv (the cohort lines behind the
explorer), output/projections.csv (current Triple-A players).
"""
import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from sparvi.config import RAW, baseball
from sparvi.provenance import provenance

CFG = baseball()
Y, P = CFG.season, CFG.prev
HERE = Path(__file__).resolve().parent
PD = HERE.parent / "pd_pipeline"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

MIN_AAA, MIN_MLB = 150, 100
SEED, BOOT, CV_REPEATS, CV_FOLDS = 20261125, 2000, 20, 5
ALPHAS = [0.1, 1.0, 10.0, 100.0]
BAND = (10, 90)  # the projection band: 80% of the cohort landed inside it

METRICS = {
    "hitter": ["chase_pct", "zone_swing_pct", "zone_contact_pct", "whiff_pct", "k_pct", "bb_pct",
               "hard_hit_pct", "ev90", "xwoba", "woba"],
    "pitcher": ["stuff_plus", "loc_plus", "pitching_plus", "k_pct", "bb_pct", "csw_pct", "whiff_pct",
                "xwoba", "rv_plus"],
}
OUTCOME = {"hitter": "xwoba", "pitcher": "rv_plus"}    # what "MLB production" means per side
BASELINE = {"hitter": "woba", "pitcher": "rv_plus"}     # Triple-A results alone

os.chdir(PD)
con = duckdb.connect("pd.duckdb", read_only=True, config={"threads": 1})
lines = con.sql("select * from fct_translation_lines").df()
con.close()

names = pd.concat([pd.read_parquet(RAW / f"milb_players_{s}.parquet", columns=["player_id", "name"]).assign(s=s)
                   for s in (P, Y)]).sort_values(["player_id", "s"]).drop_duplicates("player_id", keep="last")
NAME = names.set_index("player_id")["name"]


def pairs(side: str) -> pd.DataFrame:
    keep = ["player_id", "season", "team", "pa", "role", *METRICS[side]]
    L = lines[lines["side"] == side]
    aaa_full = L[(L.level == "AAA") & (L.segment == "full") & (L.pa >= MIN_AAA)][keep]
    aaa_pre = L[(L.level == "AAA") & (L.segment == "pre_debut") & (L.pa >= MIN_AAA)][keep]
    mlb = L[(L.level == "MLB") & (L.segment == "full") & (L.pa >= MIN_MLB)][keep]
    across = aaa_full[aaa_full.season == P].merge(mlb[mlb.season == Y], on="player_id", suffixes=("_aaa", "_mlb"))
    across["cohort"] = "across"
    within = aaa_pre.merge(mlb, on=["player_id", "season"], suffixes=("_aaa", "_mlb"))
    within["season_aaa"] = within["season_mlb"] = within["season"]
    within = within.drop(columns="season").assign(cohort="within")
    df = pd.concat([across, within], ignore_index=True)
    df["w"] = np.minimum(df["pa_aaa"], df["pa_mlb"]).astype(float)
    df["name"] = df["player_id"].map(NAME)
    df["role"] = df["role_mlb"]
    return df.sort_values(["cohort", "season_mlb", "player_id"]).reset_index(drop=True)


def wcorr(x, y, w) -> float:
    m = ~(np.isnan(x) | np.isnan(y))
    x, y, w = x[m], y[m], w[m]
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return float(cov / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w)))


def wmedian(v, w) -> float:
    m = ~np.isnan(v)
    v, w = v[m], w[m]
    o = np.argsort(v, kind="stable")
    c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, 0.5 * c[-1])])


def wquantile(v, w, q) -> float:
    o = np.argsort(v, kind="stable")
    c = np.cumsum(w[o]) / w.sum()
    return float(v[o][min(np.searchsorted(c, q), len(v) - 1)])


def boot_index(df: pd.DataFrame, rng) -> list[np.ndarray]:
    """Row indices for each bootstrap draw, resampling players (a player's rows travel together)."""
    groups = df.groupby("player_id").indices
    ids = np.array(sorted(groups))
    out = []
    for _ in range(BOOT):
        pick = rng.choice(ids, size=len(ids), replace=True)
        out.append(np.concatenate([groups[i] for i in pick]))
    return out


def ci(v) -> list[float]:
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    return [float(np.percentile(v, 5)), float(np.percentile(v, 95))]


def summarize(df: pd.DataFrame, side: str, rng) -> dict:
    """Carryover, level adjustment, and outcome correlations for one slice of pairs."""
    idx = boot_index(df, rng)
    w = df["w"].to_numpy()
    out_mlb = df[f"{OUTCOME[side]}_mlb"].to_numpy(dtype=float)
    res = {"n": int(len(df)), "players": int(df["player_id"].nunique()), "metrics": {}}
    for m in METRICS[side]:
        a, b = df[f"{m}_aaa"].to_numpy(dtype=float), df[f"{m}_mlb"].to_numpy(dtype=float)
        d = b - a
        r_b, adj_b, o_b = [], [], []
        for ix in idx:
            r_b.append(wcorr(a[ix], b[ix], w[ix]))
            adj_b.append(wmedian(d[ix], w[ix]))
            o_b.append(wcorr(a[ix], out_mlb[ix], w[ix]))
        res["metrics"][m] = {
            "n": int((~(np.isnan(a) | np.isnan(b))).sum()),
            "aaa_mean": float(np.nanmean(a)), "mlb_mean": float(np.nanmean(b)),
            "carryover_r": wcorr(a, b, w), "carryover_ci90": ci(r_b),
            "adjustment": wmedian(d, w), "adjustment_ci90": ci(adj_b),
            "r_with_mlb_outcome": wcorr(a, out_mlb, w), "r_with_mlb_outcome_ci90": ci(o_b),
        }
    return res


def cv_r2(df: pd.DataFrame, side: str, feats: list[str]) -> dict:
    """Out-of-fold R² of a ridge on `feats`, folds grouped by player, repeated with shuffled groups."""
    X = df[[f"{m}_aaa" for m in feats]].to_numpy(dtype=float)
    y = df[f"{OUTCOME[side]}_mlb"].to_numpy(dtype=float)
    w = df["w"].to_numpy()
    ok = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y, w, g = X[ok], y[ok], w[ok], df["player_id"].to_numpy()[ok]
    r2s, oof_all = [], []
    rng = np.random.default_rng(SEED)
    for _ in range(CV_REPEATS):
        perm = {pid: k for k, pid in enumerate(rng.permutation(np.unique(g)))}
        groups = np.array([perm[x] for x in g])
        oof = np.empty_like(y)
        for tr, te in GroupKFold(n_splits=CV_FOLDS).split(X, y, groups):
            model = fit_ridge(X[tr], y[tr], w[tr], groups[tr])
            oof[te] = model.predict(X[te])
        r2s.append(1 - np.average((y - oof) ** 2, weights=w) / np.average((y - np.average(y, weights=w)) ** 2, weights=w))
        oof_all.append(y - oof)
    resid = np.concatenate(oof_all)
    wr = np.tile(w, CV_REPEATS)
    return {"features": feats, "n": int(ok.sum()), "r2": float(np.mean(r2s)), "r2_range": [float(min(r2s)), float(max(r2s))],
            "band": [wquantile(resid, wr, BAND[0] / 100), wquantile(resid, wr, BAND[1] / 100)]}


class _Scaled:
    def __init__(self, scaler, ridge):
        self.scaler, self.ridge = scaler, ridge

    def predict(self, X):
        return self.ridge.predict(self.scaler.transform(X))


def fit_ridge(X, y, w, groups) -> _Scaled:
    """Standardize, pick alpha by an inner grouped CV, refit on everything given."""
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    if X.shape[1] == 1:
        return _Scaled(sc, Ridge(alpha=0.0).fit(Xs, y, sample_weight=w))
    best, best_err = ALPHAS[0], np.inf
    for a in ALPHAS:
        err = 0.0
        for tr, te in GroupKFold(n_splits=CV_FOLDS).split(Xs, y, groups):
            p = Ridge(alpha=a).fit(Xs[tr], y[tr], sample_weight=w[tr]).predict(Xs[te])
            err += float(np.sum(w[te] * (y[te] - p) ** 2))
        if err < best_err:
            best, best_err = a, err
    return _Scaled(sc, Ridge(alpha=best).fit(Xs, y, sample_weight=w))


def line_fit(df: pd.DataFrame, m: str) -> dict:
    """Weighted MLB-on-Triple-A line for one metric, and the 80% residual band around it."""
    a, b, w = (df[f"{m}_aaa"].to_numpy(dtype=float), df[f"{m}_mlb"].to_numpy(dtype=float), df["w"].to_numpy())
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b, w = a[ok], b[ok], w[ok]
    slope, icpt = np.polyfit(a, b, 1, w=np.sqrt(w))
    resid = b - (icpt + slope * a)
    return {"slope": float(slope), "intercept": float(icpt),
            "band": [wquantile(resid, w, BAND[0] / 100), wquantile(resid, w, BAND[1] / 100)]}


results = {"seasons": {"aaa_across": P, "mlb_across": Y, "within": [P, Y]},
           "min_aaa": MIN_AAA, "min_mlb": MIN_MLB, "bootstrap_samples": BOOT, "band_pct": list(BAND),
           "cv": {"folds": CV_FOLDS, "repeats": CV_REPEATS, "alphas": ALPHAS},
           "outcome": OUTCOME, "baseline": BASELINE}
all_pairs, all_proj = [], []
for side in ("hitter", "pitcher"):
    df = pairs(side)
    rng = np.random.default_rng(SEED)
    S = {"cohorts": {}}
    for cohort, sub in [("pooled", df), ("across", df[df.cohort == "across"]), ("within", df[df.cohort == "within"])]:
        S["cohorts"][cohort] = summarize(sub.reset_index(drop=True), side, rng)
    if side == "pitcher":
        S["roles"] = {r: summarize(df[df.role == r].reset_index(drop=True), side, rng) for r in ("SP", "RP")}
        S["roles"]["Swing_n"] = int((df.role == "Swing").sum())
    S["model"] = {"all": cv_r2(df, side, METRICS[side]), "results_only": cv_r2(df, side, [BASELINE[side]])}
    S["model_single"] = {m: cv_r2(df, side, [m])["r2"] for m in METRICS[side]}
    S["fits"] = {m: line_fit(df, m) for m in METRICS[side]}

    # Projections: every current Triple-A player over the minimum.
    L = lines[(lines.side == side) & (lines.level == "AAA") & (lines.segment == "full") & (lines.season == Y)
              & (lines.pa >= MIN_AAA)].copy()
    # Metric by metric: the combined model didn't beat the best single metric (see S["model"]), so the
    # projection is each metric's own line, the same one the site applies to any Triple-A line.
    proj = L[["player_id", "team", "pa", "role", *METRICS[side]]].copy()
    for m in METRICS[side]:
        f = S["fits"][m]
        proj[f"t_{m}"] = f["intercept"] + f["slope"] * proj[m]
        proj[f"t_{m}_lo"], proj[f"t_{m}_hi"] = proj[f"t_{m}"] + f["band"][0], proj[f"t_{m}"] + f["band"][1]
    proj["side"] = side
    all_proj.append(proj.sort_values("player_id"))
    S["projected_players"] = int(len(proj))
    results[side] = S
    all_pairs.append(df.assign(side=side))

results["provenance"] = provenance(sorted(RAW.glob("statcast_*_*.parquet")), seeds={"bootstrap_and_cv": SEED})
(OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
pd.concat(all_pairs, ignore_index=True).to_csv(OUT / "pairs.csv", index=False)
pd.concat(all_proj, ignore_index=True).to_csv(OUT / "projections.csv", index=False)

for side in ("hitter", "pitcher"):
    S = results[side]
    print(f"\n== {side}: pooled n={S['cohorts']['pooled']['n']} (across {S['cohorts']['across']['n']}, "
          f"within {S['cohorts']['within']['n']}), projected {S['projected_players']}")
    for m, v in S["cohorts"]["pooled"]["metrics"].items():
        print(f"  {m:16s} r={v['carryover_r']:.2f} [{v['carryover_ci90'][0]:.2f},{v['carryover_ci90'][1]:.2f}]  "
              f"adj={v['adjustment']:+.4f} [{v['adjustment_ci90'][0]:+.4f},{v['adjustment_ci90'][1]:+.4f}]  "
              f"r_out={v['r_with_mlb_outcome']:.2f} [{v['r_with_mlb_outcome_ci90'][0]:.2f},{v['r_with_mlb_outcome_ci90'][1]:.2f}]")
    for k, v in S["model"].items():
        print(f"  model {k}: CV R2 {v['r2']:.3f} range {v['r2_range'][0]:.3f}..{v['r2_range'][1]:.3f} n={v['n']}")
    if side == "pitcher":
        for r in ("SP", "RP"):
            print(f"  role {r} n={S['roles'][r]['n']}: " + ", ".join(
                f"{m} r={v['carryover_r']:.2f} adj={v['adjustment']:+.3f}" for m, v in S["roles"][r]["metrics"].items()))
