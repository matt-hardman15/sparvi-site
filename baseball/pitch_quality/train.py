"""Train the pitch-quality model and score MLB and Triple-A pitchers.

Train: MLB 2024-2025. Holdout: MLB 2026 (out of time).
One LightGBM regressor per pitch group (fastball / breaking / offspeed),
target = pitcher run value of the pitch (-delta_run_exp).

Outputs (pitch-quality/output/):
    metrics.json                     holdout fit + stability/predictiveness checks
    pitcher_season_{level}_{year}.csv  pitcher-season Stuff+ and actual RV/100
    pitch_type_{level}_{year}.csv      pitcher x pitch type Stuff+
    importance.csv                   gain importance by group
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import FEATURES, build_features

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

PARAMS = dict(
    objective="regression",
    learning_rate=0.03,
    num_leaves=63,
    min_data_in_leaf=500,      # run value is noisy; force wide leaves
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=10.0,
    verbose=-1,
    seed=7,
)
TARGET = "rv_x"     # see features.py; "rv" = raw run value
MIN_PITCHES = 500   # pitcher-season qualifier for scaling and evaluation


def load(level: str, year: int) -> pd.DataFrame:
    return build_features(pd.read_parquet(RAW / f"statcast_{level}_{year}.parquet"))


def fit_group(train: pd.DataFrame, valid: pd.DataFrame) -> lgb.Booster:
    dtr = lgb.Dataset(train[FEATURES], train[TARGET])
    dva = lgb.Dataset(valid[FEATURES], valid[TARGET], reference=dtr)
    return lgb.train(
        PARAMS, dtr, num_boost_round=2000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )


def predict(models: dict, df: pd.DataFrame) -> pd.Series:
    pred = pd.Series(np.nan, index=df.index)
    for g, m in models.items():
        idx = df["group"].eq(g)
        pred[idx] = m.predict(df.loc[idx, FEATURES], num_iteration=m.best_iteration)
    return pred


def pitcher_season(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["pitcher", "game_year"]).agg(
        name=("player_name", "first"),
        throws=("p_throws", "first"),
        team=("team", lambda s: s.mode().iloc[0]),
        pitches=("rv", "size"),
        xrv100=("xrv", lambda s: 100 * s.mean()),
        rv100=("rv", lambda s: 100 * s.mean()),
    ).reset_index()
    return agg


def scale(values: pd.Series, ref: pd.Series) -> pd.Series:
    """Stuff+ : 100 = MLB-average qualified pitcher, 10 points = 1 SD."""
    return 100 + 10 * (values - ref.mean()) / ref.std()


def split_half_r(df: pd.DataFrame, col: str) -> float:
    """Odd/even pitch split within pitcher-season, correlate halves."""
    d = df.sort_values(["pitcher", "game_date", "at_bat_number", "pitch_number"]).copy()
    d["half"] = d.groupby("pitcher").cumcount() % 2
    h = d.groupby(["pitcher", "half"])[col].mean().unstack().dropna()
    n = d.groupby("pitcher").size()
    h = h[h.index.isin(n[n >= MIN_PITCHES].index)]
    return float(h[0].corr(h[1]))


def main() -> None:
    print("loading MLB...")
    tr = pd.concat([load("mlb", 2024), load("mlb", 2025)], ignore_index=True)
    ho = load("mlb", 2026)

    # Early stopping uses a pitcher-grouped 15% slice of training data, never the holdout.
    rng = np.random.default_rng(7)
    pitchers = tr["pitcher"].unique()
    val_p = set(rng.choice(pitchers, size=int(0.15 * len(pitchers)), replace=False))
    is_val = tr["pitcher"].isin(val_p)

    models, importance = {}, []
    for g in ["fastball", "breaking", "offspeed"]:
        t = tr[tr["group"].eq(g) & ~is_val]
        v = tr[tr["group"].eq(g) & is_val]
        m = fit_group(t, v)
        models[g] = m
        importance.append(pd.DataFrame({
            "group": g, "feature": FEATURES,
            "gain": m.feature_importance("gain") / m.feature_importance("gain").sum(),
        }))
        print(f"  {g}: {m.best_iteration} trees")
        m.save_model(str(OUT / f"model_{g}.txt"), num_iteration=m.best_iteration)
    pd.concat(importance).to_csv(OUT / "importance.csv", index=False)

    metrics = {}
    for name, df in [("train", tr), ("holdout_2026", ho)]:
        df["xrv"] = predict(models, df)
        # Pitch-level R^2 is tiny by nature (single-pitch outcomes are mostly noise),
        # so report it but judge the model at the pitcher level.
        resid = df["rv"] - df["xrv"]
        metrics[f"{name}_pitch_r2"] = float(1 - resid.var() / df["rv"].var())

    # Reference distribution for scaling: qualified MLB 2026 pitcher-seasons.
    ps = {}
    for yr, df in [(2024, tr[tr.game_year == 2024]), (2025, tr[tr.game_year == 2025]), (2026, ho)]:
        ps[("mlb", yr)] = pitcher_season(df)
    ref = ps[("mlb", 2026)].query("pitches >= @MIN_PITCHES")["xrv100"]
    ref_rv = ps[("mlb", 2026)].query("pitches >= @MIN_PITCHES")["rv100"]

    # --- Stability: how fast does each metric stabilize? (split-half, 2026 holdout)
    metrics["split_half_r_stuff_2026"] = split_half_r(ho, "xrv")
    metrics["split_half_r_actual_rv_2026"] = split_half_r(ho, "rv")

    # --- Predictiveness: does 2025 Stuff+ predict 2026 RV/100 better than 2025 RV/100?
    a = ps[("mlb", 2025)].query("pitches >= @MIN_PITCHES")
    b = ps[("mlb", 2026)].query("pitches >= @MIN_PITCHES")
    yy = a.merge(b, on="pitcher", suffixes=("_25", "_26"))
    metrics["yoy_n"] = int(len(yy))
    metrics["yoy_r_stuff25_to_rv26"] = float(yy["xrv100_25"].corr(yy["rv100_26"]))
    metrics["yoy_r_rv25_to_rv26"] = float(yy["rv100_25"].corr(yy["rv100_26"]))
    metrics["yoy_r_stuff25_to_stuff26"] = float(yy["xrv100_25"].corr(yy["xrv100_26"]))
    # Note: 2025 is in-sample for the model; the 2026 side is fully out of sample.

    # --- Triple-A: score, then check translation to MLB the following season.
    aaa = {}
    for yr in [2025, 2026]:
        d = load("aaa", yr)
        d["xrv"] = predict(models, d)
        aaa[yr] = d
        ps[("aaa", yr)] = pitcher_season(d)
    up = (
        ps[("aaa", 2025)].query("pitches >= 300")
        .merge(b.query("pitches >= 300"), on="pitcher", suffixes=("_aaa25", "_mlb26"))
    )
    metrics["aaa_to_mlb_n"] = int(len(up))
    metrics["aaa_to_mlb_r_stuff"] = float(up["xrv100_aaa25"].corr(up["rv100_mlb26"]))
    metrics["aaa_to_mlb_r_actual"] = float(up["rv100_aaa25"].corr(up["rv100_mlb26"]))

    for (lvl, yr), p in ps.items():
        p["stuff_plus"] = scale(p["xrv100"], ref).round(1)
        p["rv_plus"] = scale(p["rv100"], ref_rv).round(1)
        p.to_csv(OUT / f"pitcher_season_{lvl}_{yr}.csv", index=False)

    # Pitch-type grades (2026, both levels), scaled within pitch group vs MLB.
    ho_pt = ho.groupby(["pitcher", "pitch_type", "group"]).agg(
        n=("xrv", "size"), xrv100=("xrv", lambda s: 100 * s.mean())).reset_index()
    for lvl, d in [("mlb", ho), ("aaa", aaa[2026])]:
        pt = d.groupby(["pitcher", "player_name", "pitch_type", "group"]).agg(
            n=("xrv", "size"),
            velo=("velo", "mean"), ivb=("ivb", "mean"), hb=("hb", "mean"),
            xrv100=("xrv", lambda s: 100 * s.mean()),
            team=("team", lambda s: s.mode().iloc[0]),
        ).reset_index()
        pt["stuff_plus"] = np.nan
        for g in ["fastball", "breaking", "offspeed"]:
            r = ho_pt.query("group == @g and n >= 100")["xrv100"]
            idx = pt["group"].eq(g)
            pt.loc[idx, "stuff_plus"] = scale(pt.loc[idx, "xrv100"], r).round(1)
        pt.to_csv(OUT / f"pitch_type_{lvl}_2026.csv", index=False)

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
