"""Train the pitch-quality model and score MLB and Triple-A pitchers.

Train: every configured MLB season except the current one. Holdout: the current
season (out of time). Seasons come from sparvi.toml.
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

from sparvi.config import RAW, baseball
from sparvi.provenance import provenance

CFG = baseball()
SEED = 7
AAA_MIN_PITCHES = 300  # Triple-A side of the Triple-A -> MLB check
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
    seed=SEED,
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
    Y, P = CFG.season, CFG.prev
    print(f"loading MLB {CFG.train_seasons} (train) and {Y} (holdout)...")
    tr = pd.concat([load("mlb", y) for y in CFG.train_seasons], ignore_index=True)
    ho = load("mlb", Y)

    # Early stopping uses a pitcher-grouped 15% slice of training data, never the holdout.
    rng = np.random.default_rng(SEED)
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

    metrics = {"season": Y, "min_pitches": MIN_PITCHES, "aaa_min_pitches": AAA_MIN_PITCHES}
    for name, df in [("train", tr), ("holdout", ho)]:
        df["xrv"] = predict(models, df)
        # Pitch-level R^2 is tiny by nature (single-pitch outcomes are mostly noise),
        # so report it but judge the model at the pitcher level.
        resid = df["rv"] - df["xrv"]
        metrics[f"{name}_pitch_r2"] = float(1 - resid.var() / df["rv"].var())

    # Reference distribution for scaling: qualified MLB pitcher-seasons in the current season.
    ps = {(("mlb", y)): pitcher_season(tr[tr.game_year == y]) for y in CFG.train_seasons}
    ps[("mlb", Y)] = pitcher_season(ho)
    ref = ps[("mlb", Y)].query("pitches >= @MIN_PITCHES")["xrv100"]
    ref_rv = ps[("mlb", Y)].query("pitches >= @MIN_PITCHES")["rv100"]

    # --- Stability: how fast does each metric stabilize? (split-half, holdout season)
    metrics["split_half_r_stuff"] = split_half_r(ho, "xrv")
    metrics["split_half_r_actual_rv"] = split_half_r(ho, "rv")

    # --- Predictiveness: does last season's Stuff+ predict this season's RV/100 better than
    # last season's RV/100? (Last season is in-sample here; robustness.py has the clean version.)
    a = ps[("mlb", P)].query("pitches >= @MIN_PITCHES")
    b = ps[("mlb", Y)].query("pitches >= @MIN_PITCHES")
    yy = a.merge(b, on="pitcher", suffixes=("_prev", "_cur"))
    metrics["yoy_n"] = int(len(yy))
    metrics["yoy_r_stuff_prev_to_rv_cur"] = float(yy["xrv100_prev"].corr(yy["rv100_cur"]))
    metrics["yoy_r_rv_prev_to_rv_cur"] = float(yy["rv100_prev"].corr(yy["rv100_cur"]))
    metrics["yoy_r_stuff_prev_to_stuff_cur"] = float(yy["xrv100_prev"].corr(yy["xrv100_cur"]))

    # --- Triple-A: score, then check translation to MLB the following season.
    # Triple-A side: AAA_MIN_PITCHES+ last season. MLB side: MIN_PITCHES+ this season.
    aaa = {}
    for yr in CFG.aaa_seasons:
        d = load("aaa", yr)
        d["xrv"] = predict(models, d)
        aaa[yr] = d
        ps[("aaa", yr)] = pitcher_season(d)
    up = ps[("aaa", P)].query("pitches >= @AAA_MIN_PITCHES").merge(b, on="pitcher", suffixes=("_aaa", "_mlb"))
    metrics["aaa_to_mlb_n"] = int(len(up))
    metrics["aaa_to_mlb_r_stuff"] = float(up["xrv100_aaa"].corr(up["rv100_mlb"]))
    metrics["aaa_to_mlb_r_actual"] = float(up["rv100_aaa"].corr(up["rv100_mlb"]))
    q_aaa = ps[("aaa", Y)].query("pitches >= @MIN_PITCHES")
    metrics["aaa_mean_stuff_plus_qualified"] = float(scale(q_aaa["xrv100"], ref).mean())
    metrics["n_pitches_mlb"] = int(len(tr) + len(ho))
    metrics["n_pitches_aaa"] = int(sum(len(d) for d in aaa.values()))

    for (lvl, yr), p in ps.items():
        p["stuff_plus"] = scale(p["xrv100"], ref).round(1)
        p["rv_plus"] = scale(p["rv100"], ref_rv).round(1)
        p.to_csv(OUT / f"pitcher_season_{lvl}_{yr}.csv", index=False)

    # Pitch-type grades (current season, both levels), scaled within pitch group vs MLB.
    ho_pt = ho.groupby(["pitcher", "pitch_type", "group"]).agg(
        n=("xrv", "size"), xrv100=("xrv", lambda s: 100 * s.mean())).reset_index()
    for lvl, d in [("mlb", ho), ("aaa", aaa[Y])]:
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
        pt.to_csv(OUT / f"pitch_type_{lvl}_{Y}.csv", index=False)

    files = [RAW / f"statcast_mlb_{y}.parquet" for y in CFG.mlb_seasons] +             [RAW / f"statcast_aaa_{y}.parquet" for y in CFG.aaa_seasons]
    metrics["provenance"] = provenance(files, seeds={"lightgbm": SEED, "validation_split": SEED})
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
