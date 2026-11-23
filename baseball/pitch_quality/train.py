"""Train the pitch-quality models and score MLB and Triple-A pitchers.

Three model families (MODELS), each one LightGBM regressor per pitch group (fastball / breaking / offspeed),
all with the same target (pitcher run value of the pitch, denoised on balls in play), the same
training seasons (every configured MLB season except the current one) and the same holdout (the
current season, out of time). Each regressor is a bag of N_BAGS models (v2.0.0): pitchers are split
into N_BAGS folds, model f trains on the other folds and early-stops on fold f, and a pitch's
prediction is the bag average. One model early-stopped on one validation slice was unstable: its
tree count swung with which pitchers landed in the slice, and per-pitcher grades moved by up to 64
points between seeds (review/model-changes.md, 2026-11-14). stability.py measures what remains. Seasons come from sparvi.toml; feature sets from features.py:
    stuff     Stuff+     what the ball does (velocity, movement, spin, release)
    pitching  Pitching+  that plus where it crossed the plate, the count and the batter's side
    no_loc    (no grade) the Pitching+ inputs without the location
Location+ (3.0.0, LOC-TYPE) is the location's value given the pitch: a pitch's Pitching+ prediction minus
its no_loc prediction, averaged over a pitcher's pitches. It passed its pre-registered backtest
(models/baseball/location_plus/card.yml); the model on location alone that it replaces did not predict.

Outputs (pitch_quality/output/):
    metrics.json                       holdout fit + stability/predictiveness checks
    pitcher_season_{level}_{year}.csv  pitcher-season Stuff+, Location+, Pitching+, RV+
    pitch_type_{level}_{year}.csv      pitcher x pitch type grades
    importance.csv, importance_{no_loc,pitching}.csv   gain importance by group
    scale_reference.json/.csv          the frozen prior-season scale for every grade
    pitch_predictions_*.parquet        pitch-level predictions (gitignored; snapshot marts only)
    models/{kind}_{group}_{f}.txt      the bagged boosters (gitignored, large; the registry keeps their hashes)
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from features import FEATURE_SETS, FEATURES, build_features

from sparvi.config import RAW, baseball
from sparvi.provenance import provenance

CFG = baseball()
SEED = 7
AAA_MIN_PITCHES = 300  # Triple-A side of the Triple-A -> MLB check
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
GROUPS = ["fastball", "breaking", "offspeed"]
N_BAGS = 10
# Stuff+ models early-stop after very few trees at 0.03 (12-100), so where they stop moves the grades;
# a smaller step spreads the same fit over ~10x the trees and makes the stopping point matter less.
# "loc" is the retired location-only model (2.0.0): not fitted here, kept for the Location+ backtest's baseline.
LEARNING_RATE = {"stuff": 0.005, "loc": 0.03, "no_loc": 0.03, "pitching": 0.03}
MODELS = ["stuff", "no_loc", "pitching"]  # the model families fitted
KINDS = ["stuff", "loc", "pitching"]      # the grades
GRADE = {"stuff": "stuff_plus", "loc": "loc_plus", "pitching": "pitching_plus"}
# Pitch-level prediction columns. xrv_loc is the published Location+ value: xrv_pitching - xrv_no_loc (predict_all).
XRV = {"stuff": "xrv", "loc": "xrv_loc", "no_loc": "xrv_no_loc", "pitching": "xrv_pitching"}

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
    # Same data, same model. LightGBM's default thread count follows the machine's cores, and the
    # order of its multithreaded histogram sums (so their rounding) can depend on it: two
    # refreshes on unchanged data moved published numbers in the second decimal. A fixed thread
    # count, row-wise histograms, and deterministic mode pin it. The refresh checks it on each run.
    deterministic=True,
    force_row_wise=True,
    num_threads=4,
)
TARGET = "rv_x"     # see features.py; "rv" = raw run value
MIN_PITCHES = 500   # pitcher-season qualifier for scaling and evaluation


def load(level: str, year: int) -> pd.DataFrame:
    return build_features(pd.read_parquet(RAW / f"statcast_{level}_{year}.parquet"))


def fit_group(train: pd.DataFrame, valid: pd.DataFrame, features: list[str] = FEATURES,
              seed: int = SEED, learning_rate: float | None = None) -> lgb.Booster:
    dtr = lgb.Dataset(train[features], train[TARGET])
    dva = lgb.Dataset(valid[features], valid[TARGET], reference=dtr)
    params = {**PARAMS, "seed": seed} | ({"learning_rate": learning_rate} if learning_rate else {})
    return lgb.train(
        params, dtr, num_boost_round=10_000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )


def bag_folds(pitchers: pd.Series, seed: int = SEED, n_bags: int = N_BAGS) -> pd.Series:
    """Fold number (0..n_bags-1) for each row, by pitcher: a seeded shuffle of the sorted pitcher ids."""
    ids = np.sort(pitchers.unique())
    fold_of = dict(zip(np.random.default_rng(seed).permutation(ids), np.arange(len(ids)) % n_bags, strict=True))
    return pitchers.map(fold_of)


def fit_kinds(tr: pd.DataFrame, kinds=MODELS, seed: int = SEED, n_bags: int = N_BAGS) -> dict:
    """{kind: {group: [n_bags boosters]}} for the model families in `kinds`. Model f of each bag trains on every fold but f and early-stops on f;
    its LightGBM seed is seed * 100 + f. Same seed, same data: same bags."""
    fold = bag_folds(tr["pitcher"], seed, n_bags)
    return {k: {g: [fit_group(tr[tr["group"].eq(g) & fold.ne(f)], tr[tr["group"].eq(g) & fold.eq(f)], FEATURE_SETS[k],
                              seed=seed * 100 + f, learning_rate=LEARNING_RATE[k]) for f in range(n_bags)]
                for g in GROUPS} for k in kinds}


def fitted(name: str, verify: bool = False) -> dict:
    """The fitted bags for a named training set, from the model store when its code and training data are unchanged and
    trained otherwise (artifacts.py, MODEL-ARTIFACTS). "production": every MLB season before the current one (the
    published grades). "oos": the season before last only (robustness.py's out-of-sample test)."""
    import artifacts
    seasons = {"production": CFG.train_seasons, "oos": [CFG.prev - 1]}[name]
    return artifacts.get(name, seasons, lambda: fit_kinds(pd.concat([load("mlb", y) for y in seasons], ignore_index=True)),
                         verify=verify)


def predict(models: dict, df: pd.DataFrame, features: list[str] = FEATURES) -> pd.Series:
    """Bag average: each booster at its own best iteration."""
    pred = pd.Series(np.nan, index=df.index)
    for g, bag in models.items():
        idx = df["group"].eq(g)
        if idx.any():
            pred[idx] = np.mean([m.predict(df.loc[idx, features], num_iteration=m.best_iteration) for m in bag], axis=0)
    return pred


def predict_all(all_models: dict, df: pd.DataFrame) -> None:
    """Add each fitted family's prediction column and, when both of its parts were fitted, the Location+ value:
    what the location adds to the pitch (full model minus the same model without location)."""
    for k, models in all_models.items():
        df[XRV[k]] = predict(models, df, FEATURE_SETS[k])
    if "pitching" in all_models and "no_loc" in all_models:
        df[XRV["loc"]] = df[XRV["pitching"]] - df[XRV["no_loc"]]


def pitcher_season(df: pd.DataFrame) -> pd.DataFrame:
    aggs = dict(
        name=("player_name", "first"),
        throws=("p_throws", "first"),
        team=("team", lambda s: s.mode().iloc[0]),
        pitches=("rv", "size"),
        rv100=("rv", lambda s: 100 * s.mean()),
    )
    for k in KINDS:
        if XRV[k] in df:
            aggs["xrv100" + ("" if k == "stuff" else f"_{k}")] = (XRV[k], lambda s: 100 * s.mean())
    return df.groupby(["pitcher", "game_year"]).agg(**aggs).reset_index()


def xrv100(k: str) -> str:
    return "xrv100" + ("" if k == "stuff" else f"_{k}")


def ref_stats(ref: pd.Series) -> dict:
    """Mean and sample SD of a reference distribution (what dbt's avg/stddev_samp reproduce)."""
    return {"mean": float(ref.mean()), "sd": float(ref.std()), "n": int(ref.size)}


def scale(values: pd.Series, ref: dict) -> pd.Series:
    """Grade: 100 = the reference (prior-season qualified MLB) average, 10 points = 1 SD."""
    return 100 + 10 * (values - ref["mean"]) / ref["sd"]


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

    # Early stopping uses the bag's held-out pitcher folds of the training data, never the holdout.
    all_models = fitted("production")
    (OUT / "models").mkdir(exist_ok=True)
    for old in (OUT / "models").glob("*.txt"):
        old.unlink()
    for k, models in all_models.items():
        imp = []
        for g, bag in models.items():
            gains = [m.feature_importance("gain", iteration=m.best_iteration) for m in bag]
            gain = np.mean([x / x.sum() for x in gains], axis=0)
            imp.append(pd.DataFrame({"group": g, "feature": FEATURE_SETS[k], "gain": gain}))
            print(f"  {k}/{g}: {[m.best_iteration for m in bag]} trees")
            for f, m in enumerate(bag):
                m.save_model(str(OUT / "models" / f"{k}_{g}_{f}.txt"), num_iteration=m.best_iteration)
        pd.concat(imp).to_csv(OUT / ("importance.csv" if k == "stuff" else f"importance_{k}.csv"), index=False)
    trees = {f"{k}/{g}": [m.best_iteration for m in bag] for k, models in all_models.items() for g, bag in models.items()}

    metrics = {"season": Y, "min_pitches": MIN_PITCHES, "aaa_min_pitches": AAA_MIN_PITCHES,
               "n_bags": N_BAGS, "learning_rate": LEARNING_RATE, "trees": trees}
    for name, df in [("train", tr), ("holdout", ho)]:
        predict_all(all_models, df)
        # Pitch-level R^2 is tiny by nature (single-pitch outcomes are mostly noise),
        # so report it but judge the model at the pitcher level.
        for k in KINDS:
            resid = df["rv"] - df[XRV[k]]
            metrics[f"{name}_pitch_r2" + ("" if k == "stuff" else f"_{k}")] = float(1 - resid.var() / df["rv"].var())

    # Scale reference: qualified MLB pitchers in the FULL PRIOR season, frozen for the current one.
    # A current-season reference would move 100 every week, so early-season grades (and the date
    # slider) would mean nothing. scale_reference.json records the values; dbt uses the CSV copy.
    ps = {(("mlb", y)): pitcher_season(tr[tr.game_year == y]) for y in CFG.train_seasons}
    ps[("mlb", Y)] = pitcher_season(ho)
    q_ref = ps[("mlb", P)].query("pitches >= @MIN_PITCHES")
    refs = {k: ref_stats(q_ref[xrv100(k)]) for k in KINDS}
    ref_rv = ref_stats(q_ref["rv100"])

    # --- Stability: how fast does each grade stabilize? (split-half, holdout season)
    metrics["split_half_r_stuff"] = split_half_r(ho, "xrv")
    metrics["split_half_r_loc"] = split_half_r(ho, "xrv_loc")
    metrics["split_half_r_pitching"] = split_half_r(ho, "xrv_pitching")
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
    for k in ["loc", "pitching"]:
        metrics[f"yoy_r_{k}_prev_to_{k}_cur"] = float(yy[f"{xrv100(k)}_prev"].corr(yy[f"{xrv100(k)}_cur"]))

    # --- Triple-A: score, then check translation to MLB the following season.
    # Triple-A side: AAA_MIN_PITCHES+ last season. MLB side: MIN_PITCHES+ this season.
    aaa = {}
    for yr in CFG.aaa_seasons:
        d = load("aaa", yr)
        predict_all(all_models, d)
        aaa[yr] = d
        ps[("aaa", yr)] = pitcher_season(d)
    up = ps[("aaa", P)].query("pitches >= @AAA_MIN_PITCHES").merge(b, on="pitcher", suffixes=("_aaa", "_mlb"))
    metrics["aaa_to_mlb_n"] = int(len(up))
    metrics["aaa_to_mlb_r_stuff"] = float(up["xrv100_aaa"].corr(up["rv100_mlb"]))
    metrics["aaa_to_mlb_r_actual"] = float(up["rv100_aaa"].corr(up["rv100_mlb"]))
    for k in ["loc", "pitching"]:
        metrics[f"aaa_to_mlb_r_{k}"] = float(up[f"{xrv100(k)}_aaa"].corr(up["rv100_mlb"]))
    q_aaa = ps[("aaa", Y)].query("pitches >= @MIN_PITCHES")
    q_cur = ps[("mlb", Y)].query("pitches >= @MIN_PITCHES")
    metrics["aaa_mean_stuff_plus_qualified"] = float(scale(q_aaa["xrv100"], refs["stuff"]).mean())
    metrics["mlb_mean_stuff_plus_qualified"] = float(scale(q_cur["xrv100"], refs["stuff"]).mean())
    for k in ["loc", "pitching"]:
        metrics[f"aaa_mean_{k}_plus_qualified"] = float(scale(q_aaa[xrv100(k)], refs[k]).mean())
        metrics[f"mlb_mean_{k}_plus_qualified"] = float(scale(q_cur[xrv100(k)], refs[k]).mean())
    metrics["n_pitches_mlb"] = int(len(tr) + len(ho))
    metrics["n_pitches_aaa"] = int(sum(len(d) for d in aaa.values()))

    # Unrounded on purpose: the snapshot marts must reproduce these exactly. The export rounds.
    for (lvl, yr), p in ps.items():
        for k in KINDS:
            p[GRADE[k]] = scale(p[xrv100(k)], refs[k])
        p["rv_plus"] = scale(p["rv100"], ref_rv)
        p.to_csv(OUT / f"pitcher_season_{lvl}_{yr}.csv", index=False)

    # Pitch-type grades (current season, both levels), scaled within pitch group against the
    # prior season's MLB pitch types (100+ thrown), separately for each grade.
    prior = tr[tr.game_year == P]
    ref_pt = {}
    for k in KINDS:
        pp = prior.groupby(["pitcher", "pitch_type", "group"]).agg(
            n=(XRV[k], "size"), v=(XRV[k], lambda s: 100 * s.mean())).reset_index()
        ref_pt[k] = {g: ref_stats(pp.query("group == @g and n >= 100")["v"]) for g in GROUPS}
    for lvl, d in [("mlb", ho), ("aaa", aaa[Y])]:
        aggs = dict(n=("xrv", "size"), velo=("velo", "mean"), ivb=("ivb", "mean"), hb=("hb", "mean"),
                    team=("team", lambda s: s.mode().iloc[0]))
        for k in KINDS:
            aggs[xrv100(k)] = (XRV[k], lambda s: 100 * s.mean())
        pt = d.groupby(["pitcher", "player_name", "pitch_type", "group"]).agg(**aggs).reset_index()
        for k in KINDS:
            pt[GRADE[k]] = np.nan
            for g in GROUPS:
                idx = pt["group"].eq(g)
                pt.loc[idx, GRADE[k]] = scale(pt.loc[idx, xrv100(k)], ref_pt[k][g])
        pt.to_csv(OUT / f"pitch_type_{lvl}_{Y}.csv", index=False)

    scale_ref = {"reference_season": P, "applies_to_season": Y, "min_pitches": MIN_PITCHES, "min_pitch_type": 100,
                 "stuff": refs["stuff"], "loc": refs["loc"], "pitching": refs["pitching"], "rv": ref_rv,
                 "pitch_type": ref_pt["stuff"], "pitch_type_loc": ref_pt["loc"], "pitch_type_pitching": ref_pt["pitching"]}
    (OUT / "scale_reference.json").write_text(json.dumps(scale_ref, indent=2), encoding="utf-8")
    rows = [{"kind": k, "grp": "all", **refs[k]} for k in KINDS] + [{"kind": "rv", "grp": "all", **ref_rv}]
    rows += [{"kind": "pitch_type" + ("" if k == "stuff" else f"_{k}"), "grp": g, **r} for k in KINDS for g, r in ref_pt[k].items()]
    pd.DataFrame(rows).to_csv(OUT / "scale_reference.csv", index=False, float_format="%.17g")

    # Pitch-level predictions for the snapshot marts. Internal only: gitignored, never exported.
    cols = ["game_pk", "game_date", "at_bat_number", "pitch_number", "pitcher", "pitch_type", "group",
            "xrv", "xrv_loc", "xrv_pitching", "rv"]
    for lvl, yr, d in ([("mlb", y, tr[tr.game_year == y]) for y in CFG.train_seasons] + [("mlb", Y, ho)]
                       + [("aaa", y, aaa[y]) for y in CFG.aaa_seasons]):
        d[cols].rename(columns={"xrv": "xrv_stuff"}).to_parquet(OUT / f"pitch_predictions_{lvl}_{yr}.parquet", index=False)

    files = [RAW / f"statcast_mlb_{y}.parquet" for y in CFG.mlb_seasons] + \
            [RAW / f"statcast_aaa_{y}.parquet" for y in CFG.aaa_seasons]
    metrics["provenance"] = provenance(files, seeds={"lightgbm": f"{SEED} * 100 + bag member", "bag_folds": SEED})
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "provenance"}, indent=2))


if __name__ == "__main__":
    main()
