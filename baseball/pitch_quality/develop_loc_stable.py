"""Development folds for Location+'s seed sensitivity (LOC-STABLE): where does the noise come from, and which
training change removes it without costing what the grade predicts?

DEVELOPMENT ONLY, on the folds of develop_loc.py (outcome seasons before the current one; nothing here reads the
current season at all, not even its pitches). Location+ is the full model's prediction minus the same model's without
location, so a pitcher's grade carries the seed noise of two models. For each training setup below and for two seeds
(the published 7 and 8: other pitcher folds, other LightGBM seeds), both models are fitted on each fold's two training
seasons and grade the season after.

Per setup, per fold and averaged over folds:
    points            mean absolute difference between the two seeds' grades, in Location+ points (10 points = one SD
                      of the qualified pitchers' location value in that fold), with the max and the shares over 1 and 2
    parts             the same for each model's prediction on its own, in the same points, and the correlation of the
                      two models' seed differences (shared noise cancels in the grade; independent noise adds)
    r_next_rv         correlation with next-season RV/100 (each seed, and their mean): the reason the grade ships
    trees             boosting rounds at the early stop, min and max over the bag, per model and pitch group

Setups: (learning rate full model, learning rate no-location model, bag members). "published" is 0.03 / 0.03 / 10.

    python baseball/pitch_quality/develop_loc_stable.py published            # one or more setup names
    python baseball/pitch_quality/develop_loc_stable.py                      # all of them
    python baseball/pitch_quality/develop_loc_stable.py gate both_slow_bags_20   # the pass gate's statistic for a setup

`gate` is for the setup chosen as a candidate, before its test is registered: on each development fold, at the seed
the test will use, its r with next-season RV/100 minus the r of the registered gate's baseline (the location-only
features of Location+ 2.0.0, trained on the same fold), with the harness's paired bootstrap over pitchers. Reported
beside it: the same against the published setup, and the full model on its own (it is Pitching+) under both setups.

Writes output/development_location_stability.json (setups not yet run are left out).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from develop import BOOT, CACHE, CFG, OUT, _code_key, _pitches, folds
from develop_loc import NO_LOC, WINDOW
from develop_loc_gates import _edge
from features import LOC_FEATURES, PITCHING_FEATURES
from train import GROUPS, MIN_PITCHES, SEED, bag_folds, fit_group, fit_kinds, pitcher_season, predict  # noqa: F401  (MIN_PITCHES: query strings)

from backtest.asof import Loader

SEEDS = [SEED, SEED + 1]
MIN_STABLE = 300                            # as stability.py: pitchers with 300+ pitches in the graded season
FEATS = {"full": PITCHING_FEATURES, "no_loc": NO_LOC}
SETUPS = {                                  # name: (lr full, lr no_loc, bag members)
    "published": (0.03, 0.03, 10),
    "no_loc_slow": (0.03, 0.005, 10),
    "both_slow": (0.005, 0.005, 10),
    "bags_20": (0.03, 0.03, 20),
    "both_slow_bags_20": (0.005, 0.005, 20),
}


def _fit(tr: pd.DataFrame, feats: list[str], lr: float, n_bags: int, seed: int) -> dict:
    """{group: bag}, trained as train.fit_kinds trains each family, with this setup's rate, bag size and seed."""
    fold = bag_folds(tr["pitcher"], seed, n_bags)
    return {g: [fit_group(tr[tr["group"].eq(g) & fold.ne(f)], tr[tr["group"].eq(g) & fold.eq(f)], feats,
                          seed=seed * 100 + f, learning_rate=lr) for f in range(n_bags)] for g in GROUPS}


def _predictions(loader: Loader, t: int, seasons: list[int], heights: pd.Series, setup: str) -> tuple[pd.DataFrame, dict]:
    """The graded season's pitches with p_{model}_{seed} for both models and seeds, and the trees each bag used."""
    lr_full, lr_no, n_bags = SETUPS[setup]
    d = _pitches(loader, t - 1, f"{t}-03-01", heights)
    tr, trees = None, {}
    for model, lr in (("full", lr_full), ("no_loc", lr_no)):
        for seed in SEEDS:
            # Keyed by what the fit depends on, so setups that share a model's rate and bag size share its fit.
            tag = f"locst_{t}_{'-'.join(map(str, seasons))}_{model}_lr{lr}_b{n_bags}_s{seed}_{_code_key()}"
            f_p, f_t = CACHE / f"{tag}.parquet", CACHE / f"{tag}.json"
            if not f_p.exists():
                if tr is None:
                    keep = sorted(set(PITCHING_FEATURES)) + ["pitcher", "group", "rv_x"]
                    tr = pd.concat([_pitches(loader, s, f"{t - 1}-03-01", heights)[keep] for s in seasons], ignore_index=True)
                bags = _fit(tr, FEATS[model], lr, n_bags, seed)
                CACHE.mkdir(exist_ok=True)
                pd.DataFrame({"p": predict(bags, d, FEATS[model])}).to_parquet(f_p, index=False)
                f_t.write_text(json.dumps({g: [m.best_iteration for m in bag] for g, bag in bags.items()}), encoding="utf-8")
                print(f"  outcome {t}, {setup}: fit {model} seed {seed}", flush=True)
            d[f"p_{model}_{seed}"] = pd.read_parquet(f_p)["p"].to_numpy()
            its = json.loads(f_t.read_text(encoding="utf-8"))
            trees[f"{model}_{seed}"] = {g: [min(v), max(v)] for g, v in its.items()}
    return d, trees


def _diff(a: pd.Series, b: pd.Series) -> dict:
    x = (a - b).abs()
    return {"mean_abs": round(float(x.mean()), 3), "p95": round(float(np.percentile(x, 95)), 3), "max": round(float(x.max()), 3),
            "share_over_1": round(float((x > 1).mean()), 4), "share_over_2": round(float((x > 2).mean()), 4)}


def _fold(loader: Loader, t: int, seasons: list[int], heights: pd.Series, setup: str) -> dict:
    d, trees = _predictions(loader, t, seasons, heights, setup)
    g = d.groupby("pitcher")
    ps = pd.DataFrame({"pitches": g.size()})
    for s in SEEDS:
        for model in FEATS:
            ps[f"{model}_{s}"] = 100 * g[f"p_{model}_{s}"].mean()
        ps[f"loc_{s}"] = ps[f"full_{s}"] - ps[f"no_loc_{s}"]
    # Location+ points: 10 per SD of the location value among the season's qualified pitchers (the frozen scale's rule).
    sd = float(ps.loc[ps["pitches"] >= MIN_PITCHES, f"loc_{SEEDS[0]}"].std())
    st = ps[ps["pitches"] >= MIN_STABLE]
    pts = {k: 10 * (st[f"{k}_{SEEDS[0]}"] - st[f"{k}_{SEEDS[1]}"]) / sd for k in ("loc", "full", "no_loc")}
    outcome = pitcher_season(_pitches(loader, t, f"{t + 1}-03-01", heights)).query("pitches >= @MIN_PITCHES")
    q = ps[ps["pitches"] >= MIN_PITCHES].reset_index().merge(outcome[["pitcher", "rv100"]].rename(columns={"rv100": "y"}), on="pitcher")
    r = [float(q[f"loc_{s}"].corr(q["y"])) for s in SEEDS]
    return {"train_seasons": seasons, "grade_season": t - 1, "n_stability": int(len(st)), "n_outcome": int(len(q)),
            "points": _diff(st[f"loc_{SEEDS[0]}"] * 10 / sd, st[f"loc_{SEEDS[1]}"] * 10 / sd),
            "parts": {"full": _diff(st[f"full_{SEEDS[0]}"] * 10 / sd, st[f"full_{SEEDS[1]}"] * 10 / sd),
                      "no_loc": _diff(st[f"no_loc_{SEEDS[0]}"] * 10 / sd, st[f"no_loc_{SEEDS[1]}"] * 10 / sd),
                      "r_between_models_seed_differences": round(float(pts["full"].corr(pts["no_loc"])), 3)},
            "r_next_rv": {"by_seed": [round(x, 4) for x in r], "mean": round(float(np.mean(r)), 4)},
            "r_between_seeds": round(float(st[f"loc_{SEEDS[0]}"].corr(st[f"loc_{SEEDS[1]}"])), 4),
            "trees": trees}


def _published_location(loader: Loader, t: int, seasons: list[int], heights: pd.Series, d: pd.DataFrame) -> np.ndarray:
    """The gate's baseline on the graded season's pitches: the location-only model, trained as train.py trains it."""
    f_p = CACHE / f"locst_{t}_{'-'.join(map(str, seasons))}_locbase_{_code_key()}.parquet"
    if not f_p.exists():
        keep = sorted(set(PITCHING_FEATURES)) + ["pitcher", "group", "rv_x"]
        tr = pd.concat([_pitches(loader, s, f"{t - 1}-03-01", heights)[keep] for s in seasons], ignore_index=True)
        CACHE.mkdir(exist_ok=True)
        pd.DataFrame({"p": predict(fit_kinds(tr, kinds=["loc"])["loc"], d, LOC_FEATURES)}).to_parquet(f_p, index=False)
        print(f"  outcome {t}: fit the location-only baseline", flush=True)
    return pd.read_parquet(f_p)["p"].to_numpy()


def _gate_fold(loader: Loader, t: int, seasons: list[int], heights: pd.Series, setup: str, rng) -> dict:
    cand, _ = _predictions(loader, t, seasons, heights, setup)
    cur, _ = _predictions(loader, t, seasons, heights, "published")
    assert cand[["pitcher", "game_pk"]].equals(cur[["pitcher", "game_pk"]])
    s = SEED                                # the seed train.py and the backtest use
    v = pd.DataFrame({"pitcher": cand["pitcher"],
                      "candidate": cand[f"p_full_{s}"] - cand[f"p_no_loc_{s}"], "current": cur[f"p_full_{s}"] - cur[f"p_no_loc_{s}"],
                      "published_location": _published_location(loader, t, seasons, heights, cand),
                      "pitching_candidate": cand[f"p_full_{s}"], "pitching_current": cur[f"p_full_{s}"]})
    g = v.groupby("pitcher")
    ps = (100 * g.mean()).assign(pitches=g.size()).query("pitches >= @MIN_PITCHES").reset_index()
    outcome = pitcher_season(_pitches(loader, t, f"{t + 1}-03-01", heights)).query("pitches >= @MIN_PITCHES")
    q = ps.merge(outcome[["pitcher", "rv100"]].rename(columns={"rv100": "y"}), on="pitcher").reset_index(drop=True)

    def edge(a: str, b: str) -> dict:
        e = _edge(q, a, b, rng)
        return {k: ([round(x, 4) for x in x_] if isinstance(x_ := e[k], list) else x_ if isinstance(x_, bool) else round(x_, 4)) for k in e}
    return {"n": int(len(q)), "seed": s,
            "vs_published_location": edge("candidate", "published_location"),      # the registered gate
            "vs_current_setup": edge("candidate", "current"),
            "pitching_vs_current_setup": edge("pitching_candidate", "pitching_current")}


def main() -> None:
    names = sys.argv[1:] or list(SETUPS)
    loader = Loader()
    h = loader.load("batter_heights", f"{CFG.season}-03-01")
    heights = (h.set_index("batter")["height_in"] / 12).astype(float)
    F = {t: w[WINDOW] for t, w in folds().items() if WINDOW in w}
    assert all(t < CFG.season for t in F), "a development fold may not use the current season's outcomes"
    out_file = OUT / "development_location_stability.json"
    res = json.loads(out_file.read_text(encoding="utf-8")) if out_file.exists() else {}
    res |= {"seeds": SEEDS, "min_pitches_stability": MIN_STABLE, "min_pitches_outcome": MIN_PITCHES, "window": WINDOW}
    res.setdefault("setups", {})
    if names[0] == "gate":
        rng = np.random.default_rng(SEED)
        for name in names[1:]:
            per = {str(t): _gate_fold(loader, t, seasons, heights, name, rng) for t, seasons in F.items()}
            res.setdefault("gate", {})[name] = {"bootstrap_samples": BOOT, "folds": per}
            for t, f in per.items():
                a, b, c = f["vs_published_location"], f["vs_current_setup"], f["pitching_vs_current_setup"]
                print(f"gate {name}, outcome {t} (n {f['n']}): r {a['r']:+.3f} against the location-only features' {a['r_baseline']:+.3f}, "
                      f"edge {a['edge']:+.3f} ({a['ci90'][0]:+.3f} to {a['ci90'][1]:+.3f}), strict gate {'passes' if a['strict_gate'] else 'fails'}; "
                      f"against the published setup {b['edge']:+.3f} ({b['ci90'][0]:+.3f} to {b['ci90'][1]:+.3f}); "
                      f"Pitching+ r {c['r']:+.3f} against {c['r_baseline']:+.3f}, {c['edge']:+.3f} ({c['ci90'][0]:+.3f} to {c['ci90'][1]:+.3f})", flush=True)
        out_file.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
        return
    for name in names:
        lr_full, lr_no, n_bags = SETUPS[name]
        per = {str(t): _fold(loader, t, seasons, heights, name) for t, seasons in F.items()}
        res["setups"][name] = {
            "learning_rate": {"full": lr_full, "no_loc": lr_no}, "bag_members": n_bags, "folds": per,
            "mean_abs_points": round(float(np.mean([f["points"]["mean_abs"] for f in per.values()])), 3),
            "share_over_1": round(float(np.mean([f["points"]["share_over_1"] for f in per.values()])), 4),
            "r_next_rv": round(float(np.mean([f["r_next_rv"]["mean"] for f in per.values()])), 4)}
        s = res["setups"][name]
        print(f"{name}: {s['mean_abs_points']} points between seeds ({s['share_over_1']:.0%} over 1); r next-season {s['r_next_rv']:+.3f}; "
              + "; ".join(f"{t}: full {f['parts']['full']['mean_abs']}, no_loc {f['parts']['no_loc']['mean_abs']}, "
                          f"r of the two {f['parts']['r_between_models_seed_differences']}" for t, f in per.items()), flush=True)
        res["setups"] = {k: res["setups"][k] for k in SETUPS if k in res["setups"]}
        out_file.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
