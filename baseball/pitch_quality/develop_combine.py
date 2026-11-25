"""Development folds for a next-season pitcher forecast that combines the grades with last season's record (PQ-COMBINE).

DEVELOPMENT ONLY, on the folds of develop_loc.py (outcome seasons before the current one; nothing here reads the
current season). The prediction audit (plans/AUDIT-prediction.md, section 1) found that a pitcher's strikeout and walk
rates predict his next season beyond what Stuff+ and Pitching+ see. This measures that properly: a fixed list of
specifications per target (SPECS, written before the run), each fitted by least squares on one fold and scored on the
other, with the grades from models trained as train.py trains them.

Fold (outcome season t): models fit on seasons t-3 and t-2 grade season t-1; the record is season t-1's; the outcome
is season t's. Pitchers with MIN_PITCHES+ pitches in both seasons. Targets (season t):
    rv100    run value per 100 pitches (what the grades are tested against)
    k_bb     strikeout rate minus walk rate, per plate appearance (an L1b target)
    xwoba    expected wOBA allowed per plate appearance (an L1b target)
Inputs (season t-1): stuff, loc (Location+ value: full model minus the same model without location), k_pct, bb_pct,
xwobacon (expected wOBA on balls in play), csw (called strikes plus whiffs per pitch), and each target's own value.

Per target and specification: r and RMSE out of fold (the mean of the two folds), the edge in r and in RMSE over the
target's reference with a paired bootstrap over pitchers (a pitcher's rows in both folds travel together; the weights
are refitted in every draw), the coverage of an 80% interval built from the fitting fold's residual SD, and the same r
with the grades of a second training seed. Standardized weights are from a fit on both folds.

    python baseball/pitch_quality/develop_combine.py

Writes output/development_combine.json (aggregates only; the pitcher-level tables stay in the gitignored cache).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from develop import BOOT, CACHE, CFG, KEY, OUT, _code_key, _pitches, folds
from develop_loc import WINDOW
from develop_loc_stable import SEEDS, _fit, _predictions
from features import FEATURES
from train import LEARNING_RATE, MIN_PITCHES, N_BAGS, SEED, predict

from backtest.asof import Loader

WHIFF = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
TARGETS = {"rv100": "higher is better for the pitcher", "k_bb": "per plate appearance", "xwoba": "per plate appearance"}
# target: (reference, {name: inputs}). Written before the run; the reference is what each other spec is compared with.
SPECS = {
    "rv100": ("stuff", {
        "stuff": ["stuff"],
        "last": ["rv100"],
        "k+bb": ["k_pct", "bb_pct"],
        "stuff+loc": ["stuff", "loc"],
        "stuff+k+bb": ["stuff", "k_pct", "bb_pct"],
        "stuff+k+bb+xwobacon": ["stuff", "k_pct", "bb_pct", "xwobacon"],
        "stuff+loc+k+bb+xwobacon": ["stuff", "loc", "k_pct", "bb_pct", "xwobacon"],
    }),
    "k_bb": ("last", {
        "last": ["k_bb"],
        "k+bb": ["k_pct", "bb_pct"],
        "stuff": ["stuff"],
        "k+bb+stuff": ["k_pct", "bb_pct", "stuff"],
        "k+bb+stuff+loc": ["k_pct", "bb_pct", "stuff", "loc"],
        "k+bb+stuff+loc+csw": ["k_pct", "bb_pct", "stuff", "loc", "csw"],
    }),
    "xwoba": ("last", {
        "last": ["xwoba"],
        "k+bb+xwobacon": ["k_pct", "bb_pct", "xwobacon"],
        "stuff": ["stuff"],
        "k+bb+xwobacon+stuff": ["k_pct", "bb_pct", "xwobacon", "stuff"],
        "k+bb+xwobacon+stuff+loc": ["k_pct", "bb_pct", "xwobacon", "stuff", "loc"],
    }),
}
Z80 = 1.2815515655446004


def _record(loader: Loader, season: int, as_of: str, heights: pd.Series) -> pd.DataFrame:
    """One row per pitcher: the season's record, from the pitches the models use."""
    raw = loader.load(f"statcast_mlb_{season}", as_of)
    d = _pitches(loader, season, as_of, heights).merge(
        raw[[*KEY, "events", "woba_denom"]].drop_duplicates(KEY), on=KEY, how="left", validate="one_to_one")
    pa = d["woba_denom"].fillna(0).gt(0)                       # plate appearances that count toward wOBA
    bip = d["description"].eq("hit_into_play")
    xw = d["estimated_woba_using_speedangle"].where(bip).fillna(d["woba_value"])
    g = d["pitcher"]
    n_pa = pa.groupby(g).sum()
    t = pd.DataFrame({
        "pitches": d.groupby(g).size(),
        "rv100": 100 * d["rv"].groupby(g).mean(),
        "k_pct": (d["events"].isin(["strikeout", "strikeout_double_play"]) & pa).groupby(g).sum() / n_pa,
        "bb_pct": (d["events"].eq("walk") & pa).groupby(g).sum() / n_pa,
        "xwoba": xw.where(pa).groupby(g).sum() / n_pa,
        "xwobacon": d["estimated_woba_using_speedangle"].where(bip).groupby(g).mean(),
        "csw": (d["description"].isin(WHIFF) | d["description"].eq("called_strike")).groupby(g).mean(),
    })
    t["k_bb"] = t["k_pct"] - t["bb_pct"]
    return t


def _stuff(loader: Loader, t: int, seasons: list[int], heights: pd.Series, d: pd.DataFrame, seed: int) -> np.ndarray:
    """Stuff+ pitch values for the graded season, from a model trained as train.fit_kinds trains it, at this seed."""
    f_p = CACHE / f"comb_{t}_{'-'.join(map(str, seasons))}_stuff_s{seed}_{_code_key()}.parquet"
    if not f_p.exists():
        keep = sorted(set(FEATURES)) + ["pitcher", "group", "rv_x"]
        tr = pd.concat([_pitches(loader, s, f"{t - 1}-03-01", heights)[keep] for s in seasons], ignore_index=True)
        CACHE.mkdir(exist_ok=True)
        pd.DataFrame({"p": predict(_fit(tr, FEATURES, LEARNING_RATE["stuff"], N_BAGS, seed), d, FEATURES)}).to_parquet(f_p, index=False)
        print(f"  outcome {t}: fit stuff seed {seed}", flush=True)
    return pd.read_parquet(f_p)["p"].to_numpy()


def _table(loader: Loader, t: int, seasons: list[int], heights: pd.Series) -> pd.DataFrame:
    """Pitcher rows of a fold: grades (both seeds), the graded season's record, the outcome season's targets (y_*)."""
    d, _ = _predictions(loader, t, seasons, heights, "published")
    g = d["pitcher"]
    ps = pd.DataFrame(index=np.sort(g.unique())).rename_axis("pitcher")
    for s in SEEDS:
        sfx = "" if s == SEED else f"_s{s}"
        ps[f"stuff{sfx}"] = 100 * pd.Series(_stuff(loader, t, seasons, heights, d, s)).groupby(g.to_numpy()).mean()
        ps[f"loc{sfx}"] = 100 * (d[f"p_full_{s}"] - d[f"p_no_loc_{s}"]).groupby(g).mean()
    prev = _record(loader, t - 1, f"{t}-03-01", heights)
    cur = _record(loader, t, f"{t + 1}-03-01", heights)
    q = (ps.join(prev[prev["pitches"] >= MIN_PITCHES], how="inner")
         .join(cur.loc[cur["pitches"] >= MIN_PITCHES, list(TARGETS)].add_prefix("y_"), how="inner"))
    return q.reset_index().assign(fold=t)


def _oos(data: pd.DataFrame, cols: list[str], target: str) -> pd.DataFrame:
    """Out-of-fold forecast of y_{target}: least squares on the other fold(s). Columns: fold, y, pred, sd (the fitting
    fold's residual SD)."""
    out = []
    for t in sorted(data["fold"].unique()):
        tr, te = data[data["fold"] != t], data[data["fold"] == t]
        X = np.column_stack([np.ones(len(tr)), tr[cols].to_numpy(float)])
        beta = np.linalg.lstsq(X, tr[f"y_{target}"].to_numpy(float), rcond=None)[0]
        sd = float(np.std(tr[f"y_{target}"].to_numpy(float) - X @ beta, ddof=len(cols) + 1))
        pred = np.column_stack([np.ones(len(te)), te[cols].to_numpy(float)]) @ beta
        out.append(pd.DataFrame({"fold": t, "y": te[f"y_{target}"].to_numpy(float), "pred": pred, "sd": sd}))
    return pd.concat(out, ignore_index=True)


def _score(o: pd.DataFrame) -> tuple[float, float]:
    """(r, RMSE), each the mean over folds."""
    by = [(float(np.corrcoef(f["pred"], f["y"])[0, 1]), float(np.sqrt(np.mean((f["pred"] - f["y"]) ** 2)))) for _, f in o.groupby("fold")]
    return float(np.mean([b[0] for b in by])), float(np.mean([b[1] for b in by]))


def _other_seed(data: pd.DataFrame) -> pd.DataFrame:
    s = SEEDS[1]
    return data.drop(columns=["stuff", "loc"]).rename(columns={f"stuff_s{s}": "stuff", f"loc_s{s}": "loc"})


def main() -> None:
    loader = Loader()
    h = loader.load("batter_heights", f"{CFG.season}-03-01")
    heights = (h.set_index("batter")["height_in"] / 12).astype(float)
    F = {t: w[WINDOW] for t, w in folds().items() if WINDOW in w}
    assert all(t < CFG.season for t in F), "a development fold may not use the current season's outcomes"
    data = pd.concat([_table(loader, t, seasons, heights) for t, seasons in F.items()], ignore_index=True)
    inputs = sorted({c for _, specs in SPECS.values() for cols in specs.values() for c in cols})
    data = data.dropna(subset=inputs + [f"y_{t}" for t in TARGETS]).reset_index(drop=True)
    groups = list(data.groupby("pitcher").indices.values())
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(BOOT)]
    res = {"folds": {str(t): {"train_seasons": s, "grade_season": t - 1, "n": int((data["fold"] == t).sum())} for t, s in F.items()},
           "n": int(len(data)), "pitchers": len(groups), "min_pitches": MIN_PITCHES, "seeds": SEEDS, "bootstrap_samples": BOOT,
           "interval": "80%, from the fitting fold's residual SD", "targets": {}}
    for target, (ref, specs) in SPECS.items():
        base = _score(_oos(data, specs[ref], target))
        boot_ref = [_score(_oos(data.iloc[ix], specs[ref], target)) for ix in draws]
        tr = {"reference": ref, "sd_outcome": round(float(np.mean([f[f"y_{target}"].std() for _, f in data.groupby("fold")])), 4), "specs": {}}
        for name, cols in specs.items():
            o = _oos(data, cols, target)
            r, rmse = _score(o)
            by_fold = {str(t): round(float(np.corrcoef(f["pred"], f["y"])[0, 1]), 4) for t, f in o.groupby("fold")}
            z = data[cols].apply(lambda c: (c - c.mean()) / c.std())
            w = np.linalg.lstsq(np.column_stack([np.ones(len(z)), z.to_numpy(float)]), data[f"y_{target}"].to_numpy(float), rcond=None)[0][1:]
            s = {"inputs": cols, "r": round(r, 4), "r_by_fold": by_fold, "rmse": round(rmse, 4),
                 "coverage_80": round(float(((o["y"] - o["pred"]).abs() <= Z80 * o["sd"]).mean()), 3),
                 "r_other_seed": round(_score(_oos(_other_seed(data), cols, target))[0], 4),
                 "weights_per_sd": {c: round(float(x), 4) for c, x in zip(cols, w, strict=True)}}
            if name != ref:
                b = np.array([_score(_oos(data.iloc[ix], cols, target)) for ix in draws]) - np.array(boot_ref)
                s["vs_reference"] = {"r": round(r - base[0], 4), "r_ci90": [round(float(x), 4) for x in np.percentile(b[:, 0], [5, 95])],
                                     "rmse": round(rmse - base[1], 4), "rmse_ci90": [round(float(x), 4) for x in np.percentile(b[:, 1], [5, 95])]}
            tr["specs"][name] = s
            e = s.get("vs_reference")
            print(f"{target:6s} {name:26s} r {r:+.3f} ({' '.join(f'{v:+.3f}' for v in by_fold.values())})  rmse {rmse:.4f}  cover {s['coverage_80']:.2f}  "
                  f"seed {SEEDS[1]}: {s['r_other_seed']:+.3f}"
                  + (f"  vs {ref}: r {e['r']:+.3f} ({e['r_ci90'][0]:+.3f} to {e['r_ci90'][1]:+.3f}), rmse {e['rmse']:+.4f} "
                     f"({e['rmse_ci90'][0]:+.4f} to {e['rmse_ci90'][1]:+.4f})" if e else ""), flush=True)
        res["targets"][target] = tr
    (OUT / "development_combine.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
