"""Development folds for Stuff+ as an outcome chain (PQ-CHAIN): does modeling what happens to a pitch step by step,
then valuing the steps, predict better than one model on the pitch's run value?

DEVELOPMENT ONLY, on the folds of develop.py (train on the two seasons ending at t-2, grade season t-1 pitchers,
compare with season t run value per 100 pitches, outcome seasons before the current one). Nothing here reads an
outcome from the current season. Needs the MLB history seasons and develop.py's cached Stuff+ fits for the same
folds (`python baseball/pitch_quality/develop.py windows` and `... targets --window 2`).

Run value per pitch is nearly all noise, which is why Stuff+ needs a small learning rate and a bag of models. Each
step of what happens to a pitch has far more signal. The chain, on Stuff+'s features (shape only; no location, no
count), per pitch group, each step a bag of models over pitcher folds as train.py builds them:
    swing     P(swing)                       every pitch
    whiff     P(miss | swing)                swings
    foul      P(foul | contact)              swings with contact
    called    P(called strike | take)        takes
    contact   expected value of the ball in play, relative to the average ball in play (from expected wOBA)
The grade is the expected context-neutral run value of the pitch: each outcome's average run value in the training
seasons (develop.context_neutral's classes), weighted by the chain's probabilities.

Compared on the same pitchers with the published target (`rv_x`) and the context-neutral target in one model (`cn`,
PQ-TARGET), both from develop.py's cache. 90% intervals: paired bootstrap over pitchers, per fold and pooled.

    python baseball/pitch_quality/develop_chain.py        # about 35 minutes on 4 cores; resumes from its cache

Writes output/development_chain.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from develop import BOOT, CACHE, CFG, OUT, OUTCOME_CLASS, _code_key, _pitches, folds
from features import FEATURES, WOBA_SCALE
from train import GROUPS, MIN_PITCHES, N_BAGS, PARAMS, SEED, bag_folds, pitcher_season  # noqa: F401  (MIN_PITCHES: query strings)

from backtest.asof import Loader

WINDOW = "2"
SWINGS = ("swinging_strike", "foul", "in_play")
# step -> (objective, learning rate, early-stopping rounds). The classifiers have enough signal for a larger step
# than Stuff+'s 0.005; the contact model is as noisy as run value and keeps a small one.
STEPS = {"swing": ("binary", 0.05, 100), "whiff": ("binary", 0.05, 100), "foul": ("binary", 0.05, 100),
         "called": ("binary", 0.05, 100), "contact": ("regression", 0.01, 200)}
REFS = ("rv_x", "cn")


def _key() -> str:
    return hashlib.sha256((_code_key() + Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n")).encode()).hexdigest()[:10]


def labels(d: pd.DataFrame) -> pd.DataFrame:
    """Each step's rows (`in_<step>`) and target (`y_<step>`)."""
    cls = d["description"].map(OUTCOME_CLASS)
    xw = d["estimated_woba_using_speedangle"].fillna(d["woba_value"])
    out = pd.DataFrame({"cls": cls}, index=d.index)
    swing, contact = cls.isin(SWINGS), cls.isin(("foul", "in_play"))
    out["in_swing"], out["y_swing"] = cls.notna(), swing.astype(float)
    out["in_whiff"], out["y_whiff"] = swing, cls.eq("swinging_strike").astype(float)
    out["in_foul"], out["y_foul"] = contact, cls.eq("foul").astype(float)
    out["in_called"], out["y_called"] = cls.notna() & ~swing, cls.eq("called_strike").astype(float)
    out["in_contact"], out["xw"] = cls.eq("in_play") & xw.notna(), xw
    return out


def values(tr: pd.DataFrame, lab: pd.DataFrame) -> dict:
    """Average run value (pitcher's view) of each outcome in the training seasons, and the average ball in play's
    expected wOBA."""
    v = tr["rv"].groupby(lab["cls"]).mean()
    take_no_strike = lab["cls"].isin(("ball", "hit_by_pitch"))
    return {"called_strike": float(v["called_strike"]), "swinging_strike": float(v["swinging_strike"]), "foul": float(v["foul"]),
            "in_play": float(v["in_play"]), "ball": float(tr.loc[take_no_strike, "rv"].mean()),
            "mean_xw": float(lab.loc[lab["in_contact"], "xw"].mean())}


def fit_step(tr: pd.DataFrame, y: pd.Series, rows: pd.Series, fold: pd.Series, step: str) -> tuple[dict, dict]:
    """{group: bag} for one step, and each group's mean held-out score (log loss or squared error)."""
    objective, lr, stop = STEPS[step]
    bags, score = {}, {}
    for g in GROUPS:
        use = rows & tr["group"].eq(g)
        bag = []
        for f in range(N_BAGS):
            a, b = use & fold.ne(f), use & fold.eq(f)
            dtr = lgb.Dataset(tr.loc[a, FEATURES], y[a])
            m = lgb.train({**PARAMS, "objective": objective, "learning_rate": lr, "seed": SEED * 100 + f}, dtr, num_boost_round=10_000,
                          valid_sets=[lgb.Dataset(tr.loc[b, FEATURES], y[b], reference=dtr)],
                          callbacks=[lgb.early_stopping(stop, verbose=False)])
            bag.append(m)
        bags[g] = bag
        score[g] = {"trees": [m.best_iteration for m in bag],
                    "held_out": float(np.mean([next(iter(m.best_score["valid_0"].values())) for m in bag]))}
    return bags, score


def predict_step(bags: dict, d: pd.DataFrame) -> np.ndarray:
    out = np.full(len(d), np.nan)
    for g, bag in bags.items():
        idx = d["group"].eq(g).to_numpy()
        if idx.any():
            out[idx] = np.mean([m.predict(d.loc[idx, FEATURES], num_iteration=m.best_iteration) for m in bag], axis=0)
    return out


def chain_value(p: pd.DataFrame, v: dict) -> pd.Series:
    """Expected context-neutral run value of each pitch from the steps' predictions."""
    in_play = v["in_play"] + p["contact"]
    contact = p["foul"] * v["foul"] + (1 - p["foul"]) * in_play
    swing = p["whiff"] * v["swinging_strike"] + (1 - p["whiff"]) * contact
    take = p["called"] * v["called_strike"] + (1 - p["called"]) * v["ball"]
    return p["swing"] * swing + (1 - p["swing"]) * take


def _fold(loader: Loader, t: int, seasons: list[int], heights: pd.Series) -> tuple[pd.DataFrame, dict]:
    tag = f"chain_{t}_{'-'.join(map(str, seasons))}"
    f_ps, f_info = CACHE / f"{tag}_{_key()}.parquet", CACHE / f"{tag}_{_key()}.json"
    if f_ps.exists() and f_info.exists():
        return pd.read_parquet(f_ps), json.loads(f_info.read_text(encoding="utf-8"))
    keep = [*FEATURES, "pitcher", "group", "rv", "description", "estimated_woba_using_speedangle", "woba_value"]
    tr = pd.concat([_pitches(loader, s, f"{t - 1}-03-01", heights)[keep] for s in seasons], ignore_index=True)
    lab = labels(tr)
    v = values(tr, lab)
    lab["y_contact"] = -(lab["xw"] - v["mean_xw"]) / WOBA_SCALE          # pitcher's view, relative to the average ball in play
    fold = bag_folds(tr["pitcher"], SEED, N_BAGS)
    d = _pitches(loader, t - 1, f"{t}-03-01", heights)
    info = {"values": v, "train_pitches": int(len(tr)), "steps": {}}
    CACHE.mkdir(exist_ok=True)
    for step in STEPS:
        # Each step's predictions are kept as it finishes (private, gitignored), so a stopped run resumes there.
        f_p, f_s = CACHE / f"{tag}_{step}_{_key()}.parquet", CACHE / f"{tag}_{step}_{_key()}.json"
        if not (f_p.exists() and f_s.exists()):
            bags, score = fit_step(tr, lab[f"y_{step}"], lab[f"in_{step}"], fold, step)
            pd.DataFrame({"p": predict_step(bags, d)}).to_parquet(f_p, index=False)
            f_s.write_text(json.dumps({"rows": int(lab[f"in_{step}"].sum()), "groups": score}), encoding="utf-8")
            print(f"  outcome {t}: fit {step}", flush=True)
            del bags
        d[step] = pd.read_parquet(f_p)["p"].to_numpy()
        info["steps"][step] = json.loads(f_s.read_text(encoding="utf-8"))
    del tr, lab
    d["v_chain"] = chain_value(d, v)
    d["v_whiff_only"] = d["swing"] * d["whiff"]                               # a single step, for scale
    ps = pitcher_season(d).query("pitches >= @MIN_PITCHES")
    grades = (100 * d.groupby("pitcher")[["v_chain", "v_whiff_only", *STEPS]].mean()).rename(columns={"v_chain": "chain", "v_whiff_only": "whiff_per_pitch"}).reset_index()
    outcome = pitcher_season(_pitches(loader, t, f"{t + 1}-03-01", heights)).query("pitches >= @MIN_PITCHES")
    ps = ps[["pitcher", "pitches", "rv100"]].merge(grades, on="pitcher").merge(
        outcome[["pitcher", "rv100"]].rename(columns={"rv100": "y"}), on="pitcher")
    for ref in REFS:
        f = CACHE / f"{t}_{'-'.join(map(str, seasons))}_{ref}_stuff_{_code_key()}.parquet"
        ps = ps.merge(pd.read_parquet(f)[["pitcher", "xrv100"]].rename(columns={"xrv100": ref}), on="pitcher")
    ps.to_parquet(f_ps, index=False)
    f_info.write_text(json.dumps(info), encoding="utf-8")
    return ps, info


def _r(d: pd.DataFrame, col: str, by_fold: bool) -> float:
    return float(np.mean([g[col].corr(g["y"]) for _, g in d.groupby("fold")])) if by_fold else float(d[col].corr(d["y"]))


def _summ(d: pd.DataFrame, rng, by_fold: bool) -> dict:
    groups = list(d.groupby("pitcher").indices.values())          # a pitcher's rows (one per fold) travel together
    draws = [d.iloc[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])] for _ in range(BOOT)]
    cols = ["chain", *REFS, "rv100", "whiff_per_pitch"]
    out = {"r": {c: {"est": _r(d, c, by_fold), "ci90": [float(np.percentile(b, 5)), float(np.percentile(b, 95))]}
                 for c in cols for b in [[_r(s, c, by_fold) for s in draws]]}, "chain_minus": {}}
    for ref in (*REFS, "rv100"):
        b = [_r(s, "chain", by_fold) - _r(s, ref, by_fold) for s in draws]
        out["chain_minus"][ref] = {"est": _r(d, "chain", by_fold) - _r(d, ref, by_fold),
                                   "ci90": [float(np.percentile(b, 5)), float(np.percentile(b, 95))]}
    return out


def main() -> None:
    loader = Loader()
    h = loader.load("batter_heights", f"{CFG.season}-03-01")
    heights = (h.set_index("batter")["height_in"] / 12).astype(float)
    rng = np.random.default_rng(SEED)
    F = {t: w[WINDOW] for t, w in folds().items() if WINDOW in w}
    assert all(t < CFG.season for t in F), "a development fold may not use the current season's outcomes"
    missing = [f"{t}_{ref}" for t, s in F.items() for ref in REFS
               if not (CACHE / f"{t}_{'-'.join(map(str, s))}_{ref}_stuff_{_code_key()}.parquet").exists()]
    if missing:
        raise SystemExit(f"run develop.py windows and develop.py targets --window 2 first: no cached fit for {missing}")
    res = {"window": WINDOW, "min_pitches": MIN_PITCHES, "bootstrap_samples": BOOT,
           "steps": {k: {"objective": o, "learning_rate": lr, "early_stopping": es} for k, (o, lr, es) in STEPS.items()}, "folds": {}}
    frames = []
    for t, seasons in F.items():
        ps, info = _fold(loader, t, seasons, heights)
        frames.append(ps.assign(fold=t))
        res["folds"][str(t)] = {"train_seasons": seasons, "grade_season": t - 1, "n": int(len(ps)), **info, **_summ(ps, rng, False),
                                "r_step_with_next_rv": {s: float(ps[s].corr(ps["y"])) for s in STEPS}}
        f = res["folds"][str(t)]
        print(f"outcome {t} (n {len(ps)}): " + ", ".join(f"{c} r {v['est']:+.3f}" for c, v in f["r"].items()) + "; chain minus "
              + ", ".join(f"{k} {v['est']:+.3f} [{v['ci90'][0]:+.3f}, {v['ci90'][1]:+.3f}]" for k, v in f["chain_minus"].items()), flush=True)
    allf = pd.concat(frames, ignore_index=True)
    res["pooled"] = {"n": int(len(allf)), **_summ(allf, rng, True)}
    (OUT / "development_chain.json").write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(res["pooled"], indent=2))


if __name__ == "__main__":
    main()
