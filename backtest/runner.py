"""Run a model's pre-registered backtest (its card's `backtest:` block) through the harness.

    python -m backtest.runner stuff_plus            # full backtest of the current version
    python -m backtest.runner stuff_plus --fast     # the most recent fold(s) only (pull requests)

The model module implements fit(loader, fold), predict(fitted, loader, fold) -> entity_id, point,
<baselines...>, and outcome(loader, fold) -> entity_id, y. The harness owns the dates: a model only
ever sees data through the point-in-time loader, as of the dates in its fold, and the timestamp audit
fails the run if anything later slipped into a prediction.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import uuid
from datetime import UTC, datetime

import pandas as pd

from backtest import metrics, report
from backtest.asof import Loader
from backtest.splits import season_folds
from sparvi import registry
from sparvi.config import ROOT


def load_module(path: str):
    spec = importlib.util.spec_from_file_location(f"bt_{abs(hash(path))}", ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def folds_for(spec: dict, fast: bool):
    s = spec["splits"]
    fs = season_folds(s["first_test"], s["last_test"], s.get("train_window"), s.get("decision", "03-01"), s.get("predictor_lag", 1),
                      s.get("train_until"), s.get("resolve"), s.get("first_train"))
    return fs[-s.get("fast_folds", 1):] if fast else fs


def fold_predictions(mod, loader: Loader, fold) -> tuple[pd.DataFrame, dict]:
    """Fit and predict one fold: the model's predictions (every entity, before outcomes are joined) and the timestamp audit."""
    start = len(loader.served)
    fitted = mod.fit(loader, fold)
    fit_max = loader.max_available_at(start)
    mid = len(loader.served)
    pred = mod.predict(fitted, loader, fold)
    pred_max = loader.max_available_at(mid)
    audit = {"fit_max_available_at": fit_max, "train_until": fold.train_until,
             "predict_max_available_at": pred_max, "decision_date": fold.decision_date,
             "passes": (fit_max is None or fit_max <= fold.train_until) and (pred_max is None or pred_max <= fold.decision_date)}
    return pred, audit


def score_fold(mod, loader: Loader, fold, pred: pd.DataFrame) -> pd.DataFrame:
    return pred.merge(mod.outcome(loader, fold), on="entity_id", how="inner").assign(fold=fold.test_season)


def run_fold(mod, loader: Loader, fold) -> tuple[pd.DataFrame, dict]:
    pred, audit = fold_predictions(mod, loader, fold)
    return score_fold(mod, loader, fold, pred), audit


def evaluate(df: pd.DataFrame, spec: dict, segments: bool = True) -> dict:
    """The primary metric and each baseline's paired comparison; with the spec's `segment` (a prediction column, e.g.
    position group), the same per segment under "segments"."""
    metric = spec["primary_metric"]
    block = spec.get("block", "entity_id")
    out = {"metric": metric, "n": int(len(df)), "point": metrics.score(metric, df["point"], df["y"]), "baselines": {},
           "by_fold": {int(f): metrics.score(metric, g["point"], g["y"]) for f, g in df.groupby("fold")}}
    for b in spec["baselines"]:
        out["baselines"][b] = {"value": metrics.score(metric, df[b], df["y"]),
                               "paired": metrics.paired_bootstrap(df.reset_index(drop=True), metric, "point", b, "y", block,
                                                                  n=spec.get("bootstrap", 2000), seed=spec.get("seed", 7))}
    # Pre-registered either way: reported beside the gate (Location+: `report_partial_given`), or the gate itself
    # (Pitching+: `pass_gate.partial_given`).
    given = {x for x in (spec.get("report_partial_given"), (spec.get("pass_gate") or {}).get("partial_given")) if x}
    if len(given) > 1:
        raise ValueError(f"report_partial_given and pass_gate.partial_given name different columns: {sorted(given)}")
    if given:
        out["partial_r"] = metrics.partial_bootstrap(df.reset_index(drop=True), "point", given.pop(), "y", block,
                                                     n=spec.get("bootstrap", 2000), seed=spec.get("seed", 7))
    noise = (spec.get("pass_gate") or {}).get("calibration_noise")
    if noise:   # F4's calibration gate: the ECE must sit below what perfect calibration shows on these rows by chance
        out["ece_noise"] = metrics.ece_noise(df["point"], q=noise.get("quantile", 0.9), draws=noise.get("draws", 2000),
                                             seed=spec.get("seed", 7))
    if segments and spec.get("segment"):
        out["segments"] = {str(k): evaluate(g.reset_index(drop=True), spec, segments=False) for k, g in df.groupby(spec["segment"])}
    return out


def gate(ev: dict, g: dict) -> dict:
    """The pre-registered pass gate. Comparisons are oriented so 'better' is positive. beats_baseline is one baseline
    or a list; with a list, every check must hold against each of them (gap and lower bound are the worst). A gate with
    no beats_baseline (a calibration gate, e.g. F4's lookup-table components) checks min_point alone.
    A gate with partial_given (Pitching+; Matt, Nov 25 2026) asks whether the prediction knows something a named
    column doesn't: its partial correlation with the outcome given that column must be above zero, and with
    ci_excludes_zero so must the low end of its 90% interval. It can't be combined with beats_baseline."""
    sign = 1 if metrics.higher_is_better(ev["metric"]) else -1
    if g.get("partial_given"):
        if g.get("beats_baseline"):
            raise ValueError("a gate has partial_given or beats_baseline, not both")
        p = ev["partial_r"]
        if p["given"] != g["partial_given"]:
            raise ValueError(f"the partial r was computed given {p['given']}, the gate names {g['partial_given']}")
        lo = min(p["ci90"])
        checks = {"partial_above_zero": p["est"] > 0, "ci_excludes_zero": (lo > 0) if g.get("ci_excludes_zero") else True}
        return {"checks": checks, "passed": all(checks.values()), "gap": p["est"], "per_baseline": {},
                "partial": {"given": p["given"], "est": p["est"], "ci90_low": lo}}
    if not g.get("beats_baseline"):
        if g.get("calibration_noise"):
            bar = ev["ece_noise"]["bar"]
            ok = ev["point"] < bar
            return {"checks": {"calibration_noise": ok}, "passed": ok, "gap": bar - ev["point"], "per_baseline": {},
                    "noise_bar": bar}
        if "min_point" not in g:
            raise ValueError("a gate without beats_baseline needs min_point or calibration_noise")
        ok = ev["point"] >= g["min_point"] if sign > 0 else ev["point"] <= g["min_point"]
        return {"checks": {"min_point": ok}, "passed": ok, "gap": sign * (ev["point"] - g["min_point"]), "per_baseline": {}}
    names = g["beats_baseline"] if isinstance(g["beats_baseline"], list) else [g["beats_baseline"]]
    gaps, los = [], []
    for b in names:
        base = ev["baselines"][b]
        gaps.append(sign * base["paired"]["est"])
        los.append(min(sign * base["paired"]["ci90"][0], sign * base["paired"]["ci90"][1]))
    gap, lo = min(gaps), min(los)
    checks = {"beats_baseline": gap > 0,
              "min_skill": gap >= g.get("min_skill", 0.0),
              "ci_excludes_zero": (lo > 0) if g.get("ci_excludes_zero") else True}
    if "min_point" in g:
        checks["min_point"] = ev["point"] >= g["min_point"] if metrics.higher_is_better(ev["metric"]) else ev["point"] <= g["min_point"]
    return {"checks": checks, "passed": all(checks.values()), "gap": gap,
            "per_baseline": {b: {"gap": x, "ci90_low": y} for b, x, y in zip(names, gaps, los, strict=True)}}


def gate_all(ev: dict, g: dict) -> dict:
    """The gate on the pooled rows, or with pass_gate.each_segment (Matt, CFB-PROJ: strict in every position group) on
    every segment, each against every baseline; the overall result is the worst segment's."""
    if not g.get("each_segment"):
        return gate(ev, g)
    per = {k: gate(v, g) for k, v in ev["segments"].items()}
    worst = min(per, key=lambda k: per[k]["gap"])
    return {"checks": {k: v["passed"] for k, v in per.items()}, "passed": all(v["passed"] for v in per.values()),
            "gap": per[worst]["gap"], "per_segment": per}


def tested_version(m) -> str:
    """The version a card's backtest block tests: the champion's, unless the block names a `candidate` (a proposed
    version tested before it replaces anything; its report never validates the published version)."""
    return str(m.card.get("backtest", {}).get("candidate", m.version))


def run(name: str, fast: bool = False, loader: Loader | None = None, write: bool = True, leakage: bool = True) -> dict:
    m = registry.get(name)
    spec = m.card["backtest"]
    mod = load_module(spec["module"])
    loader = loader or Loader()
    frames, audits, preds = [], [], {}
    for fold in folds_for(spec, fast):
        preds[fold.test_season], audit = fold_predictions(mod, loader, fold)
        frames.append(score_fold(mod, loader, fold, preds[fold.test_season]))
        audits.append({"fold": fold.test_season, **audit})
    df = pd.concat(frames, ignore_index=True)
    ev = evaluate(df, spec)
    g = gate_all(ev, spec["pass_gate"])
    leak = {"timestamp": {"passed": all(a["passes"] for a in audits), "folds": audits}}
    if leakage:
        from backtest import leakage as lk
        last = folds_for(spec, True)[-1]
        # The leakage tests compare against this fold's clean predictions, which the loop above just made: training is
        # deterministic, so they are passed in and not refit (BT-REUSE). Not from a loader that corrupts what it serves.
        leak.update(lk.run_all(name, mod, spec, last, df, clean=preds.get(last.test_season) if loader.poison is None else None))
    passed = g["passed"] and all(v["passed"] for v in leak.values())
    rep = {
        "run_id": uuid.uuid4().hex[:12], "model": name, "version": tested_version(m), "kind": "fast" if fast else "full",
        "ran_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "code_sha": report.code_sha(), "lockfile": report.lock_hash(),
        "preregistered_at": spec["preregistered_at"], "spec": spec, "folds": [a["fold"] for a in audits],
        "evaluation": ev, "gate": g, "leakage": leak, "passed": passed,
        "primary": {"metric": ev["metric"], "point": ev["point"], **{b: v["value"] for b, v in ev["baselines"].items()}},
        "revision_safe": loader.revision_safe(), "approximate_tables": loader.approximate_tables(),
    }
    if write:
        report.write(m.dir, tested_version(m), rep)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-leakage", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="don't write the report or the run log")
    a = ap.parse_args()
    rep = run(a.model, fast=a.fast, write=not a.dry_run, leakage=not a.no_leakage)
    show = {k: rep[k] for k in ("model", "version", "kind", "folds", "passed", "revision_safe", "approximate_tables")}
    show["evaluation"] = {k: rep["evaluation"][k] for k in ("metric", "n", "point")}
    show["baselines"] = {b: {"value": v["value"], "gap": v["paired"]["est"], "ci90": v["paired"]["ci90"]} for b, v in rep["evaluation"]["baselines"].items()}
    if "partial_r" in rep["evaluation"]:
        show["partial_r"] = rep["evaluation"]["partial_r"]
    show["gate"], show["leakage"] = rep["gate"], {k: v["passed"] for k, v in rep["leakage"].items()}
    print(json.dumps(show, indent=2, default=str))
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
