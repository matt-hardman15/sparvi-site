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
    fs = season_folds(s["first_test"], s["last_test"], s.get("train_window"), s.get("decision", "03-01"), s.get("predictor_lag", 1))
    return fs[-s.get("fast_folds", 1):] if fast else fs


def run_fold(mod, loader: Loader, fold) -> tuple[pd.DataFrame, dict]:
    start = len(loader.served)
    fitted = mod.fit(loader, fold)
    fit_max = loader.max_available_at(start)
    mid = len(loader.served)
    pred = mod.predict(fitted, loader, fold)
    pred_max = loader.max_available_at(mid)
    audit = {"fit_max_available_at": fit_max, "train_until": fold.train_until,
             "predict_max_available_at": pred_max, "decision_date": fold.decision_date,
             "passes": (fit_max is None or fit_max <= fold.train_until) and (pred_max is None or pred_max <= fold.decision_date)}
    y = mod.outcome(loader, fold)
    df = pred.merge(y, on="entity_id", how="inner").assign(fold=fold.test_season)
    return df, audit


def evaluate(df: pd.DataFrame, spec: dict) -> dict:
    metric = spec["primary_metric"]
    block = spec.get("block", "entity_id")
    out = {"metric": metric, "n": int(len(df)), "point": metrics.score(metric, df["point"], df["y"]), "baselines": {}}
    for b in spec["baselines"]:
        out["baselines"][b] = {"value": metrics.score(metric, df[b], df["y"]),
                               "paired": metrics.paired_bootstrap(df.reset_index(drop=True), metric, "point", b, "y", block,
                                                                  n=spec.get("bootstrap", 2000), seed=spec.get("seed", 7))}
    return out


def gate(ev: dict, g: dict) -> dict:
    """The pre-registered pass gate. Comparisons are oriented so 'better' is positive."""
    sign = 1 if metrics.higher_is_better(ev["metric"]) else -1
    base = ev["baselines"][g["beats_baseline"]]
    gap = sign * base["paired"]["est"]
    lo = min(sign * base["paired"]["ci90"][0], sign * base["paired"]["ci90"][1])
    checks = {"beats_baseline": gap > 0,
              "min_skill": gap >= g.get("min_skill", 0.0),
              "ci_excludes_zero": (lo > 0) if g.get("ci_excludes_zero") else True}
    if "min_point" in g:
        checks["min_point"] = ev["point"] >= g["min_point"] if metrics.higher_is_better(ev["metric"]) else ev["point"] <= g["min_point"]
    return {"checks": checks, "passed": all(checks.values()), "gap": gap}


def run(name: str, fast: bool = False, loader: Loader | None = None, write: bool = True, leakage: bool = True) -> dict:
    m = registry.get(name)
    spec = m.card["backtest"]
    mod = load_module(spec["module"])
    loader = loader or Loader()
    frames, audits = [], []
    for fold in folds_for(spec, fast):
        df, audit = run_fold(mod, loader, fold)
        frames.append(df)
        audits.append({"fold": fold.test_season, **audit})
    df = pd.concat(frames, ignore_index=True)
    ev = evaluate(df, spec)
    g = gate(ev, spec["pass_gate"])
    leak = {"timestamp": {"passed": all(a["passes"] for a in audits), "folds": audits}}
    if leakage:
        from backtest import leakage as lk
        leak.update(lk.run_all(name, mod, spec, folds_for(spec, True)[-1], df))
    passed = g["passed"] and all(v["passed"] for v in leak.values())
    rep = {
        "run_id": uuid.uuid4().hex[:12], "model": name, "version": m.version, "kind": "fast" if fast else "full",
        "ran_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "code_sha": report.code_sha(), "lockfile": report.lock_hash(),
        "preregistered_at": spec["preregistered_at"], "spec": spec, "folds": [a["fold"] for a in audits],
        "evaluation": ev, "gate": g, "leakage": leak, "passed": passed,
        "primary": {"metric": ev["metric"], "point": ev["point"], **{b: v["value"] for b, v in ev["baselines"].items()}},
        "revision_safe": loader.revision_safe(), "approximate_tables": loader.approximate_tables(),
    }
    if write:
        report.write(m.dir, m.version, rep)
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
    show["gate"], show["leakage"] = rep["gate"], {k: v["passed"] for k, v in rep["leakage"].items()}
    print(json.dumps(show, indent=2, default=str))
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
