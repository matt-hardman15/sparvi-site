"""Backtest harness (A5) on synthetic Statcast: the point-in-time loader, the gate, and the leakage
tests, including that the future-poisoning test catches a deliberately leaky model."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from synthetic_statcast import write_fixture

import ingest.snapshot as snap
from backtest import leakage, metrics, runner
from backtest.asof import SNAPSHOTS_START, Loader
from backtest.splits import season_folds
from sparvi.config import baseball

SEASON = baseball().season
FOLD = season_folds(SEASON, SEASON, train_window=1)[0]


@pytest.fixture(scope="module", autouse=True)
def raw(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    write_fixture(d, SEASON)
    mp = pytest.MonkeyPatch()
    mp.setattr(snap, "RAW", d)
    yield d
    mp.undo()


def _season_rv(loader, season, as_of):
    p = loader.load(f"statcast_mlb_{season}", as_of)
    return p.groupby("pitcher")["release_speed"].mean()      # a column the fixture fills


def honest_model():
    def fit(loader, fold, permute_seed=None):
        return {}

    def predict(fitted, loader, fold):
        prev = _season_rv(loader, fold.test_season - 1, fold.decision_date)
        return pd.DataFrame({"entity_id": prev.index, "point": prev.to_numpy(), "last_season": prev.to_numpy()})

    def outcome(loader, fold):
        cur = _season_rv(loader, fold.test_season, fold.resolve_after)
        return pd.DataFrame({"entity_id": cur.index, "y": cur.to_numpy()})
    return SimpleNamespace(fit=fit, predict=predict, outcome=outcome)


def leaky_model():
    m = honest_model()

    def predict(fitted, loader, fold):          # peeks at the season it's supposed to predict
        cur = _season_rv(loader, fold.test_season, fold.resolve_after)
        return pd.DataFrame({"entity_id": cur.index, "point": cur.to_numpy(), "last_season": cur.to_numpy()})
    return SimpleNamespace(fit=m.fit, predict=predict, outcome=m.outcome)


def test_loader_serves_only_rows_available_by_as_of():
    lo = Loader()
    full = lo.load(f"statcast_mlb_{SEASON}", f"{SEASON + 1}-01-01")
    mid = lo.load(f"statcast_mlb_{SEASON}", f"{SEASON}-06-01")
    assert 0 < len(mid) < len(full)
    assert pd.to_datetime(mid["game_date"]).max() < pd.Timestamp(f"{SEASON}-06-01")
    assert lo.max_available_at() <= pd.Timestamp(f"{SEASON + 1}-01-01")
    assert not lo.revision_safe()                     # local files before snapshots began
    lo.load("batter_heights", f"{SEASON}-03-01")
    assert lo.approximate_tables() == ["batter_heights"] and SNAPSHOTS_START > pd.Timestamp(f"{SEASON}-03-01")


def test_honest_model_passes_every_leakage_test():
    mod, lo = honest_model(), Loader()
    df, audit = runner.run_fold(mod, lo, FOLD)
    assert audit["passes"]
    out = leakage.run_all("toy", mod, {"group_by": "none"}, FOLD, df)
    assert out["future_poisoning"]["passed"] and out["label_permutation"]["passed"] is not None
    assert df["point"].notna().all() and len(df) > 20


def test_future_poisoning_and_timestamp_catch_a_leaky_model():
    mod, lo = leaky_model(), Loader()
    df, audit = runner.run_fold(mod, lo, FOLD)
    assert not audit["passes"], "the timestamp audit must see data after the decision date"
    clean = mod.predict(mod.fit(Loader(), FOLD), Loader(), FOLD)
    assert not leakage.future_poisoning(mod, FOLD, clean)["passed"]


def test_label_permutation_needs_the_hook():
    m = honest_model()
    no_hook = SimpleNamespace(fit=lambda loader, fold: {}, predict=m.predict, outcome=m.outcome)
    assert not leakage.label_permutation(no_hook, FOLD)["passed"]


def test_gate_is_strict_about_the_interval():
    ev = {"metric": "pearson_r", "point": 0.30, "baselines": {"last_season": {"value": 0.19, "paired": {"est": 0.11, "ci90": [-0.002, 0.216]}}}}
    assert not runner.gate(ev, {"beats_baseline": "last_season", "ci_excludes_zero": True})["passed"]
    assert runner.gate(ev, {"beats_baseline": "last_season", "ci_excludes_zero": False})["passed"]
    ev_mae = {"metric": "mae", "point": 12.5, "baselines": {"elo": {"value": 12.7, "paired": {"est": -0.13, "ci90": [-0.20, -0.07]}}}}
    assert runner.gate(ev_mae, {"beats_baseline": "elo", "ci_excludes_zero": True})["passed"]


def test_paired_bootstrap_is_seeded_and_blocked():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"entity_id": np.repeat(range(50), 3), "y": rng.normal(size=150)})
    df["a"], df["b"] = df["y"] + rng.normal(0, 0.5, 150), rng.normal(size=150)
    r1 = metrics.paired_bootstrap(df, "pearson_r", "a", "b", "y", "entity_id", n=300)
    r2 = metrics.paired_bootstrap(df, "pearson_r", "a", "b", "y", "entity_id", n=300)
    assert r1 == r2 and r1["blocks"] == 50 and r1["ci90"][0] > 0


def test_publish_gate(tmp_path):
    import json

    import yaml

    from backtest.gate import BacktestRequired, require_passing
    from sparvi import registry
    d = tmp_path / "models" / "test" / "toy"
    (d / "backtest").mkdir(parents=True)
    (tmp_path / "toy.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "m.json").write_text('{"mae": 1.0}', encoding="utf-8")
    card = {"name": "toy", "title": "Toy", "sport": "test", "status": "champion", "version": "1.0.0", "code": ["toy.py"],
            "validation": {"file": "m.json", "metrics": {"mae": "mae"}},
            "eval": {"set": "s", "set_keys": {}, "primary": "mae", "higher_is_better": False}}
    (d / "card.yml").write_text(yaml.safe_dump(card), encoding="utf-8")
    with pytest.raises(BacktestRequired):
        require_passing("toy", root=tmp_path)                               # no backtest, not flagged
    (d / "backtest" / "1.0.0.json").write_text(json.dumps({"version": "1.0.0", "passed": False}), encoding="utf-8")
    with pytest.raises(BacktestRequired, match="failed its gate"):
        require_passing("toy", root=tmp_path)
    card["experimental"] = "not yet validated"
    (d / "card.yml").write_text(yaml.safe_dump(card), encoding="utf-8")
    assert require_passing("toy", root=tmp_path)["state"] == "experimental"
    (d / "backtest" / "1.0.0.json").write_text(json.dumps({"version": "1.0.0", "passed": True}), encoding="utf-8")
    assert require_passing("toy", root=tmp_path)["state"] == "validated"
    (d / "backtest" / "1.0.0.json").write_text(json.dumps({"version": "0.9.0", "passed": True}), encoding="utf-8")
    del card["experimental"]
    (d / "card.yml").write_text(yaml.safe_dump(card), encoding="utf-8")
    with pytest.raises(BacktestRequired):
        require_passing("toy", root=tmp_path)                               # a report for another version doesn't count
    assert registry.get("toy", tmp_path).version == "1.0.0"


def test_model_code_reads_data_only_through_the_loader():
    """PLAN-backtest §2: no direct raw-table reads in models/ (or a future features/)."""
    import re

    from sparvi.config import ROOT
    bad = []
    for p in list((ROOT / "models").rglob("*.py")) + list((ROOT / "features").rglob("*.py") if (ROOT / "features").exists() else []):
        text = p.read_text(encoding="utf-8")
        for pat in (r"read_parquet\(", r"read_csv\(", r"\bRAW\b", r"data/raw", r"duckdb\.(sql|connect)\("):
            if re.search(pat, text):
                bad.append(f"{p.relative_to(ROOT)}: {pat}")
    assert not bad, "read through backtest.asof.Loader instead:\n" + "\n".join(bad)
