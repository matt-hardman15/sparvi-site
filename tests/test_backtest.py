"""Backtest harness (A5) on synthetic Statcast: the point-in-time loader, the gate, and the leakage
tests, including that the future-poisoning test catches a deliberately leaky model."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from synthetic_statcast import write_fixture

import ingest.snapshot as snap
from backtest import ci, leakage, metrics, runner
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


def test_gate_against_several_baselines_needs_every_one():
    ev = {"metric": "recall", "point": 0.30, "baselines": {
        "a": {"value": 0.20, "paired": {"est": 0.10, "ci90": [0.04, 0.16]}},
        "b": {"value": 0.27, "paired": {"est": 0.03, "ci90": [-0.01, 0.07]}}}}
    assert runner.gate(ev, {"beats_baseline": ["a"], "ci_excludes_zero": True})["passed"]
    g = runner.gate(ev, {"beats_baseline": ["a", "b"], "ci_excludes_zero": True})
    assert not g["passed"] and g["gap"] == 0.03 and g["per_baseline"]["b"]["ci90_low"] == -0.01


def test_recall_metric_and_training_cutoff():
    assert metrics.recall([1.0, 0.5, 0.0, 0.9], [1, 1, 1, 0]) == 0.5          # mean credit over the positive rows
    assert metrics.higher_is_better("recall")
    f = season_folds(2023, 2023, train_window=1, decision_month_day="05-01", train_until_month_day="05-01")[0]
    assert f.train_until == pd.Timestamp("2023-05-01") and f.decision_date == pd.Timestamp("2023-05-01")
    assert season_folds(2023, 2023, train_window=1)[0].train_until == pd.Timestamp("2021-12-31 23:59:59")


def test_ci_compares_a_fast_run_with_the_same_fold():
    full = {"kind": "full", "folds": [2021, 2022], "evaluation": {"metric": "recall", "point": 0.15, "by_fold": {"2021": 0.18, "2022": 0.12}}}
    fast = {"kind": "fast", "folds": [2022], "evaluation": {"metric": "recall", "point": 0.12}, "leakage": {}, "gate": {"passed": True}}
    m = SimpleNamespace(card={"backtest": {}, "experimental": None})
    assert ci.comparable_point(fast, full) == 0.12
    assert ci.judge(m, fast, full) == (True, [])                       # not "worse than 0.15": that's two folds pooled
    old = {"kind": "full", "folds": [2021, 2022], "evaluation": {"metric": "recall", "point": 0.15}}
    ok, notes = ci.judge(m, fast, old)                                   # older report without per-fold scores
    assert ok and "skipped" in notes[0]
    assert ci.comparable_point({**fast, "evaluation": {"metric": "recall", "point": 0.10}}, full) == 0.12
    assert not ci.judge(m, {**fast, "evaluation": {"metric": "recall", "point": 0.10}}, full)[0]


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
    model_code = [ROOT / "cfb" / "fit" / "model.py"]          # feature code that lives beside its sport
    for p in list((ROOT / "models").rglob("*.py")) + list((ROOT / "features").rglob("*.py") if (ROOT / "features").exists() else [])             + [q for q in model_code if q.exists()]:
        text = p.read_text(encoding="utf-8")
        for pat in (r"read_parquet\(", r"read_csv\(", r"\bRAW\b", r"data/raw", r"duckdb\.(sql|connect)\("):
            if re.search(pat, text):
                bad.append(f"{p.relative_to(ROOT)}: {pat}")
    assert not bad, "read through backtest.asof.Loader instead:\n" + "\n".join(bad)


def _toy_perm(signal: float):
    """A model whose shuffled-label predictions correlate with the outcome by `signal` (a leak) plus seed noise."""
    ids = [f"e{i}" for i in range(200)]
    y = np.random.default_rng(0).normal(size=200)

    def fit(loader, fold, permute_seed=None):
        return {"seed": permute_seed}

    def predict(fitted, loader, fold):
        noise = np.random.default_rng(fitted["seed"] or 0).normal(size=200)
        return pd.DataFrame({"entity_id": ids, "point": signal * y + noise})

    return SimpleNamespace(fit=fit, predict=predict, outcome=lambda loader, fold: pd.DataFrame({"entity_id": ids, "y": y}))


def test_label_permutation_over_several_seeds():
    """permutation_seeds (Matt, Nov 15): the mean shuffled-label r across seeds must include zero; a leak present in
    every seed still fails."""
    spec = {"primary_metric": "pearson_r", "permutation_seeds": 10}
    clean = leakage.label_permutation(_toy_perm(0.0), FOLD, spec=spec)
    assert clean["passed"] is True and clean["seeds"] == 10 and len(clean["r_by_seed"]) == 10
    assert leakage.label_permutation(_toy_perm(0.5), FOLD, spec=spec)["passed"] is False


def test_gate_in_every_segment():
    """pass_gate.each_segment (CFB-PROJ, Matt): one weak segment fails the model even if the pooled rows pass."""
    rng = np.random.default_rng(3)
    rows = []
    for seg, skill in (("QB", 0.8), ("RB", 0.8), ("TE", 0.0)):
        y = rng.normal(size=300)
        base = y * 0.3 + rng.normal(size=300)
        rows.append(pd.DataFrame({"seg": seg, "y": y, "point": y * skill + rng.normal(size=300) * 0.5, "carry": base,
                                  "player": [f"{seg}{i}" for i in range(300)], "fold": 2021}))
    df = pd.concat(rows, ignore_index=True)
    spec = {"primary_metric": "pearson_r", "baselines": ["carry"], "block": "player", "segment": "seg", "bootstrap": 300}
    ev = runner.evaluate(df, spec)
    assert set(ev["segments"]) == {"QB", "RB", "TE"}
    g = {"beats_baseline": ["carry"], "ci_excludes_zero": True}
    assert runner.gate_all(ev, g)["passed"]                                   # pooled: passes
    each = runner.gate_all(ev, {**g, "each_segment": True})
    assert not each["passed"] and each["checks"] == {"QB": True, "RB": True, "TE": False}


def test_not_applicable_needs_a_reason_and_never_waives_timing():
    """A card may declare label permutation not applicable only with a written reason; timing tests can't be waived."""
    ok = leakage.not_applicable("label_permutation", {"label_permutation": "structural model: skill comes from roster composition"})
    assert ok["passed"] is True and "roster" in ok["not_applicable"]
    assert leakage.not_applicable("label_permutation", {"label_permutation": ""})["passed"] is False
    assert leakage.not_applicable("future_poisoning", {"future_poisoning": "a long enough reason to pass the length check"})["passed"] is False
    assert leakage.not_applicable("label_permutation", {}) is None


def test_ci_runs_only_the_models_a_change_touches():
    """backtest.ci --changed (Matt, Nov 16): a PR runs the backtests of the models whose code or folder it touches;
    harness files run every model; site files and committed reports run none."""
    pick = lambda files: ci.selected(files)  # noqa: E731
    names = lambda files: sorted(m.name for m in pick(files))  # noqa: E731
    assert names(["site/index.qmd", "README.md"]) == []
    assert names(["models/cfb/system_fit/backtest/1.1.8.json", "backtest/runs/runs.jsonl"]) == []
    assert names(["cfb/proj/pit.py"]) == ["cfb_player_projection"]
    assert names(["baseball/pitch_quality/train.py"]) == ["location_plus", "stuff_plus"]
    assert names(["baseball/aaa_translation/develop.py"]) == ["aaa_translation"]      # the candidate's code (M5-EB)
    assert "cfb_player_profile" in names(["cfb/profile/features.yml"]) and "stuff_plus" not in names(["cfb/profile/features.yml"])
    assert pick(["backtest/leakage.py"]) is None and pick(["uv.lock"]) is None


def test_partial_r_is_reported_with_an_interval():
    """LOC-TYPE: a grade that adds to a baseline has a partial r above zero; a noisy copy of the baseline doesn't."""
    rng = np.random.default_rng(0)
    z, extra = rng.normal(size=400), rng.normal(size=400)
    df = pd.DataFrame({"entity_id": range(400), "stuff": z, "y": z + extra + rng.normal(size=400)})
    df["adds"], df["copy"] = extra + rng.normal(0, 0.5, 400), z + rng.normal(0, 0.5, 400)
    a = metrics.partial_bootstrap(df, "adds", "stuff", "y", "entity_id", n=300)
    b = metrics.partial_bootstrap(df, "copy", "stuff", "y", "entity_id", n=300)
    assert a == metrics.partial_bootstrap(df, "adds", "stuff", "y", "entity_id", n=300) and a["ci90"][0] > 0
    assert b["ci90"][0] < 0 < b["ci90"][1] and abs(b["est"]) < 0.1
    spec = {"primary_metric": "pearson_r", "baselines": ["stuff"], "report_partial_given": "stuff", "bootstrap": 100}
    ev = runner.evaluate(df.rename(columns={"adds": "point"}).assign(fold=1), spec)
    assert ev["partial_r"]["given"] == "stuff" and ev["partial_r"]["est"] == a["est"]


def test_a_candidate_report_never_validates_the_published_version(tmp_path):
    """LOC-TYPE: a block with `candidate` writes its report under that version, so the champion stays experimental."""
    import json

    import yaml

    from backtest.gate import status
    from sparvi import registry
    d = tmp_path / "models" / "test" / "toy"
    (d / "backtest").mkdir(parents=True)
    card = {"name": "toy", "title": "Toy", "sport": "test", "status": "champion", "version": "1.0.0", "experimental": "old test",
            "backtest": {"candidate": "2.0.0", "vault": {}}}
    (d / "card.yml").write_text(yaml.safe_dump(card), encoding="utf-8")
    m = registry.get("toy", tmp_path)
    assert runner.tested_version(m) == "2.0.0" and ci.last_report(m) is None
    rep = {"version": "2.0.0", "passed": True, "ran_at": "2026-11-19T00:00:00Z", "folds": [2026], "revision_safe": False,
           "gate": {"passed": True}, "leakage": {"timestamp": {"passed": True}},
           "evaluation": {"metric": "pearson_r", "n": 10, "point": 0.2, "baselines": {
               "published": {"value": 0.0, "paired": {"est": 0.2, "ci90": [0.1, 0.3]}}}}}
    (d / "backtest" / "2.0.0.json").write_text(json.dumps(rep), encoding="utf-8")
    assert ci.last_report(m)["version"] == "2.0.0"
    assert status("toy", tmp_path)["state"] == "experimental"             # the published 1.0.0 has no report of its own
    text = registry.site_numbers(tmp_path)["toy_backtest"]
    assert text.startswith("Backtest of the proposed v2.0.0 (not the published version)")
    # Once the candidate is the published version, the page says which part of the model the test covers (M5-EB: the
    # small-sample line passed; the line for 150+ was never in the harness).
    card["version"], card["backtest"]["covers"] = "2.0.0", "the small part only"
    (d / "card.yml").write_text(yaml.safe_dump(card), encoding="utf-8")
    assert status("toy", tmp_path)["state"] == "validated"
    assert registry.site_numbers(tmp_path)["toy_status"] == "validated in the backtest harness: the small part only"


def _toy_pitches(level, season, batters, rng, start="04-01"):
    """One-pitch plate appearances: each batter bats once a day for 80 days."""
    days = pd.date_range(f"{season}-{start}", periods=80)
    n = len(batters) * len(days)
    k = rng.random(n) < 0.25
    return pd.DataFrame({
        "game_pk": np.arange(n) + season * 1_000_000 + (0 if level == "aaa" else 500_000), "game_date": np.repeat(days, len(batters)),
        "game_year": season, "game_type": "R", "pitcher": 900_000 + rng.integers(0, 40, n) + (0 if level == "aaa" else 100),
        "batter": np.tile(batters, len(days)), "at_bat_number": 1, "pitch_number": 1, "zone": rng.integers(1, 15, n).astype(float),
        "description": np.where(k, "swinging_strike", "hit_into_play"), "events": np.where(k, "strikeout", "field_out"),
        "launch_speed": np.where(k, np.nan, rng.normal(90, 8, n)), "estimated_woba_using_speedangle": np.where(k, np.nan, rng.random(n)),
        "woba_value": np.where(k, 0.0, rng.random(n)), "woba_denom": 1.0, "plate_x": rng.normal(0, 0.8, n), "plate_z": rng.normal(2.5, 0.8, n)})


def test_translation_candidates_read_no_outcome_and_nothing_after_the_call_up():
    """M5-EB: a within-season prediction is made on the day of the call-up, a date the harness's one-date-per-fold
    checks can't see. The module's own rule: from the outcome season's MLB pitches only what defines a first game, and
    no Triple-A pitch on or after it."""
    mod = runner.load_module("models/baseball/aaa_translation/backtest.py")
    t, rng = SEASON, np.random.default_rng(3)
    batters = np.arange(1, 9)
    aaa, prev = _toy_pitches("aaa", t, batters, rng), _toy_pitches("aaa", t - 1, batters, rng)
    mlb = _toy_pitches("mlb", t, batters, rng, start="06-10")            # everyone's first MLB game: June 10
    debut = pd.Timestamp(f"{t}-06-10")
    heights = pd.DataFrame({"batter": batters, "height_in": 72.0})

    class Fake:
        def __init__(self, tables):
            self.tables, self.asked = tables, {}

        def load(self, table, as_of, columns=None):
            self.asked[table] = columns
            d = self.tables[table]
            return d[columns].reset_index(drop=True) if columns else d.reset_index(drop=True)

    fold = SimpleNamespace(test_season=t, decision_date=pd.Timestamp(f"{t}-12-31"))
    names = {f"statcast_aaa_{t}": aaa, f"statcast_aaa_{t - 1}": prev, f"statcast_mlb_{t}": mlb, "batter_heights": heights}
    clean = Fake(names)
    a = mod._candidates(clean, fold)
    hit = a[a["side"].eq("hitter")]
    assert set(hit["cohort"]) == {"across", "within"} and (hit[hit["cohort"].eq("within")]["pa"] == 70).all()   # Apr 1 - Jun 9
    assert clean.asked[f"statcast_mlb_{t}"] == mod.DEBUT_COLS
    assert not {"events", "launch_speed", "woba_value", "estimated_woba_using_speedangle", "plate_x", "plate_z", "zone"} & set(mod.DEBUT_COLS)

    def scramble(d, late):
        d = d.copy()
        d.loc[late, "description"] = "hit_into_play"
        d.loc[late, "events"] = "home_run"
        d.loc[late, ["launch_speed", "woba_value", "estimated_woba_using_speedangle", "plate_x", "plate_z"]] = 1e6
        return d

    after = pd.to_datetime(aaa["game_date"]) >= debut
    b = mod._candidates(Fake(names | {f"statcast_aaa_{t}": scramble(aaa, after), f"statcast_mlb_{t}": scramble(mlb, mlb.index >= 0)}), fold)
    hb = b[b["side"].eq("hitter")]
    pd.testing.assert_frame_equal(hit.reset_index(drop=True), hb.reset_index(drop=True))
    c = mod._candidates(Fake(names | {f"statcast_aaa_{t}": scramble(aaa, ~after)}), fold)         # the test can see a change
    assert not c[c["side"].eq("hitter") & c["cohort"].eq("within")]["k_pct"].equals(hit[hit["cohort"].eq("within")]["k_pct"])


def test_small_sample_translation_formula_and_committed_constants():
    """M5-EB 2.0.0: the constants the site applies. A rate from few plate appearances is pulled most of the way to the
    Triple-A mean, one from many hardly at all; the range widens as the sample shrinks; and the published file carries
    exactly the committed constants (the pages apply the same formula: tests/test_site.py compares the two)."""
    import json
    import sys

    from sparvi.config import ROOT
    sys.path.insert(0, str(ROOT / "baseball" / "aaa_translation"))
    import small_sample
    P = json.loads(small_sample.PARAMS.read_text(encoding="utf-8"))
    c = P["hitter"]["k_pct"]
    x = c["mean"] + 0.10
    t, half = small_sample.apply(c, [x, x, x], [1, P["min_pa"], 100000])
    at_mean = c["intercept"] + c["slope"] * c["mean"]
    assert abs(t[0] - at_mean) < 0.2 * abs(t[2] - at_mean) and at_mean < t[1] < t[2]
    assert abs(t[2] - (c["intercept"] + c["slope"] * x)) < 1e-3 and half[0] > half[1] > half[2] > 0
    assert P["max_pa"] == 149 and P["min_pa"] == 50
    pub = json.loads((ROOT / "site" / "data" / "baseball" / "translation_fits.json").read_text(encoding="utf-8"))
    assert pub["min_aaa"] == P["max_pa"] + 1
    for side in ("hitter", "pitcher"):
        assert set(pub["small"][side]) == set(P[side]) and not {"stuff_plus", "loc_plus", "pitching_plus", "rv_plus"} & set(P[side])
        for m, k in P[side].items():
            assert all(abs(pub["small"][side][m][f] - k[f]) < 1e-7 for f in ("mean", "k", "talent", "intercept", "slope", "base"))
