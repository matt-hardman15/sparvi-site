"""Model registry (A2): the real cards check out, and the version / champion-challenger rules hold on a
toy model: a code change without a new version fails the check, a challenger is promoted only if it
beats the champion on the same frozen set, every promotion is logged, and versions are never rewritten."""
import json
from datetime import date

import pytest
import yaml

from sparvi import registry


def test_real_registry_checks_out():
    assert registry.check() == []
    names = {m.name for m in registry.models()}
    assert {"stuff_plus", "location_plus", "pitching_plus", "aaa_translation", "nfl_team_efficiency", "cfb_power_rating"} <= names
    for m in registry.models():
        rec = m.record(m.version)
        assert rec["eval"]["primary"] in rec["metrics"]
        assert m.card["status"] == "champion"


@pytest.fixture()
def toy(tmp_path):
    root = tmp_path
    d = root / "models" / "test" / "toy"
    d.mkdir(parents=True)
    (root / "toy").mkdir()
    (root / "toy" / "fit.py").write_text("print('v1')\n", encoding="utf-8")
    (root / "toy" / "metrics.json").write_text(json.dumps({"mae": 10.0, "set": {"seasons": [2024, 2025]}}), encoding="utf-8")
    card = {"name": "toy", "title": "Toy", "sport": "test", "status": "champion", "version": "1.0.0",
            "code": ["toy/fit.py"], "training_data": [], "artifacts": [],
            "validation": {"file": "toy/metrics.json", "metrics": {"mae": "mae"}},
            "eval": {"set": "toy set", "set_keys": {"seasons": "set.seasons"}, "primary": "mae", "higher_is_better": False}}
    (d / "card.yml").write_text(yaml.safe_dump(card), encoding="utf-8")
    registry.register("toy", "1.0.0", root=root, today=date(2026, 11, 13))
    return root


def set_mae(root, v, seasons=(2024, 2025)):
    (root / "toy" / "metrics.json").write_text(json.dumps({"mae": v, "set": {"seasons": list(seasons)}}), encoding="utf-8")


def test_code_change_needs_a_new_version(toy):
    assert registry.check(toy) == []
    (toy / "toy" / "fit.py").write_text("print('v2')\n", encoding="utf-8")
    probs = registry.check(toy)
    assert len(probs) == 1 and "code changed since v1.0.0" in probs[0] and "toy/fit.py" in probs[0]


def test_line_endings_dont_count_as_a_change(toy):
    (toy / "toy" / "fit.py").write_bytes(b"print('v1')\r\n")
    assert registry.check(toy) == []


def test_better_challenger_is_promoted_and_logged(toy):
    (toy / "toy" / "fit.py").write_text("print('v2')\n", encoding="utf-8")
    set_mae(toy, 9.5)
    registry.register("toy", "1.1.0", challenger=True, root=toy, today=date(2026, 11, 14))
    d = registry.promote("toy", "1.1.0", root=toy, today=date(2026, 11, 15))
    assert d["promoted"] and d["frozen_set_passes"] and not d["live"]["available"]
    card = yaml.safe_load((toy / "models" / "test" / "toy" / "card.yml").read_text(encoding="utf-8"))
    assert card["version"] == "1.1.0" and card["history"][0]["version"] == "1.0.0" and not card["challengers"]
    log = (toy / "review" / "model-changes.md").read_text(encoding="utf-8")
    assert "Toy 1.0.0 → 1.1.0 (promoted)" in log and "mae: 10 → 9.5" in log
    assert registry.check(toy) == []


def test_worse_challenger_stays_a_challenger(toy):
    set_mae(toy, 10.4)
    registry.register("toy", "1.0.1", challenger=True, root=toy)
    d = registry.promote("toy", "1.0.1", root=toy)
    assert not d["promoted"]
    assert registry.get("toy", toy).version == "1.0.0"
    assert not (toy / "review" / "model-changes.md").exists()


def test_different_frozen_set_is_refused(toy):
    set_mae(toy, 9.0, seasons=(2025, 2026))
    registry.register("toy", "2.0.0", challenger=True, root=toy)
    with pytest.raises(ValueError, match="different frozen sets"):
        registry.promote("toy", "2.0.0", root=toy)


def test_versions_are_never_rewritten(toy):
    with pytest.raises(FileExistsError):
        registry.register("toy", "1.0.0", root=toy)
    with pytest.raises(ValueError):
        registry.register("toy", "1.1", root=toy)
