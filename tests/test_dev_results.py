"""DEV-CI: a development run's result files survive the trip through a job log, and nothing else can ride along."""
import io
import json
import sys

import pytest

from sparvi.config import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import dev_results  # noqa: E402


def test_result_files_round_trip_through_a_log(tmp_path, monkeypatch):
    monkeypatch.setattr(dev_results, "ROOT", tmp_path)
    out = tmp_path / "baseball" / "pitch_quality" / "output"
    out.mkdir(parents=True)
    import random
    rnd = random.Random(0)
    big = {"setups": {f"s{i}": {"points": rnd.random(), "note": "é"} for i in range(2000)}}    # several log lines long
    (out / "development_location_stability.json").write_text(json.dumps(big, indent=1), encoding="utf-8")
    (out / "development.json").write_text("{}\n", encoding="utf-8")
    buf = io.StringIO()
    assert dev_results.emit([str(out / "development_location_stability.json"), str(out / "development.json")], buf) == 0
    lines = buf.getvalue().splitlines()
    assert len(lines) > 3 and all(len(x) < 3200 for x in lines)
    # A job log puts a job name and a timestamp before each line and other output around them.
    log = "\n".join(["develop\tRun\t2026-11-23T00:00:00.0000000Z starting"] +
                    [f"develop\tResults\t2026-11-23T00:00:0{i % 10}.0000000Z {x}" for i, x in enumerate(lines)] + ["done"])
    got = dev_results.parse(log)
    assert set(got) == {"baseball/pitch_quality/output/development_location_stability.json", "baseball/pitch_quality/output/development.json"}
    assert json.loads(got["baseball/pitch_quality/output/development_location_stability.json"]) == big
    with pytest.raises(ValueError, match="pieces"):
        dev_results.parse("\n".join(log.splitlines()[:3]))


def test_only_development_result_files_are_carried(tmp_path, monkeypatch):
    monkeypatch.setattr(dev_results, "ROOT", tmp_path)
    for rel in ("data/raw/statcast_mlb_2025.parquet", "baseball/pitch_quality/output/dev_cache/x.json", "baseball/pitch_quality/train.py"):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")
        assert dev_results.emit([str(f)], io.StringIO()) == 1
    with pytest.raises(ValueError, match="not a development result"):
        dev_results.parse("DEV-RESULT sparvi/config.py 1/1 eJwDAAAAAAE=")
