"""The refresh skips rebuilding when this fingerprint is unchanged, so it must cover the code and
data the build reads and leave out what the build writes."""
import importlib.util
import subprocess

from sparvi.config import ROOT

spec = importlib.util.spec_from_file_location("build_fingerprint", ROOT / "scripts" / "build_fingerprint.py")
bf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bf)


def _tracked():
    out = subprocess.run(["git", "ls-files", "--", *bf.CODE, *bf.EXCLUDE], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split()
    return set(out)


def test_covers_code_but_not_outputs():
    files = _tracked()
    assert "baseball/pitch_quality/train.py" in files
    assert "baseball/export_site_data.py" in files
    assert "sparvi.toml" in files and "uv.lock" in files
    assert not any("/output/" in f for f in files)
    assert bf.STORED.relative_to(ROOT).as_posix() not in files


def test_per_input_record_is_left_out_and_decides_the_fingerprint():
    """FP-DIAG: the per-input record must not feed the fingerprint it explains, and anything that changes the
    fingerprint shows in the record."""
    assert bf.INPUTS.relative_to(ROOT).as_posix() not in _tracked()
    per_input = {"data/raw/a.parquet": "11", "data/raw/b.parquet": "22", "baseball/x.py": "100644 abc"}
    code = "100644 abc 0\tbaseball/x.py\n"
    fp = bf.fingerprint((per_input, code))
    assert fp == bf.fingerprint((dict(per_input), code))
    assert fp != bf.fingerprint(({**per_input, "data/raw/b.parquet": "23"}, code))
    assert fp != bf.fingerprint((per_input, code.replace("abc", "abd")))


def test_fingerprint_recipe_is_unchanged():
    """The stored fingerprints were made by hashing each raw file's name and sha256, then git's listing of the code."""
    import hashlib
    per_input, code = {"data/raw/a.parquet": "11", "baseball/x.py": "100644 abc"}, "100644 abc 0\tbaseball/x.py\n"
    h = hashlib.sha256()
    h.update(b"a.parquet\x0011\n")
    h.update(code.encode())
    assert bf.fingerprint((per_input, code)) == h.hexdigest()


def test_a_mismatch_names_what_changed(tmp_path, monkeypatch):
    import json
    old = {"data/raw/a.parquet": "11", "data/raw/gone.parquet": "33", "baseball/x.py": "100644 abc", "uv.lock": "100644 l1"}
    new = {"data/raw/a.parquet": "12", "data/raw/new.parquet": "44", "baseball/x.py": "100644 abc", "uv.lock": "100644 l2"}
    assert bf.differences(old, new) == ["added: data/raw/new.parquet", "removed: data/raw/gone.parquet",
                                        "changed: data/raw/a.parquet", "changed: uv.lock"]
    monkeypatch.setattr(bf, "INPUTS", tmp_path / "build_inputs.json")
    assert "no per-input record yet" in bf.explain(new)[0]
    bf.INPUTS.write_text(json.dumps(old), encoding="utf-8")
    assert bf.explain(new) == bf.differences(old, new)
    assert "no input differs" in bf.explain(old)[0]
