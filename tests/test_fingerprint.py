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
