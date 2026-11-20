"""Every third-party GitHub Action is pinned to a full commit SHA (OPS-1): a moved tag can't change
what runs with write access to the repo. Dependabot (github-actions) keeps the pins current."""
import re

import yaml

from sparvi.config import ROOT

USES = re.compile(r"^\s*(?:-\s*)?uses:\s*(\S+)", re.M)


def test_actions_are_pinned_to_commit_shas():
    loose = []
    for wf in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        for ref in USES.findall(wf.read_text(encoding="utf-8")):
            if ref.startswith("./"):
                continue                          # this repo's own reusable workflows
            if not re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", ref):
                loose.append(f"{wf.name}: {ref}")
    assert not loose, "pin these to a commit SHA (with a # vX.Y.Z comment):\n" + "\n".join(loose)


def _site_paths() -> re.Pattern:
    wf = (ROOT / ".github" / "workflows" / "check.yml").read_text(encoding="utf-8")
    return re.compile(re.search(r"SITE_PATHS: '(.+)'", wf).group(1))


def test_site_check_runs_for_everything_the_site_reads():
    """check.yml skips the render and browser test only when a pull request touches none of these."""
    pat = _site_paths()
    must_run = ["site/baseball/stuff-plus/index.qmd", "site/data/nfl/meta.json", "site/assets/js/sparvi.js",
                "site/build/seo.py", "site/_quarto.yml", "sparvi/io.py", "sparvi/registry.py", "sparvi.toml",
                "baseball/export_site_data.py", "nfl/export_site_data.py", "cfb/export_site_data.py", "nfl/team_pages.py",
                "cfb/team_pages.py", "football/common/pages.py", "tests/test_site.py", "tests/conftest.py", "uv.lock",
                "pyproject.toml", "build.sh", ".github/workflows/check.yml"]
    assert [f for f in must_run if not pat.search(f)] == []
    may_skip = ["models/baseball/stuff_plus/card.yml", "models/nfl/team_efficiency/versions/1.0.1.json",
                "nfl/team_efficiency/adjusted.py", "review/model-changes.md", "backtest/runner.py", "README.md",
                "tests/test_registry.py", "ingest/nfl/pull.py", ".github/workflows/nfl.yml"]
    assert [f for f in may_skip if pat.search(f)] == []


def test_site_check_never_skips_when_change_detection_fails():
    wf = (ROOT / ".github" / "workflows" / "check.yml").read_text(encoding="utf-8")
    assert "always() && (needs.changes.result != 'success' || needs.changes.outputs.site == 'true')" in wf


def test_backtest_gets_the_history_season_from_cache_before_r2():
    """PQ-TEST-2: the Stuff+ backtest trains on a season the refresh never pulls. Pull requests without secrets
    (Dependabot) must find it in the cache; R2 is read only when the cache misses."""
    steps = {s.get("name"): s for s in yaml.safe_load((ROOT / ".github/workflows/backtest.yml").read_text(encoding="utf-8"))
             ["jobs"]["backtest"]["steps"]}
    cache, r2, save = (steps[n] for n in ("Restore the history season (cache)", "Restore the history season (R2 snapshots)",
                                           "Save the history season"))
    assert cache["with"]["key"] == save["with"]["key"] and cache["with"]["path"] == save["with"]["path"]
    assert "steps.hist.outputs.cache-hit != 'true'" in r2["if"] and "steps.hist.outputs.cache-hit != 'true'" in save["if"]
    assert "ingest.baseball.history restore" in r2["run"]
    names = list(steps)
    assert names.index("Restore the history season (R2 snapshots)") < names.index("Pull missing batter heights (MLB Stats API)")
