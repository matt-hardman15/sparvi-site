"""Every third-party GitHub Action is pinned to a full commit SHA (OPS-1): a moved tag can't change
what runs with write access to the repo. Dependabot (github-actions) keeps the pins current."""
import re

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
