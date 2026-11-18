"""The About page's privacy promise, checked on the rendered site: no page loads a script from another host except
the Umami loader in site/_partials/metadata.html, which runs only on sparvi.io and never records search text."""
import re

import pytest

from sparvi.config import ROOT

DOCS = ROOT / "docs"
pytestmark = pytest.mark.skipif(not (DOCS / "index.html").exists(), reason="site not rendered")
ALLOWED = {"cloud.umami.is"}


def test_no_third_party_scripts():
    bad = []
    for f in DOCS.rglob("*.html"):
        html = f.read_text(encoding="utf-8", errors="ignore")
        hosts = set(re.findall(r'<script[^>]+src="https?://([^/"]+)', html)) | set(re.findall(r'\.src = "https?://([^/"]+)', html))
        bad += [f"{f.relative_to(DOCS)}: {h}" for h in hosts - ALLOWED]
    assert not bad, "\n".join(bad[:20])


def test_umami_loader_is_guarded():
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    assert 'location.hostname !== "sparvi.io"' in html and '"data-do-not-track", "true"' in html
    assert 'track("search", { control: labelOf(el) })' in html, "search events carry the control only, never the text"
