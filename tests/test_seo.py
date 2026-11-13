"""site/build/seo.py: clean canonical URLs, and reruns that find the already-rewritten sitemap."""
import importlib.util

from sparvi.config import ROOT

spec = importlib.util.spec_from_file_location("seo", ROOT / "site" / "build" / "seo.py")
seo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seo)


def test_clean_url_drops_index_html():
    assert seo.clean_url("https://sparvi.io/index.html") == "https://sparvi.io/"
    assert seo.clean_url("https://sparvi.io/nfl/teams/kc/index.html") == "https://sparvi.io/nfl/teams/kc/"
    assert seo.clean_url("https://sparvi.io/glossary.html") == "https://sparvi.io/glossary.html"


def test_html_path_accepts_clean_and_raw_urls():
    for loc in ("https://sparvi.io/nfl/teams/kc/", "https://sparvi.io/nfl/teams/kc/index.html"):
        assert seo.html_path(loc) == seo.OUT / "nfl/teams/kc/index.html"
    assert seo.html_path("https://sparvi.io/") == seo.OUT / "index.html"


def test_source_for_maps_to_qmd():
    assert seo.source_for("nfl/teams/kc/index.html") == ROOT / "site" / "nfl/teams/kc/index.qmd"
