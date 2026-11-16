"""PLAN-seo S3 building blocks: JSON-LD builders and the social preview cards (no render needed).
tests/test_site.py checks the rendered pages carry them."""
import io
import json
import sys

from PIL import Image

from sparvi.config import ROOT

sys.path.insert(0, str(ROOT / "site" / "build"))
import jsonld  # noqa: E402
import social  # noqa: E402


def parse(block: str) -> dict:
    return json.loads(block.split(">", 1)[1].rsplit("</script>", 1)[0])


def test_script_escapes_closing_tags():
    b = jsonld.script({"@type": "Thing", "name": "</script><b>"})
    assert b.count("</script>") == 1 and parse(b)["name"] == "</script><b>"
    assert parse(b)["@context"] == "https://schema.org"


def test_breadcrumbs_are_numbered_from_one():
    b = jsonld.breadcrumbs([("Sparvi", "https://sparvi.io/"), ("Baseball", "https://sparvi.io/baseball/"), ("X", "https://sparvi.io/baseball/x/")])
    assert [i["position"] for i in b["itemListElement"]] == [1, 2, 3] and b["itemListElement"][-1]["item"].endswith("/x/")


def test_article_has_the_fields_google_reads():
    a = jsonld.article("https://sparvi.io/a/", "H" * 200, "d", "https://sparvi.io/a/og.png", "2026-11-10", "2026-11-12")
    assert len(a["headline"]) <= 110 and a["datePublished"] == "2026-11-10" and a["dateModified"] == "2026-11-12"
    assert a["publisher"]["logo"]["url"].endswith("/assets/logo.png") and a["author"]["name"] == "Sparvi"


def test_dataset_license_only_where_the_source_grants_one():
    assert jsonld.dataset("u", "n", "d", "nfl", "2016/2026", "2026-11-01")["license"].startswith("https://creativecommons.org/")
    for sport in ("baseball", "cfb"):
        assert "license" not in jsonld.dataset("u", "n", "d", sport, "2026", "2026-11-01")


def test_cards_are_1200x630_small_and_deterministic():
    a = social.card("Ryan Ramsey", "OMA · Triple-A · 2026", "Statcast profile", [("Stuff+", "81"), ("K%", "20.9%"), ("BB%", "12.1%")])
    assert a == social.card("Ryan Ramsey", "OMA · Triple-A · 2026", "Statcast profile", [("Stuff+", "81"), ("K%", "20.9%"), ("BB%", "12.1%")])
    assert Image.open(io.BytesIO(a)).size == (1200, 630) and len(a) < social.MAX_BYTES
    long = social.card("A title long enough that it has to wrap onto a second line and then some more words", "k", "x " * 200,
                       thumb=ROOT / "site" / "baseball" / "stuff-plus" / "thumb.png")
    assert Image.open(io.BytesIO(long)).size == (1200, 630) and len(long) < social.MAX_BYTES
    assert Image.open(io.BytesIO(social.logo())).size == (512, 512)


def test_set_image_points_both_tags_at_the_card():
    h = ('<head><meta property="og:image" content="old.png"><meta name="twitter:card" content="summary">'
         '<meta name="twitter:image" content="old.png"></head>')
    out = social.set_image(h, "https://sparvi.io/x/og.png")
    assert out.count("https://sparvi.io/x/og.png") == 2 and "old.png" not in out
    assert 'content="summary_large_image"' in out and 'og:image:width" content="1200"' in out
