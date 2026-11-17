"""PLAN-seo S5: first-mention entity links, "Mentioned in", and the internal link check (no render needed)."""
import importlib.util
import sys

from sparvi.config import ROOT

sys.path.insert(0, str(ROOT / "site" / "build"))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "site" / "build" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mentions, linkcheck = load("mentions"), load("linkcheck")
NAMES = {"Georgia": "cfb/teams/georgia/", "Georgia Tech": "cfb/teams/georgia-tech/", "José Urquidy": "baseball/players/jose-urquidy-664353/"}


def page(body: str) -> str:
    return f"<html><head><title>Georgia</title></head><body><main>{body}</main><footer>Georgia</footer></body></html>"


def test_links_only_the_first_mention_and_prefers_longer_names():
    out, done = mentions.link_first_mentions(page("<p>Georgia Tech beat Georgia. Georgia then won.</p>"), NAMES)
    assert done == ["Georgia Tech", "Georgia"]
    assert out.count('href="/cfb/teams/georgia/"') == 1 and out.count('href="/cfb/teams/georgia-tech/"') == 1
    assert "Georgia then won" in out


def test_leaves_headings_links_scripts_and_the_rest_of_the_page_alone():
    body = ('<h2>Georgia</h2><a href="/x/">Georgia</a><script>var t = "Georgia";</script>'
            "<code>Georgia</code><p>Back to Georgia.</p>")
    out, done = mentions.link_first_mentions(page(body), NAMES)
    assert done == ["Georgia"]
    assert "<h2>Georgia</h2>" in out and '"Georgia";</script>' in out and "<code>Georgia</code>" in out
    assert '<a href="/cfb/teams/georgia/" class="entity-link">Georgia</a>.' in out
    assert "<title>Georgia</title>" in out and "<footer>Georgia</footer>" in out


def test_an_existing_link_counts_as_the_first_mention():
    out, done = mentions.link_first_mentions(page('<p><a href="../../cfb/teams/georgia/">UGA</a> and Georgia.</p>'), NAMES)
    assert done == [] and out.count("cfb/teams/georgia/") == 1


def test_accented_and_whole_word_names():
    out, done = mentions.link_first_mentions(page("<p>José Urquidy and José Urquidyson; Georgians.</p>"), NAMES)
    assert done == ["José Urquidy"] and "Georgians" in out and 'href="/cfb/teams/georgia/"' not in out


def test_linkcheck_resolves_relative_absolute_and_full_urls(tmp_path, monkeypatch):
    monkeypatch.setattr(linkcheck, "OUT", tmp_path)
    (tmp_path / "a" / "b").mkdir(parents=True)
    p = tmp_path / "a" / "b" / "index.html"
    assert linkcheck.resolve(p, "../c.html") == tmp_path / "a" / "b" / ".." / "c.html"
    assert linkcheck.resolve(p, "/x/") == tmp_path / "x" / "index.html"
    assert linkcheck.resolve(p, "https://sparvi.io/x/y.png?v=1") == tmp_path / "x" / "y.png"
    for ext in ("https://example.com/", "mailto:a@b.c", "#top", "data:image/png;base64,AAAA"):
        assert linkcheck.resolve(p, ext) is None


def test_linkcheck_fails_on_a_broken_link_and_an_orphan(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(linkcheck, "OUT", tmp_path)
    (tmp_path / "index.html").write_text('<a href="/ok/">ok</a>', encoding="utf-8")
    (tmp_path / "ok").mkdir()
    (tmp_path / "ok" / "index.html").write_text('<a href="/">home</a>', encoding="utf-8")
    (tmp_path / "lone").mkdir()
    (tmp_path / "lone" / "index.html").write_text('<a href="/">home</a>', encoding="utf-8")
    (tmp_path / "sitemap.xml").write_text("<loc>https://sparvi.io/</loc><loc>https://sparvi.io/ok/</loc>", encoding="utf-8")
    assert linkcheck.main() == 0
    (tmp_path / "sitemap.xml").write_text("<loc>https://sparvi.io/ok/</loc><loc>https://sparvi.io/lone/</loc>", encoding="utf-8")
    assert linkcheck.main() == 1 and "orphan: https://sparvi.io/lone/" in capsys.readouterr().err
    (tmp_path / "ok" / "index.html").write_text('<a href="/missing/">x</a><a href="/lone/">l</a>', encoding="utf-8")
    assert linkcheck.main() == 1 and "broken: /missing/" in capsys.readouterr().err
