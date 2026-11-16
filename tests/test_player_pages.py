"""Static player pages (PLAN-seo S2): URLs, the URL registry, summaries, page metadata, and the budget check.

Runs on the committed site/data (no render needed). tests/test_site.py covers the rendered pages.
"""
import html
import importlib.util
import json
import re
import sys
from collections import Counter

import pytest

from sparvi.config import ROOT
from sparvi.urls import player_path, slug, update_registry

BUILD = ROOT / "site" / "build"
DATA = ROOT / "site" / "data" / "baseball"
sys.path.insert(0, str(BUILD))


def load(name):
    spec = importlib.util.spec_from_file_location(name, BUILD / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sm, pp, budget = load("summarize"), load("player_pages"), load("check_budget")
META = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
PLAYERS = [json.loads((DATA / "players" / f"{i}.json").read_text(encoding="utf-8"))
           for i in json.loads((DATA / "players_index.json").read_text(encoding="utf-8"))]


# ---- URLs ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,want", [
    ("Ryan Ramsey", "ryan-ramsey"), ("José Ramírez", "jose-ramirez"), ("Ha-Seong Kim", "ha-seong-kim"),
    ("Ke'Bryan Hayes", "ke-bryan-hayes"), ("A.J. Puk", "a-j-puk"), ("Luis García Jr.", "luis-garcia-jr"),
    ("Kala'i Rosario", "kala-i-rosario"), ("Jasson Domínguez", "jasson-dominguez"), ("", "player"),
])
def test_slug(name, want):
    assert slug(name) == want


def test_registry_only_grows(tmp_path):
    f = tmp_path / "urls.json"
    reg = update_registry(f, {1: "Ann Lee", 22: "Bo Diaz"})
    assert reg == {"1": ["ann-lee-1"], "22": ["bo-diaz-22"]}
    f.write_text(json.dumps(reg), encoding="utf-8")
    reg = update_registry(f, {1: "Ann Lee-Park"})            # renamed; 22 has no data this season
    assert reg == {"1": ["ann-lee-1", "ann-lee-park-1"], "22": ["bo-diaz-22"]}
    f.write_text(json.dumps(reg), encoding="utf-8")
    assert update_registry(f, {1: "Ann Lee"})["1"] == ["ann-lee-park-1", "ann-lee-1"]   # current is always last


def test_committed_registry_covers_every_player():
    reg = json.loads((DATA / "player_urls.json").read_text(encoding="utf-8"))
    for p in PLAYERS:
        assert reg[str(p["id"])][-1] == player_path(p["id"], p["name"]), p["name"]
    paths = [x for v in reg.values() for x in v]
    assert len(paths) == len(set(paths)), "two players share a page path"


# ---- summaries and static tables ---------------------------------------------------------------
NUM = re.compile(r"(?<![\w.])[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)%?")


def page_text(p):
    th = {lvl: pp.short(p["timeline"][lvl]["dates"][-1]) for lvl in ("AAA", "MLB") if lvl in (p.get("timeline") or {})}
    return html.unescape(re.sub(r"<[^>]+>", " ", sm.static_body(p, th) + " " + pp.kicker(p, META["season"])))


def test_every_summary_number_is_on_the_page():
    """PLAN-seo S2: every number in a summary equals a number in the page's own tables."""
    checked = 0
    for p in PLAYERS:
        if not sm.indexable(sm.final(p)):
            continue
        text, tokens = page_text(p), set(NUM.findall(page_text(p)))
        for n in NUM.findall(sm.summary(p, META["season"])):
            ordinal_free = n.rstrip("%")
            assert n in tokens or ordinal_free in tokens or n in text, f"{p['name']}: {n} not in the page's tables"
        checked += 1
    assert checked > 1500


def test_percentile_ordinals_match_the_table():
    for p in PLAYERS:
        s = sm.summary(p, META["season"])
        for n in re.findall(r"\((\d+)(?:st|nd|rd|th)\)|the (\d+)(?:st|nd|rd|th) percentile", s):
            v = next(x for x in n if x)
            assert re.search(rf">{v}</td>", sm.static_body(p, {"AAA": "x", "MLB": "x"})), f"{p['name']}: percentile {v}"


def test_summaries_are_deterministic_and_plain():
    hype = re.compile(r"game-changing|unlock|dive in|leverage|elite|incredible", re.I)
    for p in PLAYERS[:300]:
        a, b = sm.summary(p, 2026), sm.summary(p, 2026)
        assert a == b and not hype.search(a) and "  " not in a


def test_small_samples_are_noindex_and_say_so():
    small = [p for p in PLAYERS if not sm.indexable(sm.final(p))]
    assert small and all("small" in sm.summary(p, 2026) for p in small)


def test_titles_and_descriptions_fit_and_are_unique():
    n = Counter(p["name"] for p in PLAYERS)
    t, d = Counter(), Counter()
    for p in PLAYERS:
        title, desc = pp.titles(p, sm.final(p), META["season"], n[p["name"]] > 1)
        assert len(title) + len(" – Sparvi") <= 60, title
        assert 120 <= len(desc) <= 155, (len(desc), desc)
        t[title] += 1
        d[desc] += 1
    assert t.most_common(1)[0][1] == 1 and d.most_common(1)[0][1] == 1


def test_fill_escapes_and_marks_noindex():
    shell = "<head><title>ZZTITLEZZ – Sparvi</title></head><h1>ZZHONEZZ</h1><p>ZZDESCRIPTIONZZ</p>ZZBODYZZ"
    out = pp.fill(shell, 'A "B" & C', "<x>", "d", "<b>body</b>", "https://sparvi.io/baseball/players/a-1/", noindex=True)
    assert "A &quot;B&quot; &amp; C" in out and "&lt;x&gt;" in out and "<b>body</b>" in out
    assert '<meta name="robots" content="noindex">' in out and 'rel="canonical"' in out


def test_embedded_data_cannot_close_its_script_tag():
    p = {**PLAYERS[0], "name": "</script><script>alert(1)</script>"}
    b = pp.body(p, 2026, sm.final(p))
    data = b[b.index('id="page-data">'):]
    assert data.count("</script>") == 1


# ---- hosting budget ------------------------------------------------------------------------------
def test_budget_thresholds():
    cfg = {"host": "h", "warn_at": 0.6, "fail_at": 0.75, "hosts": {"h": {"max_bytes": 100, "max_files": 10}}}
    m = lambda b, f: {"bytes": b, "files": f, "largest_file": ("x", 1)}  # noqa: E731
    assert budget.check(m(50, 5), cfg)[:2] == ([], [])
    warn, fail, _ = budget.check(m(60, 5), cfg)
    assert warn and not fail
    warn, fail, _ = budget.check(m(10, 8), cfg)
    assert fail and "files" in fail[0]
