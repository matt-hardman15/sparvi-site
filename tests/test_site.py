"""Smoke-test the rendered site: every page loads, every control works, nothing logs an error.

Needs docs/ (run `cd site && quarto render`) and a browser:
    uv run playwright install chromium          # CI does this
    SPARVI_BROWSER_CHANNEL=chrome uv run pytest  # or use an installed Chrome locally
"""
from __future__ import annotations

import functools
import http.server
import json
import os
import threading

import pytest

from sparvi.config import ROOT

DOCS = ROOT / "docs"
DATA = ROOT / "site" / "data"
DATA_BUDGET_MB = 25  # GitHub Pages and git history both pay for every byte; revisit before raising

pytestmark = pytest.mark.skipif(not (DOCS / "index.html").exists(), reason="docs/ not built")


# ---- static checks ------------------------------------------------------------------------
def test_no_unresolved_prose_numbers():
    bad = [p.relative_to(DOCS).as_posix() for p in DOCS.rglob("*.html") if "?meta:" in p.read_text(encoding="utf-8")]
    assert not bad, f"pages with unresolved {{{{< meta >}}}} values: {bad}"


def test_site_data_is_utf8_json():
    for p in DATA.rglob("*.json"):
        json.loads(p.read_bytes().decode("utf-8"))  # raises on cp1252 bytes or bad JSON


def test_site_data_within_budget():
    mb = sum(p.stat().st_size for p in DATA.rglob("*") if p.is_file()) / 1e6
    assert mb < DATA_BUDGET_MB, f"site/data is {mb:.1f} MB (budget {DATA_BUDGET_MB} MB)"


# ---- browser checks -----------------------------------------------------------------------
@pytest.fixture(scope="module")
def base_url():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DOCS))
    handler.log_message = lambda *a: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture(scope="module")
def browser():
    sync = pytest.importorskip("playwright.sync_api")
    with sync.sync_playwright() as pw:
        b = pw.chromium.launch(channel=os.environ.get("SPARVI_BROWSER_CHANNEL") or None)
        yield b
        b.close()


def wait(p, ms=350):
    p.wait_for_timeout(ms)


def seg(p, scope, text):
    p.locator(f"{scope} .seg button", has_text=text).first.click(); wait(p)


def select(p, scope, label, value):
    p.locator(f"{scope} label", has_text=label).locator("select").select_option(value); wait(p)


def search(p, scope, text):
    p.locator(f"{scope} input[type=search]").first.fill(text); wait(p)


def rows(p, scope):
    return p.locator(f"{scope} table.s-table tbody tr").count()


def nth_option(p, scope, label, n):
    return p.locator(f"{scope} label", has_text=label).locator("option").nth(n).get_attribute("value")


def act_stuff_plus(p):
    assert "r = " in p.text_content("#cap1")
    seg(p, "#c1", "results"); search(p, "#c1", "a")
    seg(p, "#c3", "Breaking balls"); seg(p, "#c3", "Offspeed")
    assert p.locator("#p3 svg rect").count() >= 12


def act_shape(p):
    assert p.locator("#p0 svg circle").count() == 6
    for m in ["Ride (IVB)", "Arm-side run", "Velocity"]:
        seg(p, "#c1", m)
    select(p, "#c1", "Pitches", "fb"); select(p, "#c1", "Show", "1"); search(p, "#c1", "a")
    assert "pitch types" in p.text_content("#cap1")


def act_two_week(p):
    seg(p, "#c0", "Pitchers")
    seg(p, "#c1", "Pitchers")
    for m in ["Whiff rate", "Walk rate", "xwOBA allowed", "Fastball velocity"]:
        seg(p, "#c1", m)
    assert "r = " in p.text_content("#cap1")


def num_input(p):
    return p.locator("#c input[type=number]")


def col_values(p, scope, header):
    """Text of every visible cell in the column whose header is `header`."""
    heads = [h.strip() for h in p.locator(f"{scope} table.s-table thead th").all_text_contents()]
    i = heads.index(header) + 1
    return p.locator(f"{scope} table.s-table tbody tr td:nth-child({i})").all_text_contents()


def role_on(p, label):
    return p.locator("#c .seg button.on", has_text=label).count() == 1


def query(p):
    from urllib.parse import parse_qs, urlparse
    return {k: v[0] for k, v in parse_qs(urlparse(p.url).query).items()}


def act_leaderboard(p):
    assert rows(p, "#t") == 25
    assert "500+ pitches for starters, 250+ for relievers" in p.text_content("#scap")
    assert p.locator("#c .seg button", has_text="Swing").count() == 1  # MLB has swingmen
    p.locator("#t th", has_text="RV+").first.click(); wait(p)
    p.locator(".pager button", has_text="Next").click(); wait(p)
    # Role filter: table, caption, and URL follow it.
    seg(p, "#c", "Relievers")
    assert set(col_values(p, "#t", "Role")) == {"RP"}
    assert "250+ pitches (relievers minimum)" in p.text_content("#scap")
    assert query(p).get("role") == "RP"
    # ...and it survives switching level, team, and view.
    seg(p, "#c", "Triple-A")
    assert role_on(p, "Relievers") and set(col_values(p, "#t", "Role")) == {"RP"}
    assert "150+ pitches (relievers minimum)" in p.text_content("#scap")
    select(p, "#c", "Team", nth_option(p, "#c", "Team", 1))
    assert role_on(p, "Relievers") and set(col_values(p, "#t", "Role")) <= {"RP"}
    seg(p, "#c", "Individual pitches")
    assert role_on(p, "Relievers") and set(col_values(p, "#t", "Role")) <= {"RP"}
    q = query(p)
    assert q.get("role") == "RP" and q.get("level") == "AAA" and q.get("view") == "pitches" and q.get("team")
    # Reloading the URL restores the view.
    p.reload(wait_until="networkidle"); wait(p, 800)
    assert role_on(p, "Relievers") and set(col_values(p, "#t", "Role")) <= {"RP"}
    select(p, "#c", "Pitch group", "breaking")
    seg(p, "#c", "Pitchers"); seg(p, "#c", "MLB"); seg(p, "#c", "Starters")
    assert set(col_values(p, "#t", "Role")) == {"SP"} and "500+ pitches (starters minimum)" in p.text_content("#scap")
    num_input(p).fill("1500"); num_input(p).press("Enter"); wait(p)
    assert "1500+ pitches" in p.text_content("#scap") and query(p).get("min") == "1500"
    p.set_viewport_size({"width": 700, "height": 900}); wait(p, 600)
    assert p.locator("#scatter svg").count() == 1


def act_triple_a(p):
    assert rows(p, "#t") == 25
    assert "r = " in p.text_content("#scap")
    assert p.locator("#c .seg button", has_text="Starters").count() == 0  # no role filter for hitters
    select(p, "#c", "Team", nth_option(p, "#c", "Team", 5))
    assert p.locator("#flags .flag-list").count() == 1
    seg(p, "#c", "Pitchers"); p.locator("#t th", has_text="BB%").click(); wait(p)
    assert "300+ pitches for starters, 150+ for relievers" in p.text_content("#scap")
    seg(p, "#c", "Starters")
    assert set(col_values(p, "#t", "Role")) <= {"SP"} and query(p).get("role") == "SP"
    select(p, "#c", "Team", "")
    assert role_on(p, "Starters") and set(col_values(p, "#t", "Role")) == {"SP"}
    assert "300+ pitches (starters minimum)" in p.text_content("#scap")
    p.reload(wait_until="networkidle"); wait(p, 800)
    assert role_on(p, "Starters") and set(col_values(p, "#t", "Role")) == {"SP"}
    search(p, "#c", "a"); seg(p, "#c", "Hitters")
    assert "role" not in query(p)
    p.set_viewport_size({"width": 700, "height": 900}); wait(p, 600)
    assert p.locator("#scatter svg").count() == 1


def first_id(file, key, pred):
    return next(r[key] for r in json.loads((DATA / "baseball" / file).read_text(encoding="utf-8")) if pred(r))


def player_hitter():
    return first_id("aaa_hitters.json", "player_id", lambda r: r["pa"] >= 300)


def player_pitcher():
    return first_id("aaa_pitchers.json", "player_id", lambda r: r["bf"] >= 300)


def act_player(*sections, role_tag=False):
    def check(p):
        h2 = p.locator("#root h2").all_text_contents()
        for s in sections:
            assert s in h2, f"missing section {s!r}: {h2}"
        if role_tag:
            import re
            tags = p.locator("#root .role-tag").all_text_contents()
            assert tags and all(re.fullmatch(r"(SP|RP|Swing) · \d+ G, \d+ GS", t) for t in tags), tags
    return check


PAGES = [
    ("home", "/", None),
    ("baseball", "/baseball/", None),
    ("stuff-plus", "/baseball/stuff-plus/", act_stuff_plus),
    ("shape-changes", "/baseball/shape-changes/", act_shape),
    ("two-week", "/baseball/two-week-signal/", act_two_week),
    ("stuff-leaderboard", "/baseball/tools/stuff-leaderboard.html", act_leaderboard),
    ("triple-a", "/baseball/tools/triple-a.html", act_triple_a),
    ("player-hitter", lambda: f"/baseball/player.html?id={player_hitter()}", act_player("Hitting · Triple-A")),
    ("player-pitcher", lambda: f"/baseball/player.html?id={player_pitcher()}", act_player("Pitching · Triple-A", role_tag=True)),
    ("player-missing", "/baseball/player.html?id=1", lambda p: "No " in p.text_content("#root")),
    ("glossary", "/glossary.html", None),
    ("about", "/about.html", None),
]


@pytest.mark.parametrize("name,path,act", PAGES, ids=[p[0] for p in PAGES])
def test_page(browser, base_url, name, path, act):
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    page = ctx.new_page()
    errors = []
    page.on("console", lambda m: m.type in ("error", "warning") and errors.append(f"console.{m.type}: {m.text}"))
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.on("requestfailed", lambda r: errors.append(f"requestfailed: {r.url}"))
    page.on("response", lambda r: r.status >= 400 and errors.append(f"HTTP {r.status}: {r.url}"))
    try:
        page.goto(base_url + (path() if callable(path) else path), wait_until="networkidle")
        wait(page, 800)
        assert page.locator(".loading").count() == 0, "page still shows Loading…"
        if act:
            act(page)
            wait(page, 300)
    finally:
        ctx.close()
    assert not errors, "\n".join(errors)
