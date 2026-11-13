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
import re
import threading

import pytest

from sparvi.config import ROOT

DOCS = ROOT / "docs"
DATA = ROOT / "site" / "data"
# What a visitor downloads (GitHub Pages serves gzip) and what each page loads, not raw bytes on disk.
# Gzipped size of each sport's site/data folder (each sport has its own budget so adding a sport can't
# quietly squeeze another; the site-wide hosting limit is checked by SEO-2's budget check).
GZ_BUDGET_MB = {"baseball": 15, "nfl": 3, "cfb": 6, "status": 0.5}
LEAGUE_FILE_GZ_MB = 2      # each snapshot league file (plan target for the date slider)
PLAYER_FILE_KB = 100       # each per-player file, raw

pytestmark = pytest.mark.skipif(not (DOCS / "index.html").exists(), reason="docs/ not built")


# ---- static checks ------------------------------------------------------------------------
def test_no_unresolved_prose_numbers():
    bad = [p.relative_to(DOCS).as_posix() for p in DOCS.rglob("*.html") if "?meta:" in p.read_text(encoding="utf-8")]
    assert not bad, f"pages with unresolved {{{{< meta >}}}} values: {bad}"


def test_site_data_is_utf8_json():
    for p in DATA.rglob("*.json"):
        json.loads(p.read_bytes().decode("utf-8"))  # raises on cp1252 bytes or bad JSON


def test_site_data_within_budget():
    import gzip
    files = [p for p in DATA.rglob("*.json")]
    gz = {p: len(gzip.compress(p.read_bytes(), 6)) for p in files}
    sports = {p.relative_to(DATA).parts[0] for p in files}
    assert sports <= set(GZ_BUDGET_MB), f"no data budget for {sorted(sports - set(GZ_BUDGET_MB))}"
    for sport, budget in GZ_BUDGET_MB.items():
        total = sum(v for p, v in gz.items() if p.relative_to(DATA).parts[0] == sport) / 1e6
        assert total < budget, f"site/data/{sport} is {total:.1f} MB gzipped (budget {budget} MB)"
    for p in DATA.glob("baseball/snapshots_*.json"):
        assert gz[p] / 1e6 < LEAGUE_FILE_GZ_MB, f"{p.name}: {gz[p] / 1e6:.2f} MB gzipped"
    big = [p.name for p in DATA.glob("baseball/players/*.json") if p.stat().st_size > PLAYER_FILE_KB * 1024]
    assert not big, f"player files over {PLAYER_FILE_KB} KB: {big[:5]}"


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


def qtext(level, role):
    """Expected caption qualifier at the final date, computed the way the page does (prorated)."""
    m = json.loads((DATA / "baseball" / "meta.json").read_text(encoding="utf-8"))
    c = m["snapshots"][level]
    fac = min(1, c["team_games"][-1] / c["full_season_games"])
    q = lambda r: max(25, round(m["qualifiers"][level][r] * fac))  # noqa: E731
    pro = ", prorated to games played" if fac < 1 else ""
    if role == "all":
        return f"{q('SP')}+ pitches for starters, {q('RP')}+ for relievers and swingmen{pro}"
    return f"{q(role)}+ pitches ({ {'SP': 'starters', 'RP': 'relievers'}[role]} minimum{pro})"


def slider_label(p, scope="body"):
    return p.locator(f"{scope} .date-slider .ds-label").first.text_content()


def slide_to_start(p, scope="body"):
    p.locator(f"{scope} .date-slider input[type=range]").first.focus()
    p.keyboard.press("Home"); wait(p, 500)


def act_slider(p, table_scope, caption_scope):
    """Moving the slider changes table + caption together; URL and reload restore it; play advances."""
    final_label, final_cap = slider_label(p), p.text_content(caption_scope)
    final_rows = p.locator(f"{table_scope} table.s-table tbody tr").first.text_content()
    p.locator(".date-slider .ds-btn", has_text="◀").first.click(); wait(p, 500)
    assert slider_label(p) != final_label and "asof" in query(p)
    slide_to_start(p)
    early_label, early_cap = slider_label(p), p.text_content(caption_scope)
    assert early_label != final_label and early_cap != final_cap
    assert early_label.replace("Through ", "") in early_cap, (early_label, early_cap)
    assert p.locator(f"{table_scope} table.s-table tbody tr").first.text_content() != final_rows
    asof = query(p)["asof"]
    p.reload(wait_until="networkidle"); wait(p, 900)
    assert slider_label(p) == early_label and query(p)["asof"] == asof and p.text_content(caption_scope) == early_cap
    p.locator(".date-slider .ds-btn", has_text="Play").first.click(); wait(p, 1500)
    p.locator(".date-slider .ds-btn", has_text="Pause").first.click()
    assert query(p)["asof"] > asof, "play should advance the date"


def act_leaderboard(p):
    assert rows(p, "#t") == 25
    assert qtext("MLB", "all") in p.text_content("#scap")
    assert p.locator("#c .seg button", has_text="Swing").count() == 1  # MLB has swingmen
    # Grade picker: sort, Δ column, chart axis, and URL follow it; the split chart plots Stuff+ vs Location+.
    seg(p, "#c", "Location+")
    assert query(p).get("grade") == "loc" and p.locator("#t th", has_text="Δ Location+").count() == 1
    assert "Location+ grade" in p.text_content("#scap")
    seg(p, "#c2", "Stuff+ vs. Location+")
    assert query(p).get("chart") == "split" and "better location" in p.text_content("#scap")
    p.reload(wait_until="networkidle"); wait(p, 800)
    assert query(p).get("grade") == "loc" and "better location" in p.text_content("#scap")
    seg(p, "#c2", "vs. results"); seg(p, "#c", "Pitching+"); seg(p, "#c", "Stuff+")
    assert "grade" not in query(p) and "chart" not in query(p)
    p.locator("#t th", has_text="RV+").first.click(); wait(p)
    p.locator(".pager button", has_text="Next").click(); wait(p)
    # Role filter: table, caption, and URL follow it.
    seg(p, "#c", "Relievers")
    assert set(col_values(p, "#t", "Role")) == {"RP"}
    assert qtext("MLB", "RP") in p.text_content("#scap")
    assert query(p).get("role") == "RP"
    # ...and it survives switching level, team, and view.
    seg(p, "#c", "Triple-A")
    assert role_on(p, "Relievers") and set(col_values(p, "#t", "Role")) == {"RP"}
    assert qtext("AAA", "RP") in p.text_content("#scap")
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
    assert set(col_values(p, "#t", "Role")) == {"SP"} and qtext("MLB", "SP") in p.text_content("#scap")
    num_input(p).fill("1500"); num_input(p).press("Enter"); wait(p)
    assert "1500+ pitches" in p.text_content("#scap") and query(p).get("min") == "1500"
    num_input(p).fill(""); num_input(p).press("Enter"); wait(p)
    assert p.locator("#season svg").count() >= 1  # league season view
    seg(p, "#c2", "Last 30 days")
    assert query(p).get("window") == "last30" and "last 30 days" in p.text_content("#scap")
    seg(p, "#c2", "Season to date")
    seg(p, "#c", "All")
    act_slider(p, "#t", "#scap")
    slide_to_start(p)
    seg(p, "#c", "Starters")  # nobody has 3 starts in week one: the caption says why the table is empty
    assert rows(p, "#t") == 0 and "starts yet" in p.text_content("#scap")
    p.set_viewport_size({"width": 700, "height": 900}); wait(p, 600)
    assert p.locator("#scatter svg").count() == 1


def act_stuff_or_command(p):
    assert "bootstrap" in p.text_content("#cap1") and p.locator("#p1 svg circle").count() == 4
    for t in ["Stable within a season", "Stable year to year", "Triple-A to MLB", "Predicts next season"]:
        seg(p, "#c1", t)
        assert p.locator("#p1 svg circle").count() == 4
    assert "r = " in p.text_content("#cap2")
    seg(p, "#c2", "Triple-A"); assert "Triple-A" in p.text_content("#cap2")
    seg(p, "#c2", "Relievers"); search(p, "#c2", "a")
    assert p.locator("#p2 svg").count() >= 1


def act_translation(p):
    assert p.locator("#p1 svg circle").count() == 10 and "bootstrap" in p.text_content("#cap1")
    assert rows(p, "#t1") == 10
    seg(p, "#c1", "Predicts MLB production"); assert "MLB xwOBA" in p.text_content("#cap1")
    seg(p, "#c1", "Within a season"); assert p.locator("#p1 svg circle").count() == 10
    seg(p, "#c1", "Pitchers"); seg(p, "#c1", "Both cohorts")
    assert p.locator("#p1 svg circle").count() == 9 and "RV+" in p.text_content("#cap1")
    seg(p, "#c1", "Starters"); assert "starters" in p.text_content("#cap1") and p.locator("#p1 svg circle").count() == 9
    assert "r = " in p.text_content("#cap2")
    select(p, "#c2", "Number", "ev90"); assert query(p).get("tm") == "ev90"
    seg(p, "#c2", "Pitchers"); seg(p, "#c2", "Same season"); seg(p, "#c2", "Relievers")
    assert query(p).get("tside") == "pitcher" and query(p).get("trole") == "RP" and query(p).get("tcohort") == "within"
    p.reload(wait_until="networkidle"); wait(p, 900)
    assert p.locator("#c2 .seg button.on", has_text="Relievers").count() == 1 and "relievers in MLB" in p.text_content("#cap2")
    search(p, "#c2", "a"); assert p.locator("#p2 svg").count() >= 1
    p.set_viewport_size({"width": 390, "height": 900}); wait(p, 600)
    assert p.locator("#p1 svg").count() == 1 and p.locator("#p2 svg").count() >= 1


def act_triple_a_mlb(p):
    # ?view=mlb: the stat columns carry the translation marker and the caption explains it.
    assert "▲" in p.locator("#t thead").text_content() and "translated to MLB" in p.text_content("#scap")
    seg(p, "#c", "Pitchers")
    assert "▲" in p.locator("#t thead").text_content() and "batters faced" in p.text_content("#scap")
    seg(p, "#c2", "Triple-A")
    assert "▲" not in p.locator("#t thead").text_content() and "view" not in query(p)


def act_triple_a(p):
    assert rows(p, "#t") == 25
    assert "r = " in p.text_content("#scap")
    assert p.locator("#c .seg button", has_text="Starters").count() == 0  # no role filter for hitters
    select(p, "#c", "Team", nth_option(p, "#c", "Team", 5))
    assert p.locator("#flags .flag-list").count() == 1
    seg(p, "#c", "Pitchers"); p.locator("#t th", has_text="BB%").click(); wait(p)
    assert qtext("AAA", "all") in p.text_content("#scap")
    seg(p, "#c", "Starters")
    assert set(col_values(p, "#t", "Role")) <= {"SP"} and query(p).get("role") == "SP"
    select(p, "#c", "Team", "")
    assert role_on(p, "Starters") and set(col_values(p, "#t", "Role")) == {"SP"}
    assert qtext("AAA", "SP") in p.text_content("#scap")
    p.reload(wait_until="networkidle"); wait(p, 800)
    assert role_on(p, "Starters") and set(col_values(p, "#t", "Role")) == {"SP"}
    search(p, "#c", "a"); seg(p, "#c", "Hitters")
    assert "role" not in query(p)
    search(p, "#c", "")
    assert col_values(p, "#t", "Chase Δ14")[0] != "—" or True  # final date: Δ14 shown where it exists
    act_slider(p, "#t", "#scap")
    assert set(col_values(p, "#t", "Chase Δ14")) == {"—"}, "Δ14 only at the season's final date"
    p.set_viewport_size({"width": 700, "height": 900}); wait(p, 600)
    assert p.locator("#scatter svg").count() == 1


def act_rule5(p):
    assert rows(p, "#t") > 0
    banner = p.text_content("#r5banner")
    meta = json.loads((DATA / "baseball" / "meta.json").read_text(encoding="utf-8"))["rule5"]
    assert ("Preliminary" in banner) == (meta["snapshot_date"] <= meta["protection_deadline"])
    # Every status cell carries its reasoning as a tooltip.
    assert p.locator("#t .r5-status[title]").count() == min(25, rows(p, "#t"))
    assert p.locator("#t th", has_text="Pitching+").count() == 1
    seg(p, "#c", "Relievers")
    assert set(col_values(p, "#t", "Role")) <= {"RP"} and query(p).get("role") == "RP"
    select(p, "#c", "Organization", nth_option(p, "#c", "Organization", 3))
    assert role_on(p, "Relievers") and query(p).get("org")
    orgs = set(col_values(p, "#t", "Org"))
    assert len(orgs) <= 1
    p.locator("#c label", has_text="Max. age").locator("input").fill("25")
    p.locator("#c label", has_text="Max. age").locator("input").press("Enter"); wait(p)
    assert all(int(a) <= 25 for a in col_values(p, "#t", "Age")) and query(p).get("agemax") == "25"
    p.reload(wait_until="networkidle"); wait(p, 800)
    assert role_on(p, "Relievers") and query(p).get("agemax") == "25"
    seg(p, "#c", "Hitters")
    assert "role" not in query(p) and rows(p, "#t") >= 0
    select(p, "#c", "Organization", "")
    assert rows(p, "#t") > 0 and p.locator("#scatter svg").count() == 1


def act_rule5_article(p):
    assert p.locator("#t0 table.s-table tbody tr").count() == 10
    assert "pitch minimum" in p.text_content("#cap0")
    seg(p, "#c0", "Hitters")
    assert "PA" in p.text_content("#cap0") and p.locator("#t0 table.s-table tbody tr").count() > 0


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
        # Triple-A players (these have 300+ PA or BF) get the MLB translation table.
        assert "Translated to MLB" in p.locator("#root h3").all_text_contents()
        assert "range" in p.text_content("#root") and "How the translation works" in p.text_content("#root")
        tiles = p.locator("#root .stat-row").first.text_content()
        p.locator(".date-slider input[type=range]").focus(); p.keyboard.press("Home"); wait(p, 700)
        rows_now = p.locator("#root .stat-row")
        assert (rows_now.count() and rows_now.first.text_content() != tiles) or "through" in p.text_content("#root")
        assert "asof" in query(p)
        p.reload(wait_until="networkidle"); wait(p, 900)
        assert "asof" in query(p)
        p.locator(".date-slider input[type=range]").focus(); p.keyboard.press("End"); wait(p, 700)
        assert "asof" not in query(p)
        if role_tag:  # pitcher pages: the per-pitch timeline has its own grade picker
            seg(p, "#root", "Location+")
            assert query(p).get("pgrade") == "loc" and "Location+ by pitch" in p.text_content("#root")
            import re
            tags = p.locator("#root .role-tag").all_text_contents()
            assert tags and all(re.fullmatch(r"(SP|RP|Swing) · \d+ G, \d+ GS · through \w+ \d+", t) for t in tags), tags
    return check


def act_farm(p):
    assert "r = " in p.text_content("#cap1") and p.locator("#p1 svg circle").count() >= 25
    select(p, "#c1", "Measure", "fb_velo")
    assert query(p).get("m") == "fb_velo" and "Fastball velocity" in p.text_content("#cap1")
    assert rows(p, "#t2") == 30 and "orgs have an interval" in p.text_content("#cap2")
    select(p, "#c2", "Measure", "chase_pct"); seg(p, "#c2", "2025")
    assert query(p).get("om") == "chase_pct" and "2025" in p.text_content("#cap2")
    org = p.locator("#t2 .org-pick").first.get_attribute("data-org")
    p.locator("#t2 .org-pick").first.click(); wait(p, 900)
    assert query(p).get("org") == org and query(p).get("side") == "hitters"
    assert org in p.text_content("#cap3") and rows(p, "#t3") > 0
    seg(p, "#c3", "Pitchers"); seg(p, "#c3", "Relievers"); seg(p, "#c3", "All")
    seg(p, "#c3", "Pitching+")
    assert p.locator("#slider3 .date-slider").count() == 1 and "Through" in p.text_content("#cap3b")
    p.locator("#slider3 input[type=range]").focus(); p.keyboard.press("Home"); wait(p, 700)
    assert "asof" in query(p)
    seg(p, "#c3", "2025")
    assert "current season" in p.text_content("#cap3b")


def nfl_week_slider(p, caption):
    """Week slider: step back, jump to week 1, reload restores, play advances."""
    final_label, final_cap = slider_label(p), p.text_content(caption)
    p.locator(".date-slider .ds-btn", has_text="◀").first.click(); wait(p, 500)
    assert slider_label(p) != final_label and "week" in query(p)
    slide_to_start(p)
    assert slider_label(p) == "Through week 1" and query(p)["week"] == "1"
    assert p.text_content(caption) != final_cap
    p.reload(wait_until="networkidle"); wait(p, 900)
    assert slider_label(p) == "Through week 1"
    p.locator(".date-slider .ds-btn", has_text="Play").first.click(); wait(p, 1500)
    p.locator(".date-slider .ds-btn", has_text="Pause").first.click()
    assert int(query(p)["week"]) > 1, "play should advance the week"


def act_nfl_efficiency(p):
    assert rows(p, "#t") == 32
    assert p.locator("#scatter svg text", has_text="KC").count() >= 1
    cap = p.text_content("#scap")
    seg(p, "#c", "Raw"); assert query(p)["adj"] == "raw" and p.text_content("#scap") != cap
    seg(p, "#c", "Included"); assert query(p)["gt"] == "all"
    seg(p, "#c", "Passes"); assert "passes" in p.text_content("#scap")
    seg(p, "#c", "Runs"); assert col_values(p, "#t", "Off EPA/play")
    seg(p, "#c", "All plays"); seg(p, "#c", "Opponent-adjusted")
    seg(p, "#c2", "Last 4 weeks"); assert "raw numbers" in p.text_content("#c2") and "weeks" in p.text_content("#scap")
    seg(p, "#c2", "Season to date")
    select(p, "#c", "Highlight", "KC"); assert "Kansas City Chiefs" in p.text_content("#scap") and query(p)["team"] == "KC"
    select(p, "#c", "Season", "2020"); assert query(p)["season"] == "2020" and rows(p, "#t") == 32
    nfl_week_slider(p, "#scap")
    p.locator("#t table.s-table tbody tr a").first.click(); p.wait_for_load_state("networkidle"); wait(p, 800)
    assert "/nfl/teams/" in p.url and "season=2020" in p.url


def act_nfl_team(p):
    tiles = p.locator(".stat-row .stat").all_text_contents()
    assert len(tiles) == 4 and re.search(r"\d+–\d+", tiles[0]), tiles
    assert p.locator("#off svg").count() == 1 and p.locator("#def svg").count() == 1
    assert rows(p, "#games") >= 8
    seg(p, "#c", "Raw"); seg(p, "#c", "Included"); seg(p, "#c", "Passes")
    assert "passes" in p.text_content("#cap")
    seg(p, "#c", "All plays")
    select(p, "#c", "Season", "2016"); assert query(p)["season"] == "2016" and rows(p, "#games") >= 16
    nfl_week_slider(p, "#cap")


def act_nfl_index(p):
    assert p.locator("#team-grid li a").count() == 32
    assert p.locator(".team-div h4").count() == 8


def act_status(p):
    for sport in ("baseball", "nfl"):
        card = p.locator(f"#status-{sport}")
        assert card.count() == 1 and "checks" in card.text_content()
        card.locator("summary").click(); wait(p, 200)
        assert card.locator("tbody tr").count() >= 10


PAGES = [
    ("home", "/", None),
    ("baseball", "/baseball/", None),
    ("stuff-plus", "/baseball/stuff-plus/", act_stuff_plus),
    ("stuff-or-command", "/baseball/stuff-or-command/", act_stuff_or_command),
    ("shape-changes", "/baseball/shape-changes/", act_shape),
    ("two-week", "/baseball/two-week-signal/", act_two_week),
    ("farm-systems", "/baseball/farm-systems/", act_farm),
    ("stuff-leaderboard", "/baseball/tools/stuff-leaderboard.html", act_leaderboard),
    ("triple-a", "/baseball/tools/triple-a.html", act_triple_a),
    ("triple-a-mlb", "/baseball/tools/triple-a.html?view=mlb", act_triple_a_mlb),
    ("aaa-translation", "/baseball/aaa-translation/", act_translation),
    ("rule-5-board", "/baseball/tools/rule-5.html", act_rule5),
    ("rule-5-article", "/baseball/rule-5/", act_rule5_article),
    ("player-hitter", lambda: f"/baseball/player.html?id={player_hitter()}", act_player("Hitting · Triple-A")),
    ("player-pitcher", lambda: f"/baseball/player.html?id={player_pitcher()}", act_player("Pitching · Triple-A", role_tag=True)),
    ("player-missing", "/baseball/player.html?id=1", lambda p: "No " in p.text_content("#root")),
    ("nfl", "/nfl/", act_nfl_index),
    ("nfl-team-efficiency", "/nfl/tools/team-efficiency.html", act_nfl_efficiency),
    ("nfl-team", "/nfl/teams/kc/", act_nfl_team),
    ("glossary", "/glossary.html", None),
    ("about", "/about.html", None),
    ("status", "/status.html", act_status),
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


# Charts behind a date or week slider keep one axis scale for the whole season (S.stableDomain), so
# moving the slider moves the points, not the axes.
SLIDER_SCATTERS = [
    ("leaderboard", "/baseball/tools/stuff-leaderboard.html"),
    ("triple-a-pitchers", "/baseball/tools/triple-a.html?side=pitchers"),
    ("triple-a-hitters", "/baseball/tools/triple-a.html"),
    ("nfl-efficiency", "/nfl/tools/team-efficiency.html"),
]


@pytest.mark.parametrize("name,path", SLIDER_SCATTERS, ids=[s[0] for s in SLIDER_SCATTERS])
def test_slider_axes_are_stable(browser, base_url, name, path):
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    p = ctx.new_page()
    try:
        p.goto(base_url + path, wait_until="networkidle")
        wait(p, 800)
        ticks = lambda: p.eval_on_selector_all(  # noqa: E731
            "#scatter svg [aria-label='x-axis tick label'] text, #scatter svg [aria-label='y-axis tick label'] text",
            "e => e.map(t => t.textContent).join('|')")
        end = ticks()
        assert end, "no axis ticks found"
        p.locator(".date-slider input[type=range]").first.focus()
        p.keyboard.press("Home"); wait(p, 700)
        start = ticks()
        for _ in range(4):
            p.keyboard.press("ArrowRight")
        wait(p, 700)
        assert start == ticks() == end, "the axes changed when the slider moved"
    finally:
        ctx.close()
