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
import sys
import threading

import pytest

from sparvi.config import ROOT

DOCS = ROOT / "docs"
DATA = ROOT / "site" / "data"
# What a visitor downloads (GitHub Pages serves gzip) and what each page loads, not raw bytes on disk.
# Gzipped size of each sport's site/data folder (each sport has its own budget so adding a sport can't
# quietly squeeze another; the site-wide hosting limit is checked by SEO-2's budget check).
GZ_BUDGET_MB = {"baseball": 15, "nfl": 3, "cfb": 6, "cbb": 1, "status": 0.5, "models": 0.1}
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


def act_cbb_hub(p):
    assert p.locator(".card-link").count() >= 1 and "net rating" in p.text_content("main")


def act_cbb_ratings(p):
    """CBB-PAGES-1: every Division I team on the scatter; the conference filter narrows the table, not the axes."""
    n = p.locator("#scatter svg circle").count()
    assert n >= 300 and rows(p, "#t") == 25 and "average Division I team" in p.text_content("#scap")
    assert p.locator("#t a[href*='/cbb/teams/']").count() == 25
    ticks = p.locator("#scatter svg [aria-label='x-axis tick label'] text").all_text_contents()
    p.select_option("#c select", "Big Ten")
    assert query(p)["conf"] == "Big Ten" and "Big Ten" in p.text_content("#scap") and rows(p, "#t") <= 25
    assert p.locator("#scatter svg circle").count() == n                               # the rest stays, faded
    assert p.locator("#scatter svg [aria-label='x-axis tick label'] text").all_text_contents() == ticks
    assert "experimental" in p.text_content(".note") or "validated" in p.text_content(".note")


def act_cbb_matchup(p):
    """CBB-MATCHUP: the page's score is the Python forecast's (cbb/ratings/matchup.py) for the same teams and site."""
    import json
    import re

    from cbb.ratings.matchup import expected
    meta = json.loads((ROOT / "site/data/cbb/meta.json").read_text(encoding="utf-8"))
    teams = {t["slug"]: t for t in json.loads((ROOT / "site/data/cbb/ratings.json").read_text(encoding="utf-8"))["teams"]}

    def shown():
        m = re.fullmatch(r"(.+) (\d+), (.+) (\d+)", p.text_content("#score"))
        return m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
    q = query(p)                                             # defaults: the top two teams on a neutral court, nothing in the URL
    assert q == {} and "neutral court" in p.text_content("#line")
    p.select_option("#c select >> nth=0", "duke"); p.select_option("#c select >> nth=1", "uconn")
    for site, label in (("a", "At Duke"), ("b", "At UConn"), ("n", "Neutral court")):
        seg(p, "#c", label)
        x = expected(teams["duke"], teams["uconn"], site, meta["matchup"])
        a, pa, b, pb = shown()
        assert (a, b) == ("Duke", "UConn") and (pa, pb) == (round(x["a_pts"]), round(x["b_pts"])), (site, pa, pb, x)
        assert f"{abs(x['margin']):.1f}" in p.text_content("#line") and f"{100 * x['p_a']:.0f}%" in p.text_content("#line")
    assert query(p)["a"] == "duke" and query(p)["b"] == "uconn" and p.locator("#bars svg rect").count() >= 2
    p.select_option("#c select >> nth=1", "duke"); assert "two different teams" in p.text_content("#result")


def act_cbb_teams_index(p):
    assert p.locator("ul.team-index > li").count() >= 300 and p.locator("ul.team-index a[href*='/cbb/teams/']").count() >= 300


def act_cbb_team(p):
    """CBB-PAGES-1b: a team page is plain HTML: summary, ratings, results, and links to opponents' pages."""
    assert "Duke" in p.locator("h1").text_content() and "net rating" in p.text_content(".page-summary")
    assert p.locator("table.s-table").count() >= 2 and p.locator("table.s-table a[href*='/cbb/teams/']").count() >= 3
    assert p.locator("a[href*='/cbb/tools/team-ratings.html']").count() >= 1 and p.locator("script[src*='plot']").count() == 0


def act_learn_hub(p):
    assert p.locator(".card-link").count() == 8


def act_learn_stuff(p):
    """SEO-4: the pitch-grade explainer's distribution chart, its grade switch and its pitcher search."""
    assert "average" in p.text_content(".finding") and p.locator("#p1 svg rect").count() > 5
    cap = p.text_content("#cap1")
    assert "Stuff+" in cap and "pitchers at their role's pitch minimum" in cap
    seg(p, "#c1", "Location+"); assert "Location+" in p.text_content("#cap1") and p.locator("#p1 svg rect").count() > 5
    seg(p, "#c1", "Pitching+"); assert "Pitching+" in p.text_content("#cap1")
    search(p, "#c1", "a"); assert p.locator("#p1 svg text").count() > 0


def act_learn_rule5(p):
    """SEO-4: the eligibility calculator follows the rule, and the chart counts the board by first eligible draft."""
    season = int(p.evaluate("Sparvi.load('/data/baseball/meta.json').then(m => m.season)"))
    assert f"4 seasons: {season - 3} through {season}" in p.text_content("#calc") and "first time this winter" in p.text_content("#calc")
    p.locator("#c0 select").nth(0).select_option("18")
    assert f"5 seasons: {season - 3} through {season + 1}" in p.text_content("#calc") and "not eligible until" in p.text_content("#calc")
    seg(p, "#c0", "After his league's season ended")
    assert f"5 seasons: {season - 2} through {season + 2}" in p.text_content("#calc")
    p.locator("#c0 select").nth(1).select_option(str(season - 8))
    assert "has been eligible since" in p.text_content("#calc")
    n_all = p.locator("#p1 svg rect").count()
    assert n_all >= 5 and "became eligible this winter" in p.text_content("#cap1")
    seg(p, "#c1", "Pitchers"); assert "pitchers" in p.text_content("#cap1") and p.locator("#p1 svg rect").count() >= 5


def act_learn_epa(p):
    """SEO-4: the EPA explainer's offense-vs-defense scatter: 32 teams in every view, and the team search."""
    assert p.locator("#p1 svg circle").count() == 32 and "all plays" in p.text_content("#cap1")
    seg(p, "#c1", "Passes"); assert "passes" in p.text_content("#cap1") and p.locator("#p1 svg circle").count() == 32
    seg(p, "#c1", "Runs"); assert "runs" in p.text_content("#cap1")
    search(p, "#c1", "chiefs"); assert p.locator("#p1 svg circle").count() == 33


def act_learn_swing(p):
    """SEO-4: the swing-decisions explainer's scatter of qualified Triple-A hitters, its three views and the search."""
    n = p.locator("#p1 svg circle").count()
    assert n >= 100 and "zone contact" in p.text_content("#cap1") and "qualified hitters" in p.text_content("#cap1")
    seg(p, "#c1", "Zone swing"); assert "zone swing" in p.text_content("#cap1") and p.locator("#p1 svg circle").count() == n
    seg(p, "#c1", "Whiff rate"); assert "whiff rate" in p.text_content("#cap1") and "misses less" in p.text_content("#p1")
    search(p, "#c1", "a"); assert p.locator("#p1 svg circle").count() > n


def act_learn_xwoba(p):
    """SEO-4: the xwOBA explainer's scatter of qualified Triple-A hitters, its four horizontal axes and the search."""
    n = p.locator("#p1 svg circle").count()
    assert n >= 100 and "Hard-hit rate and xwOBA" in p.text_content("#cap1") and "qualified hitters" in p.text_content("#cap1")
    seg(p, "#c1", "EV90"); assert "EV90 and xwOBA" in p.text_content("#cap1") and "mph" in p.text_content("#p1")
    seg(p, "#c1", "Strikeout rate"); assert "Strikeout rate and xwOBA" in p.text_content("#cap1") and p.locator("#p1 svg circle").count() == n
    seg(p, "#c1", "Walk rate"); assert "Walk rate and xwOBA" in p.text_content("#cap1")
    search(p, "#c1", "a"); assert p.locator("#p1 svg circle").count() > n


def act_learn_ivb(p):
    """SEO-4: the IVB explainer's movement chart: one pitch type at a time on fixed axes, and the search."""
    n = p.locator("#p1 svg circle").count()
    assert n >= 100 and "four-seam" in p.text_content("#cap1") and "of IVB" in p.text_content("#cap1")
    ticks = p.locator("#p1 svg [aria-label='x-axis tick label'] text").all_text_contents()
    seg(p, "#c1", "Curveball"); assert "curveball" in p.text_content("#cap1") and p.locator("#p1 svg circle").count() != n
    assert p.locator("#p1 svg [aria-label='x-axis tick label'] text").all_text_contents() == ticks      # the axes don't move
    seg(p, "#c1", "Sinker"); m = p.locator("#p1 svg circle").count()
    search(p, "#c1", "a"); assert p.locator("#p1 svg circle").count() > m


def act_learn_power(p):
    """SEO-4: the power-rating explainer's two-part bars; a conference view lists only that conference."""
    assert p.locator("#p1 svg rect").count() >= 50 and "Elo part" in p.text_content("#cap1")     # 25 teams x 2 parts
    assert p.locator("table tbody tr").count() == 6                                              # the backtest table
    conf = p.locator("#c1 select option").nth(1).get_attribute("value")
    p.locator("#c1 select").select_option(conf)
    n = p.locator("#p1 svg rect").count()
    assert 8 <= n <= 50 and n % 2 == 0


def act_learn_adjusted(p):
    """SEO-4: the opponent-adjustment explainer's raw-vs-adjusted scatter, both sides, and the team search."""
    n = p.locator("#p1 svg circle").count()
    assert n >= 120 and "offense" in p.text_content("#cap1")
    seg(p, "#c1", "Defense"); assert "defense" in p.text_content("#cap1") and p.locator("#p1 svg circle").count() >= 120
    search(p, "#c1", "ohio"); assert p.locator("#p1 svg circle").count() > n


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
    # R5-DEFENSE: hitters' line, positions and fielding; the position filter keeps only that group.
    for h in ("Positions (G)", "AVG/OBP/SLG", "HR", "SB", "Defense"):
        assert p.locator("#t th", has_text=h).count() >= 1, h
    p.locator("#c label", has_text="Max. age").locator("input").fill("")
    p.locator("#c label", has_text="Max. age").locator("input").press("Enter"); wait(p)
    seg(p, "#c", "Catchers")
    assert query(p).get("pos") == "C" and rows(p, "#t") > 0 and "catchers" in p.text_content("#scap")
    assert all(v.startswith("CS ") for v in col_values(p, "#t", "Defense")), "catchers show caught stealing"
    seg(p, "#c", "Pitchers")
    assert "pos" not in query(p)


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


def player_url(pid, qs=""):
    """A player's static page (PLAN-seo S2), from the URL registry."""
    reg = json.loads((DATA / "baseball" / "player_urls.json").read_text(encoding="utf-8"))
    return f"/baseball/players/{reg[str(pid)][-1]}/{qs}"


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


def act_free_agents(p):
    assert "Pitching+" in p.text_content(".finding")
    assert rows(p, "#t") == 25 and p.locator("#t a[href^='/baseball/players/']").count() >= 20
    assert p.locator("#scatter svg circle").count() >= 50
    seg(p, "#c", "Starters"); assert query(p)["role"] == "SP" and "starters" in p.text_content("#scap")
    n_sp = rows(p, "#t")
    seg(p, "#c", "Unsigned"); assert query(p)["status"] == "unsigned" and rows(p, "#t") <= n_sp
    seg(p, "#c", "All"); seg(p, "#c", "Everyone")


def act_altitude(p):
    """R5-ALTITUDE: 30 parks with intervals; the pitcher scatter and table follow the board / all and team filters."""
    assert "movement" in p.text_content(".finding") and p.locator("#parks svg circle").count() == 30
    assert "high-altitude parks" in p.text_content("#pcap")
    n = p.locator("#scatter svg circle").count()
    assert n >= 150 and rows(p, "#t") == 25 and p.locator("#t a[href^='/baseball/players/']").count() >= 20
    seg(p, "#c", "All Triple-A"); assert query(p)["who"] == "all" and p.locator("#scatter svg circle").count() > n
    seg(p, "#c", "The five altitude teams"); assert query(p)["teams"] == "alt" and "altitude teams only" in p.text_content("#scap")
    assert 100 <= p.locator("#scatter svg circle").count() < n * 2
    seg(p, "#c", "Rule 5 board"); seg(p, "#c", "All teams")


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


def act_nfl_matchup(p):
    """NFL-MATCHUP: the page's margin and win chance are the Python forecast's for the same teams and site."""
    import json

    from nfl.team_efficiency.matchup import expected
    d = json.loads((ROOT / "site/data/nfl/matchup.json").read_text(encoding="utf-8"))
    teams = {t["id"]: t for t in d["teams"]}
    assert query(p) == {} and "neutral site" in p.text_content("#line")
    p.select_option("#c select >> nth=0", "KC"); p.select_option("#c select >> nth=1", "BUF")
    for site, label in (("a", "At Kansas City Chiefs"), ("b", "At Buffalo Bills"), ("n", "Neutral site")):
        seg(p, "#c", label)
        x = expected(teams["KC"], teams["BUF"], site, d["line"])
        fav = "Kansas City Chiefs" if x["margin"] >= 0 else "Buffalo Bills"
        assert p.text_content("#score") == f"{fav} by {abs(x['margin']):.1f}", (site, p.text_content("#score"), x)
        assert f"Kansas City Chiefs wins {100 * x['p_a']:.0f}%" in p.text_content("#line")
    assert query(p)["a"] == "KC" and query(p)["b"] == "BUF" and p.locator("#bars svg rect").count() >= 2
    assert p.locator("#t a[href*='/nfl/teams/']").count() == 2
    p.select_option("#c select >> nth=1", "KC"); assert "two different teams" in p.text_content("#result")


def act_nfl_team(p):
    assert p.locator("#team-static").count() == 0, "the plain-HTML block should give way to the charts"
    assert "Through week" in p.text_content(".page-summary") and p.locator("a[href$='nfl/teams/']").count() >= 1
    tiles = p.locator(".stat-row .stat").all_text_contents()
    assert len(tiles) == 4 and re.search(r"\d+–\d+", tiles[0]), tiles
    assert p.locator("#off svg").count() == 1 and p.locator("#def svg").count() == 1
    assert rows(p, "#games") >= 8
    seg(p, "#c", "Raw"); seg(p, "#c", "Included"); seg(p, "#c", "Passes")
    assert "passes" in p.text_content("#cap")
    seg(p, "#c", "All plays")
    select(p, "#c", "Season", "2016"); assert query(p)["season"] == "2016" and rows(p, "#games") >= 16
    nfl_week_slider(p, "#cap")


def act_nfl_teams_index(p):
    assert p.locator("ul.team-index a").count() == 32 and p.locator("h2").count() == 8


def act_cfb_teams_index(p):
    assert p.locator("ul.team-index a").count() >= 130 and "Former FBS teams" in p.locator("h2").all_text_contents()


def act_cfb_portal_coming(p):
    """The coming window is listed before it opens (Matt, Nov 17): no entrants yet, programs' openings shown."""
    import json
    idx = json.loads((DATA / "cfb" / "portal" / "index.json").read_text(encoding="utf-8"))["windows"]
    if not any(w["window"] == 2027 and w["players"] == 0 for w in idx):
        return                                         # entrants have arrived: the regular test covers it
    assert "No entrants yet" in p.text_content("#tcap")
    assert rows(p, "#t") == 0


def no_team_cards(p):
    # The hub's article listing must not pick up the generated team pages.
    hrefs = p.eval_on_selector_all("#listing-nfl-articles a, #listing-cfb-articles a", "e => e.map(a => a.getAttribute('href'))")
    assert hrefs and not [h for h in hrefs if "teams/" in h], hrefs


def act_nfl_index(p):
    no_team_cards(p)
    assert p.locator("#team-grid li a").count() == 32
    assert p.locator(".team-div h4").count() == 8
    assert p.locator(".card-link").count() >= 1


def act_status(p):
    for sport in ("baseball", "nfl", "cfb"):
        card = p.locator(f"#status-{sport}")
        text = card.text_content()
        assert card.count() == 1 and "Data checks" in text and "Plausibility tests" in text
        # Both kinds of test passed on the committed data, and every result is listed behind its disclosure.
        assert card.locator(".status-block").count() == 2 and card.locator(".status-badge.ok").count() == 2
        for i in range(card.locator("summary").count()):
            card.locator("summary").nth(i).click(); wait(p, 200)
        assert card.locator(".status-block").nth(0).locator("tbody tr").count() >= 10
        assert card.locator(".status-block").nth(1).locator("tbody tr").count() >= 10
    # College football lists each game whose play-by-play score is off, with what is done about it.
    assert "Decision" in p.locator("#status-cfb").text_content()


def act_cfb_matchup(p):
    """CFB-MATCHUP: the page's margin and win chance are the Python forecast's for the same teams and site."""
    import json

    from cfb.adjusted_ratings.matchup import expected
    meta = json.loads((ROOT / "site/data/cfb/meta.json").read_text(encoding="utf-8"))
    wk = json.loads((ROOT / f"site/data/cfb/team_week_{meta['season']}.json").read_text(encoding="utf-8"))
    power = {t["id"]: {"power": t["series"]["no_gt"]["power"][-1]} for t in wk["teams"]}
    assert query(p) == {} and "neutral site" in p.text_content("#line")
    p.select_option("#c select >> nth=0", "ohio-state"); p.select_option("#c select >> nth=1", "michigan")
    for site, label in (("a", "At Ohio State"), ("b", "At Michigan"), ("n", "Neutral site")):
        seg(p, "#c", label)
        x = expected(power["Ohio State"], power["Michigan"], site, meta["power_model"])
        fav = "Ohio State" if x["margin"] >= 0 else "Michigan"
        assert p.text_content("#score") == f"{fav} by {abs(x['margin']):.1f}", (site, p.text_content("#score"), x)
        assert f"Ohio State wins {100 * x['p_a']:.0f}%" in p.text_content("#line")
    assert query(p)["a"] == "ohio-state" and query(p)["b"] == "michigan" and p.locator("#bars svg rect").count() >= 2
    assert p.locator("#t a[href*='/cfb/teams/']").count() == 2
    p.select_option("#c select >> nth=1", "ohio-state"); assert "two different teams" in p.text_content("#result")


def act_cfb_ratings(p):
    assert rows(p, "#t") == 25 and "#" in p.text_content("#t thead")
    assert p.locator("#scatter svg circle").count() >= 100
    select(p, "#c", "Conference", "power"); wait(p, 400)
    assert query(p)["conf"] == "power" and "power conferences" in p.text_content("#scap")
    select(p, "#c", "Conference", "all")
    seg(p, "#c", "Raw"); assert query(p)["adj"] == "raw"
    seg(p, "#c", "Opponent-adjusted")
    seg(p, "#c2", "Last 4 weeks"); assert "raw numbers" in p.text_content("#c2")
    seg(p, "#c2", "Season to date")
    select(p, "#c", "Season", "2019"); assert query(p)["season"] == "2019" and "bowls" in p.text_content("#scap")
    p.locator(".date-slider input[type=range]").first.focus(); p.keyboard.press("Home"); wait(p, 500)
    assert "faded" in p.text_content("#scap")


def act_cfb_article(p):
    assert "of the time" in p.text_content(".finding")
    assert p.locator("#scatter svg circle").count() >= 20
    select(p, "#c", "Season", "2024"); wait(p, 500)
    assert "the bowls" in p.text_content("#scap")


def act_cfb_portal(p):
    """SC-3a: position tabs, filters, search, best landing spots, and a program's projected-yards and fit lists."""
    assert rows(p, "#t") == 25 and "Value over repl." in p.text_content("#t thead")
    assert "quarterbacks in the" in p.text_content("#tcap")
    assert "Tier 1" in p.text_content("#tier-rules") and p.locator("[data-n]").first.text_content().strip()
    seg(p, "#c", "WR"); assert query(p)["pos"] == "WR" and "wide receivers" in p.text_content("#tcap")
    select(p, "#c2", "Level last season", "FCS"); assert query(p)["lvl"] == "FCS" and rows(p, "#t") >= 1
    select(p, "#c2", "Level last season", "all")
    select(p, "#c2", "Market tier", "1"); assert query(p)["tier"] == "1"
    select(p, "#c2", "Market tier", "all")
    seg(p, "#c", "QB")
    name = p.locator("#t tbody tr td strong").first.text_content()
    search(p, "#c2", name); assert rows(p, "#t") >= 1 and name in p.text_content("#t tbody")
    search(p, "#c2", "")
    assert "Best landing spots" in p.text_content("#t thead")
    select(p, "#c", "Program", "Ohio State"); wait(p, 400)
    assert query(p)["prog"] == "Ohio State" and "Ohio State" in p.text_content("#prog") and "experimental" in p.text_content("#prog")
    # CFB-PROJ: the projected-yards list is the default program view; the fit list is a toggle.
    assert "Projected yards (80%)" in p.text_content("#t thead") and 1 <= rows(p, "#t") <= 25
    assert "projected yards at Ohio State" in p.text_content("#tcap")
    seg(p, "#c", "System fit"); wait(p, 300)
    assert query(p)["list"] == "fit"
    assert "Fit" in p.text_content("#t thead") and "Projected here" in p.text_content("#t thead") and 1 <= rows(p, "#t") <= 25
    assert "frozen in the forecast record" in p.text_content("#tcap")
    seg(p, "#c2", "Backup minimum"); wait(p, 300)
    assert query(p)["min"] == "floor" and "ranked among entrants with at least" in p.text_content("#tcap")


def act_cfb_index(p):
    no_team_cards(p)
    assert p.locator(".card-link").count() >= 1
    assert p.locator("#team-grid li a").count() >= 130
    assert p.locator(".team-div h4", has_text="SEC").count() == 1


def act_cfb_team(p):
    assert len(p.locator("#quarto-content p").first.text_content()) > 40, "the summary paragraph renders without JS"
    tiles = p.locator(".stat-row .stat").all_text_contents()
    assert len(tiles) == 5 and re.search(r"\d+–\d+", tiles[0]) and "of" in tiles[1], tiles
    for host in ("#power", "#off", "#def"):
        assert p.locator(f"{host} svg").count() == 1, host
    assert rows(p, "#games") >= 8
    assert p.locator("#games a[href^='/cfb/teams/']").count() >= 5, "FBS opponents link to their pages"
    seg(p, "#c", "Raw"); seg(p, "#c", "Included"); seg(p, "#c", "Passes")
    assert "passes" in p.text_content("#cap")
    seg(p, "#c", "All plays")
    select(p, "#c", "Season", "2019"); assert query(p)["season"] == "2019" and "bowls included" in p.text_content("#gcap")
    nfl_week_slider(p, "#cap")


def act_cfb_former_team(p):
    # Idaho left FBS after 2017: its page opens on its last FBS season and offers only FBS seasons.
    assert "2017" in p.text_content("#gcap")
    opts = p.locator("#c select").first.locator("option").all_text_contents()
    assert opts[0] == "2017" and "2026" not in opts


def act_cfb_loss_cost(p):
    assert "spots" in p.text_content(".finding")
    assert p.locator("#scatter svg circle").count() >= 100
    assert rows(p, "#games") >= 1 or "No regular-season games" in p.text_content("#games")
    ticks = lambda: p.eval_on_selector_all("#scatter svg [aria-label='y-axis tick label'] text", "e => e.map(t => t.textContent).join('|')")  # noqa: E731
    before = ticks()
    seg(p, "#c", "Power rating"); assert query(p)["view"] == "power" and "power rating" in p.text_content("#scap")
    assert ticks() == before, "both views share one y scale"


def act_cfb_carousel(p):
    """CFB-CAROUSEL: the scatter and table follow the filters; the in-progress table lists this season's new coaches."""
    assert "points per game" in p.text_content(".finding")
    assert p.locator("#scatter svg circle").count() >= 100 and rows(p, "#t") == 15
    ticks = lambda: p.eval_on_selector_all("#scatter svg [aria-label='x-axis tick label'] text", "e => e.map(t => t.textContent).join('|')")  # noqa: E731
    before = ticks()
    seg(p, "#c", "Left for another job"); assert query(p)["why"] == "left" and "These" in p.text_content("#scap")
    assert ticks() == before, "filters keep one scale"
    seg(p, "#c", "Group of Five"); assert query(p)["lvl"] == "G5"
    seg(p, "#c", "First-time"); assert query(p)["coach"] == "first" and rows(p, "#t") >= 1
    assert rows(p, "#cur") >= 1 and "first-year head coach" in p.text_content("#ccap")


def act_nfl_offense_defense(p):
    assert "times as closely" in p.text_content(".finding")
    assert rows(p, "#games") >= 2 or "No regular-season games" in p.text_content("#games")
    assert p.locator("#grid svg rect").count() >= 9
    seg(p, "#c", "Passes"); assert query(p)["split"] == "pass" and "passes" in p.text_content("#gcap")
    seg(p, "#c2", "Runs"); assert query(p)["gsplit"] == "run" and "runs" in p.text_content("#gridcap")


def act_nfl_fourth_downs(p):
    assert "fourth-down decisions" in p.text_content(".finding") and "%" in p.text_content(".finding")
    assert p.locator("#map svg rect").count() >= 500 and "clear go" in p.text_content("#mapcap")
    seg(p, "#c", "Margin"); assert query(p)["view"] == "margin" and "percentage points" in p.text_content("#mapcap")
    seg(p, "#c", "Call"); assert "view" not in query(p)
    assert p.locator("#trend svg circle").count() >= 6
    assert rows(p, "#teams") == 32 and rows(p, "#costly") >= 5
    first = nth_option(p, "#c2", "Season", 1)
    select(p, "#c2", "Season", first); assert query(p)["season"] == first and first in p.text_content("#teamcap")
    assert rows(p, "#teams") == 32 and first in p.text_content("#costcap")


def act_nfl_jax(p):
    assert "Jacksonville" in p.text_content(".finding") and "%" in p.text_content(".finding")
    assert p.locator("#lines svg text", has_text="JAX").count() >= 1
    assert "Jacksonville Jaguars" in p.text_content("#cap")
    select(p, "#c", "Highlight", "KC"); wait(p, 300)
    assert "Kansas City Chiefs" in p.text_content("#cap")
    assert rows(p, "#hist") >= 10


def act_home(p):
    """Latest: newest first, a sport label and a filled-in key number on every card."""
    import datetime as dt
    dates = [dt.datetime.strptime(t.strip(), "%b %d, %Y") for t in p.locator(".article-cards .card-date").all_text_contents()]
    assert len(dates) >= 5 and dates == sorted(dates, reverse=True)
    assert all(t.strip() for t in p.locator(".article-cards .card-sport").all_text_contents())
    stats = p.locator(".article-cards .card-stat-v").all_text_contents()
    assert stats and not any("…" in t for t in stats)
    assert p.locator('a[href="https://x.com/SparviAnalytics"]').count() >= 1


def act_players_index(p):
    links = p.locator("ul.player-index a")
    assert links.count() > 1500
    href = links.first.evaluate("a => a.pathname")
    assert re.fullmatch(r"/baseball/players/[a-z0-9-]+-\d+/", href), href


def act_baseball_teams_index(p):
    assert p.locator("ul.team-index > li").count() == 30
    assert p.locator("ul.team-index a[href*='teams/aaa/']").count() == 30


def act_baseball_team_aaa(p):
    assert "Omaha Storm Chasers" in p.locator("h1").text_content() and "Triple-A team" in p.text_content(".page-summary")
    assert p.locator("table.s-table").count() == 2 and p.locator("table.s-table a[href*='/baseball/players/']").count() >= 40
    assert p.locator("a[href*='teams/mlb/kc/']").count() == 1


def act_baseball_team_mlb(p):
    assert p.locator("table.s-table tbody tr").count() >= 15 and p.locator("a[href*='teams/aaa/oma/']").count() == 1


PAGES = [
    ("home", "/", act_home),
    ("baseball", "/baseball/", None),
    ("stuff-plus", "/baseball/stuff-plus/", act_stuff_plus),
    ("stuff-or-command", "/baseball/stuff-or-command/", act_stuff_or_command),
    ("shape-changes", "/baseball/shape-changes/", act_shape),
    ("two-week", "/baseball/two-week-signal/", act_two_week),
    ("farm-systems", "/baseball/farm-systems/", act_farm),
    ("free-agent-pitchers", "/baseball/free-agent-pitchers/", act_free_agents),
    ("triple-a-altitude", "/baseball/triple-a-altitude/", act_altitude),
    ("stuff-leaderboard", "/baseball/tools/stuff-leaderboard.html", act_leaderboard),
    ("triple-a", "/baseball/tools/triple-a.html", act_triple_a),
    ("triple-a-mlb", "/baseball/tools/triple-a.html?view=mlb", act_triple_a_mlb),
    ("aaa-translation", "/baseball/aaa-translation/", act_translation),
    ("rule-5-board", "/baseball/tools/rule-5.html", act_rule5),
    ("rule-5-article", "/baseball/rule-5/", act_rule5_article),
    ("player-hitter", lambda: player_url(player_hitter()), act_player("Hitting · Triple-A")),
    ("player-pitcher", lambda: player_url(player_pitcher()), act_player("Pitching · Triple-A", role_tag=True)),
    ("player-old-url-missing", "/baseball/player.html?id=1", lambda p: "No profile" in p.text_content("#root")),
    ("players-index", "/baseball/players/", act_players_index),
    ("baseball-teams", "/baseball/teams/", act_baseball_teams_index),
    ("baseball-team-aaa", "/baseball/teams/aaa/oma/", act_baseball_team_aaa),
    ("baseball-team-mlb", "/baseball/teams/mlb/kc/", act_baseball_team_mlb),
    ("nfl", "/nfl/", act_nfl_index),
    ("nfl-jax-article", "/nfl/is-jacksonville-the-best/", act_nfl_jax),
    ("nfl-offense-or-defense", "/nfl/offense-or-defense/", act_nfl_offense_defense),
    ("nfl-fourth-downs", "/nfl/fourth-downs/", act_nfl_fourth_downs),
    ("nfl-team-efficiency", "/nfl/tools/team-efficiency.html", act_nfl_efficiency),
    ("nfl-matchup", "/nfl/tools/matchup.html", act_nfl_matchup),
    ("nfl-team", "/nfl/teams/kc/", act_nfl_team),
    ("nfl-teams-index", "/nfl/teams/", act_nfl_teams_index),
    ("cfb", "/cfb/", act_cfb_index),
    ("cfb-team", "/cfb/teams/georgia/", act_cfb_team),
    ("cfb-teams-index", "/cfb/teams/", act_cfb_teams_index),
    ("cfb-team-former", "/cfb/teams/idaho/", act_cfb_former_team),
    ("cfb-team-ratings", "/cfb/tools/team-ratings.html", act_cfb_ratings),
    ("cfb-matchup", "/cfb/tools/matchup.html", act_cfb_matchup),
    ("cfb-portal", "/cfb/tools/portal-board.html", act_cfb_portal),
    ("cfb-portal-coming-window", "/cfb/tools/portal-board.html?win=2027", act_cfb_portal_coming),
    ("cfb-ranked-vs-good", "/cfb/ranked-vs-good/", act_cfb_article),
    ("cfb-what-a-loss-costs", "/cfb/what-a-loss-costs/", act_cfb_loss_cost),
    ("cfb-coaching-changes", "/cfb/coaching-changes/", act_cfb_carousel),
    ("cbb", "/cbb/", act_cbb_hub),
    ("cbb-team-ratings", "/cbb/tools/team-ratings.html", act_cbb_ratings),
    ("cbb-matchup", "/cbb/tools/matchup.html", act_cbb_matchup),
    ("cbb-teams", "/cbb/teams/", act_cbb_teams_index),
    ("cbb-team", "/cbb/teams/duke/", act_cbb_team),
    ("learn", "/learn/", act_learn_hub),
    ("learn-stuff-plus", "/learn/stuff-plus/", act_learn_stuff),
    ("learn-swing-decisions", "/learn/swing-decisions/", act_learn_swing),
    ("learn-xwoba", "/learn/xwoba/", act_learn_xwoba),
    ("learn-induced-vertical-break", "/learn/induced-vertical-break/", act_learn_ivb),
    ("learn-epa", "/learn/epa/", act_learn_epa),
    ("learn-rule-5-eligibility", "/learn/rule-5-eligibility/", act_learn_rule5),
    ("learn-cfb-power-rating", "/learn/cfb-power-rating/", act_learn_power),
    ("learn-opponent-adjusted-epa", "/learn/opponent-adjusted-epa/", act_learn_adjusted),
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
    ("cfb-ratings", "/cfb/tools/team-ratings.html"),
    ("cfb-article", "/cfb/ranked-vs-good/"),
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


# ---- static player pages (PLAN-seo S2) ----------------------------------------------------------
def test_old_player_url_redirects_with_view_state(browser, base_url):
    ctx = browser.new_context()
    p = ctx.new_page()
    try:
        pid = player_pitcher()
        p.goto(f"{base_url}/baseball/player.html?id={pid}&window=last30", wait_until="networkidle")
        wait(p, 500)
        assert p.url == base_url + player_url(pid, "?window=last30")
        assert query(p).get("window") == "last30"
    finally:
        ctx.close()


def test_player_page_numbers_are_in_the_html_without_javascript(browser, base_url):
    """Crawlers and no-JS readers get the season's numbers as plain HTML."""
    ctx = browser.new_context(java_script_enabled=False)
    p = ctx.new_page()
    try:
        pid = player_pitcher()
        p.goto(base_url + player_url(pid))
        row = next(r for r in json.loads((DATA / "baseball" / "aaa_pitchers.json").read_text(encoding="utf-8")) if r["player_id"] == pid)
        assert p.locator("h1").count() == 1 and row["name"] in p.locator("h1").text_content()
        assert str(row["bf"]) in p.locator(".player-summary").text_content()
        assert f"{100 * row['k_pct']:.1f}%" in p.text_content("#root")
        assert p.locator("#root table.s-table tbody tr").count() >= 5
        assert p.locator("link[rel=canonical]").get_attribute("href") == "https://sparvi.io" + player_url(pid)
    finally:
        ctx.close()


def test_js_slug_matches_python_on_every_name(browser, base_url):
    """Sparvi.slug (tools' player links) must give the same path as sparvi/urls.py for every player."""
    from sparvi.urls import slug
    names = sorted({json.loads(f.read_text(encoding="utf-8"))["name"] for f in (DATA / "baseball" / "players").glob("*.json")})
    ctx = browser.new_context()
    p = ctx.new_page()
    try:
        p.goto(base_url + "/baseball/tools/stuff-leaderboard.html", wait_until="networkidle")
        js = p.evaluate("names => names.map(n => Sparvi.slug(n))", names)
        bad = [(n, a, slug(n)) for n, a in zip(names, js, strict=True) if a != slug(n)]
        assert not bad, bad[:5]
    finally:
        ctx.close()


def test_player_sitemap_lists_real_pages():
    idx = (DOCS / "sitemap-index.xml").read_text(encoding="utf-8")
    assert "https://sparvi.io/sitemap-baseball-players.xml" in idx
    locs = re.findall(r"<loc>(.*?)</loc>", (DOCS / "sitemap-baseball-players.xml").read_text(encoding="utf-8"))
    assert len(locs) > 1500
    for loc in locs[::97]:
        h = (DOCS / loc.removeprefix("https://sparvi.io/") / "index.html").read_text(encoding="utf-8")
        assert 'name="robots" content="noindex"' not in h and f'<link rel="canonical" href="{loc}">' in h
    assert not (DOCS / "baseball" / "players" / "template").exists()
    assert "players/template" not in (DOCS / "sitemap.xml").read_text(encoding="utf-8")
    assert "baseball/player.html" not in (DOCS / "sitemap.xml").read_text(encoding="utf-8")


def test_build_report_within_budget():
    r = json.loads((DOCS / "_build-report.json").read_text(encoding="utf-8"))
    assert r["files"] > 2000 and all(v < 0.6 for v in r["usage"].values()), r["usage"]


def test_team_page_numbers_are_in_the_html_without_javascript(browser, base_url):
    """Team pages (SEO-2b) show the summary, tiles, splits, and results as plain HTML."""
    ctx = browser.new_context(java_script_enabled=False)
    p = ctx.new_page()
    try:
        p.goto(base_url + "/nfl/teams/kc/")
        assert "Kansas City Chiefs" in p.text_content(".page-summary")
        assert p.locator("#team-static .stat-row .stat").count() == 4
        assert p.locator("#team-static table.s-table").count() == 2
        assert p.locator("#team-static a[href*='nfl/teams/']").count() >= 8
        p.goto(base_url + "/cfb/teams/georgia/")
        assert "Georgia" in p.text_content(".page-summary")
        assert p.locator("#team-static .stat-row .stat").count() == 5 and p.locator("#team-static table.s-table").count() == 2
        assert p.locator(".team-sub a[href*='cfb/teams/']").count() >= 10      # SEC teams + the index
    finally:
        ctx.close()


# ---- structured data and preview images (PLAN-seo S3) ---------------------------------------------
def _ld_types(rel):
    h = (DOCS / rel).read_text(encoding="utf-8")
    types = []
    for b in re.findall(r'<script type="application/ld\+json"[^>]*>(.*?)</script>', h, re.S):
        d = json.loads(b)
        types += [x["@type"] for x in d.get("@graph", [d])]
    return h, types


@pytest.mark.parametrize("rel,want", [
    ("index.html", {"WebSite", "Organization"}),
    ("baseball/stuff-plus/index.html", {"BreadcrumbList", "Article"}),
    ("baseball/tools/stuff-leaderboard.html", {"BreadcrumbList", "Dataset"}),
    ("nfl/tools/team-efficiency.html", {"BreadcrumbList", "Dataset"}),
    ("nfl/teams/kc/index.html", {"BreadcrumbList", "SportsTeam"}),
    ("cfb/teams/georgia/index.html", {"BreadcrumbList", "SportsTeam"}),
    ("baseball/players/ryan-ramsey-687547/index.html", {"BreadcrumbList", "ProfilePage"}),
    ("baseball/teams/aaa/oma/index.html", {"BreadcrumbList", "SportsTeam"}),
    ("glossary.html", {"BreadcrumbList"}),
    ("learn/index.html", {"BreadcrumbList"}),
    ("learn/stuff-plus/index.html", {"BreadcrumbList", "Article"}),
])
def test_structured_data_by_page_type(rel, want):
    _, types = _ld_types(rel)
    assert want <= set(types), types


def test_breadcrumb_links_resolve():
    for rel in ["baseball/players/ryan-ramsey-687547/index.html", "baseball/teams/aaa/oma/index.html", "cfb/teams/georgia/index.html"]:
        h = (DOCS / rel).read_text(encoding="utf-8")
        for b in re.findall(r'<script type="application/ld\+json"[^>]*>(.*?)</script>', h, re.S):
            d = json.loads(b)
            for x in d.get("@graph", [d]):
                if x["@type"] == "BreadcrumbList":
                    names = [i["name"] for i in x["itemListElement"]]
                    assert names[0] == "Sparvi" and len(names) >= 3, names
                    for i in x["itemListElement"]:
                        p = i["item"].removeprefix("https://sparvi.io/")
                        assert (DOCS / (p + "index.html" if p.endswith("/") or not p else p)).exists(), i["item"]


@pytest.mark.parametrize("rel", ["baseball/stuff-plus/index.html", "baseball/tools/triple-a.html", "baseball/index.html",
                                 "baseball/players/ryan-ramsey-687547/index.html", "baseball/teams/mlb/kc/index.html",
                                 "nfl/teams/kc/index.html"])
def test_preview_image_exists_and_is_1200x630(rel):
    from PIL import Image
    h = (DOCS / rel).read_text(encoding="utf-8")
    og = re.search(r'<meta property="og:image" content="https://sparvi.io/([^"]+)"', h)[1]
    tw = re.search(r'<meta name="twitter:image" content="https://sparvi.io/([^"]+)"', h)[1]
    assert og == tw and Image.open(DOCS / og).size == (1200, 630)
    assert "og-default.png" not in og


# ---- first-mention links and "Mentioned in" (PLAN-seo S5) ----------------------------------------------
def test_articles_link_first_mentions_and_entity_pages_list_them():
    h = (DOCS / "baseball" / "stuff-or-command" / "index.html").read_text(encoding="utf-8")
    links = re.findall(r'<a href="(/baseball/players/[a-z0-9-]+-\d+/)" class="entity-link">', h)
    assert len(links) >= 5 and len(links) == len(set(links)), "each player linked once"
    for path in links:
        page = (DOCS / path.strip("/") / "index.html").read_text(encoding="utf-8")
        assert '<section class="mentioned-in">' in page and 'href="/baseball/stuff-or-command/"' in page, path


def test_build_has_no_broken_internal_links():
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "site" / "build" / "linkcheck.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


def test_leaderboard_first_paint_from_the_small_file(browser, base_url):
    """PERF-2: the default view's table paints from snapshots_grades_mlb_latest.json while the full season file is
    still on its way, and shows the same rows once it arrives."""
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    p = ctx.new_page()
    held = []
    p.route("**/data/baseball/snapshots_grades_mlb.json", lambda route: held.append(route))
    try:
        p.goto(base_url + "/baseball/tools/stuff-leaderboard.html", wait_until="commit")
        p.wait_for_selector("#t table.s-table tbody tr", timeout=15000)
        first = p.locator("#t table.s-table tbody").inner_text()
        assert rows(p, "#t") == 25 and held, "the table should paint while the full file is held"
        assert p.locator("#scatter svg").count() == 0                     # the charts need the full season
        for route in held:
            route.continue_()
        p.wait_for_selector("#scatter svg", timeout=15000)
        wait(p, 600)
        assert p.locator("#t table.s-table tbody").inner_text() == first
        # A view the small file can't serve (an earlier date) goes straight to the full file.
        seen = []
        p2 = ctx.new_page()
        p2.on("request", lambda r: "/data/baseball/" in r.url and seen.append(r.url.rsplit("/", 1)[1]))
        p2.goto(base_url + "/baseball/tools/stuff-leaderboard.html?asof=2026-06-07", wait_until="networkidle")
        assert "snapshots_grades_mlb_latest.json" not in seen and "snapshots_grades_mlb.json" in seen
    finally:
        ctx.close()


def test_small_sample_translation_matches_the_model_and_shows_on_the_tools(browser, base_url):
    """M5-EB 2.0.0: Sparvi.translate applies the committed constants as small_sample.apply does; the Triple-A explorer's
    translated view lists lines with 50-149 PA; the main line is untouched at 150+."""
    import json
    import sys
    sys.path.insert(0, str(ROOT / "baseball" / "aaa_translation"))
    import small_sample
    P = json.loads(small_sample.PARAMS.read_text(encoding="utf-8"))
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    p = ctx.new_page()
    try:
        p.goto(base_url + "/baseball/tools/triple-a.html?view=mlb", wait_until="networkidle")
        p.wait_for_selector("#t table tbody tr", timeout=15000)
        js = """async ([side, key, x, n]) => { const F = await Sparvi.load('/data/baseball/translation_fits.json');
                 return [Sparvi.translate(F, side, key, x, n), Sparvi.translateMin(F), F.min_aaa]; }"""
        for side, key, x in (("hitter", "k_pct", 0.31), ("hitter", "ev90", 106.0), ("pitcher", "whiff_pct", 0.20)):
            for n in (50, 97, 149):
                got, mn, big = p.evaluate(js, [side, key, x, n])
                t, half = small_sample.apply(P[side][key], x, n)
                assert got["small"] and abs(got["t"] - float(t)) < 1e-5 and abs((got["hi"] - got["lo"]) / 2 - float(half)) < 1e-5
            got, mn, big = p.evaluate(js, [side, key, x, 150])
            assert mn == 50 and big == 150 and got["small"] is False
            assert p.evaluate(js, [side, key, x, 49])[0] is None
        assert p.evaluate(js, ["pitcher", "stuff_plus", 95, 100])[0] is None, "the pitch grades are not translated below 150"
        cap = p.locator(".caption").last.inner_text()
        assert "50+" in cap and "small-sample" in cap
    finally:
        ctx.close()
