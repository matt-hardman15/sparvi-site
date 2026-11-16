"""Team pages at the S2 standard (BACKLOG SEO-2b): rule-based summaries whose numbers are on the page,
plain-HTML tables, and committed pages that match what the generator writes. Runs on committed site/data."""
import html
import json
import re

import pytest

from cfb import team_summary as cfbs
from nfl.team_summary import League, static_block, summary
from sparvi.config import ROOT, cfb, nfl

NUM = re.compile(r"(?<![\w.])[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)%?")
NFL = ROOT / "site" / "data" / "nfl"
SEASON = nfl().season


@pytest.fixture(scope="module")
def nfl_league():
    teams = json.loads((NFL / "teams.json").read_text(encoding="utf-8"))
    wk = json.loads((NFL / f"team_week_{SEASON}.json").read_text(encoding="utf-8"))
    games = json.loads((NFL / f"team_games_{SEASON}.json").read_text(encoding="utf-8"))
    return League(wk, games, teams), teams


def href(t):
    return f"/nfl/teams/{t['slug']}/"


def text_of(h: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", h))


def test_nfl_summary_numbers_are_on_the_page(nfl_league):
    L, teams = nfl_league
    for t in teams:
        page = text_of(static_block(L, t["id"], SEASON, href))
        tokens = set(NUM.findall(page))
        for n in NUM.findall(summary(L, t["id"], SEASON).replace(t["name"], "")):   # "49ers" is a name
            assert n in tokens, f"{t['name']}: {n} is in the summary but not in the page's tables"


def test_nfl_committed_pages_match_the_generator(nfl_league):
    """The weekly refresh rewrites these pages; a stale page means team_pages.py didn't run."""
    L, teams = nfl_league
    for t in teams:
        qmd = (ROOT / "site" / "nfl" / "teams" / t["slug"] / "index.qmd").read_text(encoding="utf-8")
        assert html.escape(summary(L, t["id"], SEASON)) in qmd, t["name"]
        assert 'id="team-static"' in qmd and qmd.count("<table") == 2
        links = set(re.findall(r'href="(/nfl/teams/[a-z]+/)"', qmd))
        assert len(links) >= 4, f"{t['name']}: {len(links)} links to other team pages"
        for link in links:
            assert (ROOT / "site" / link.strip("/") / "index.qmd").exists(), link
    assert (ROOT / "site" / "nfl" / "teams" / "index.qmd").exists()


def test_nfl_summaries_are_plain_and_deterministic(nfl_league):
    L, teams = nfl_league
    for t in teams:
        a = summary(L, t["id"], SEASON)
        assert a == summary(L, t["id"], SEASON)
        assert not re.search(r"game-changing|unlock|dive in|leverage|elite", a, re.I)


# ---- CFB ----------------------------------------------------------------------------------------
CFB = ROOT / "site" / "data" / "cfb"


@pytest.fixture(scope="module")
def cfb_pages():
    """(team, league for the season its page opens on, that season) for every team page."""
    teams = json.loads((CFB / "teams.json").read_text(encoding="utf-8"))
    meta = json.loads((CFB / "meta.json").read_text(encoding="utf-8"))
    leagues = {}
    out = []
    for t in teams:
        last = [s for s in t["seasons"] if s in meta["seasons"]][-1]
        if last not in leagues:
            leagues[last] = cfbs.League(json.loads((CFB / f"team_week_{last}.json").read_text(encoding="utf-8")),
                                        json.loads((CFB / f"team_games_{last}.json").read_text(encoding="utf-8")), teams)
        out.append((t, leagues[last], last))
    return out, meta


def test_cfb_summary_numbers_are_on_the_page(cfb_pages):
    pages, meta = cfb_pages
    season = cfb().season
    for t, L, last in pages:
        text, _ = cfbs.summary(L, t, last, season, meta["seasons"][-1])
        page = text_of(cfbs.static_block(L, t, last, last == season, lambda x: f"/cfb/teams/{x['slug']}/"))
        tokens = set(NUM.findall(page.replace("−", "-")))
        for n in NUM.findall(text.replace(t["name"], "").replace(t["conf"], "").replace("−", "-")):   # "Pac-12" is a name
            assert n in tokens, f"{t['name']}: {n} is in the summary but not in the page's tables"


def test_cfb_committed_pages_match_the_generator(cfb_pages):
    pages, meta = cfb_pages
    season = cfb().season
    for t, L, last in pages:
        qmd = (ROOT / "site" / "cfb" / "teams" / t["slug"] / "index.qmd").read_text(encoding="utf-8")
        assert html.escape(cfbs.summary(L, t, last, season, meta["seasons"][-1])[0]) in qmd, t["name"]
        assert 'id="team-static"' in qmd and qmd.count("<table") == 2
        for link in set(re.findall(r'href="(/cfb/teams/[a-z0-9-]+/)"', qmd)):
            assert (ROOT / "site" / link.strip("/") / "index.qmd").exists(), link
    idx = (ROOT / "site" / "cfb" / "teams" / "index.qmd").read_text(encoding="utf-8")
    assert idx.count('class="team-index"') >= 10 and len(re.findall(r'href="/cfb/teams/[a-z0-9-]+/"', idx)) == len(pages)
