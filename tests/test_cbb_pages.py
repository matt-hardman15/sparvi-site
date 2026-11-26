"""College basketball team pages (CBB-PAGES-1b): one page per team with its numbers on it, valid metadata, and links
only to pages that exist."""
from __future__ import annotations

import json
import re
import sys

import pytest

from sparvi.config import ROOT

sys.path.insert(0, str(ROOT / "site" / "build"))
import cbb_teams as ct  # noqa: E402

DATA = ROOT / "site" / "data" / "cbb"


@pytest.fixture(scope="module")
def built():
    ratings = json.loads((DATA / "ratings.json").read_text(encoding="utf-8"))
    return ct.build(DATA), ratings["teams"], json.loads((DATA / "team_games.json").read_text(encoding="utf-8"))["teams"]


def test_one_page_per_team_and_an_index(built):
    pages, teams, _ = built
    paths = [p["path"] for p in pages]
    assert len(paths) == len(set(paths)) == len(teams) + 1 and "cbb/teams" in paths
    assert {p["path"] for p in pages[:-1]} == {f"cbb/teams/{t['slug']}" for t in teams}
    for t in teams:
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", t["slug"]), t["slug"]
    index = pages[-1]["body"]
    assert len(re.findall(r'<li><a href="/cbb/teams/[a-z0-9-]+/">', index)) == len(teams)


def test_metadata_fits_the_site_rules(built):
    pages, _, _ = built
    for p in pages:
        assert len(p["title"]) <= ct.TITLE_BUDGET, p["title"]
        assert 120 <= len(p["desc"]) <= 155, (p["path"], len(p["desc"]))
        assert p["h1"] and p["og"]["title"]
    assert len({p["title"] for p in pages}) == len(pages) and len({p["desc"] for p in pages}) == len(pages)


def test_summary_numbers_and_results_are_on_the_page(built):
    pages, teams, games = built
    by_path = {p["path"]: p for p in pages}
    n = len(teams)
    for t in teams:
        body = by_path[f"cbb/teams/{t['slug']}"]["body"]
        s = ct.summary(t, n, "x", "2026-11-24")
        for want in (f"{t['w']}–{t['l']}", ct.sgn(t["net"]), ct.ordinal(t["rank"])):
            assert want in s and want in body, (t["team"], want)
        for k in ("net", "off", "def"):
            assert f"<td>{ct.sgn(t[k])}</td>" in body
        g = games[str(t["id"])]
        won, lost = sum(x[4] == "final" and x[5] > x[6] for x in g), sum(x[4] == "final" and x[5] < x[6] for x in g)
        assert (won, lost) == (t["w"], t["l"]), t["team"]                      # the record is the results listed
        assert body.count("<td>W ") == won and body.count("<td>L ") == lost
        assert len(re.findall(r'href="/cbb/', body)) >= 3                       # internal links (PLAN-seo)


def test_every_link_goes_to_a_page_that_exists(built):
    pages, teams, _ = built
    known = {f"/cbb/teams/{t['slug']}/" for t in teams} | {"/cbb/teams/", "/cbb/tools/team-ratings.html", "/cbb/tools/team-ratings.html#method"}
    for p in pages:
        for href in re.findall(r'href="(/[^"]*)"', p["body"]):
            assert href in known, (p["path"], href)
        assert "<script" not in p["body"]


def test_a_game_without_a_score_or_a_non_division_i_opponent_is_shown_plainly():
    t = {"id": 1, "slug": "a", "team": "A & M", "conf": "X", "w": 1, "l": 0, "games": 1, "net": 2.0, "off": 1.5, "def": 0.5, "tempo": 68.0,
         "rank": 1, "off_rank": 1, "def_rank": 2}
    games = [["2026-11-03", None, "Small <College>", "H", "no score", None, None], ["2026-11-05", 2, "B", "N", "final", 70, 60],
             ["2026-12-01", 2, "B", "A", "postponed", None, None]]
    body = ct.body(t, games, {1: "a", 2: "b"}, [], 2, "2026–27", "2026-11-24")
    assert "Score not recorded" in body and "Small &lt;College&gt;" in body and "<a" not in body.split("Small")[1][:40]
    assert "W 70–60" in body and "Neutral" in body and "Postponed" in body and "A &amp; M" in body
