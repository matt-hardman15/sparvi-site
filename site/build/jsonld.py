"""Structured data (schema.org JSON-LD) for every page (PLAN-seo S3). Standard library only.

seo.py adds a BreadcrumbList to every page in a sitemap, plus Article (articles), Dataset (tools), SportsTeam
(NFL/CFB team pages), and WebSite + Organization (home). The baseball generators add ProfilePage (players)
and SportsTeam (teams) through the same builders. Dataset `license` appears only where the source grants
one (nflverse: CC-BY 4.0); MLB- and CFBD-derived data carry none.
"""
from __future__ import annotations

import json

BASE = "https://sparvi.io/"
ORG = {"@type": "Organization", "@id": f"{BASE}#org", "name": "Sparvi", "url": BASE,
       "logo": {"@type": "ImageObject", "url": f"{BASE}assets/logo.png", "width": 512, "height": 512}}

# What each sport's tools are built on (Dataset.isBasedOn) and the license where the source grants one.
SOURCES = {
    "baseball": ([{"@type": "Dataset", "name": "Statcast (Baseball Savant)", "url": "https://baseballsavant.mlb.com"},
                  {"@type": "WebAPI", "name": "MLB Stats API", "url": "https://statsapi.mlb.com"}], None),
    "nfl": ([{"@type": "Dataset", "name": "nflverse play-by-play", "url": "https://github.com/nflverse"}],
            "https://creativecommons.org/licenses/by/4.0/"),
    "cfb": ([{"@type": "Dataset", "name": "CollegeFootballData.com", "url": "https://collegefootballdata.com"}], None),
}


def script(obj: dict) -> str:
    """A <script type="application/ld+json"> block; '</' is escaped so the JSON can't end the tag."""
    return ('<script type="application/ld+json">'
            + json.dumps({"@context": "https://schema.org", **obj}, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
            + "</script>")


def breadcrumbs(items: list[tuple[str, str]]) -> dict:
    """items: (name, absolute URL), root first, the page itself last."""
    return {"@type": "BreadcrumbList", "itemListElement": [
        {"@type": "ListItem", "position": i + 1, "name": n, "item": u} for i, (n, u) in enumerate(items)]}


def article(url: str, headline: str, description: str, image: str, published: str | None, modified: str) -> dict:
    out = {"@type": "Article", "headline": headline[:110], "description": description, "image": [image],
           "dateModified": modified, "author": {"@id": ORG["@id"], "@type": "Organization", "name": "Sparvi", "url": BASE},
           "publisher": ORG, "mainEntityOfPage": {"@type": "WebPage", "@id": url}}
    if published:
        out["datePublished"] = published
    return out


def dataset(url: str, name: str, description: str, sport: str, coverage: str | None, modified: str) -> dict:
    based, lic = SOURCES[sport]
    out = {"@type": "Dataset", "name": name, "description": description, "url": url, "creator": ORG,
           "isBasedOn": based, "dateModified": modified, "isAccessibleForFree": True}
    if coverage:
        out["temporalCoverage"] = coverage
    if lic:
        out["license"] = lic
    return out


def sports_team(url: str, name: str, sport: str, member_of: str | None = None, parent: str | None = None) -> dict:
    out = {"@type": "SportsTeam", "name": name, "sport": sport, "url": url}
    if member_of:
        out["memberOf"] = {"@type": "SportsOrganization", "name": member_of}
    if parent:
        out["parentOrganization"] = {"@type": "SportsTeam", "name": parent}
    return out


def profile_page(url: str, name: str, teams: list[str], modified: str | None = None) -> dict:
    person = {"@type": "Person", "name": name, "url": url}
    if teams:
        person["affiliation"] = [{"@type": "SportsTeam", "name": t, "sport": "Baseball"} for t in teams]
    out = {"@type": "ProfilePage", "url": url, "mainEntity": person}
    if modified:
        out["dateModified"] = modified
    return out


def website() -> list[dict]:
    return [{"@type": "WebSite", "name": "Sparvi", "url": BASE, "publisher": {"@id": ORG["@id"]}}, ORG]
