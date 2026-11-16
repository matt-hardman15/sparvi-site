"""Static baseball team pages (PLAN-seo S2 team pages; BACKLOG SEO-2b). Called by player_pages.py.

- /baseball/teams/aaa/<club>/: the 30 Triple-A clubs. Hitters and pitchers whose most recent Triple-A team
  this season is the club, with their numbers as plain HTML linking every player page, the final-14-day
  flags, and the MLB parent.
- /baseball/teams/mlb/<club>/: the 30 MLB pitching staffs (the data covers MLB pitchers, not MLB hitters),
  linking the Triple-A affiliate.
- /baseball/teams/: every club, MLB with its affiliate.

No team totals or team percentiles: each player is credited to his most recent team only, so a team sum
would misstate any club whose players moved. Summaries are built by rule (tests/test_player_pages.py checks
their numbers are on the page). Club names and parents come from site/data/baseball/farm_orgs.json; MLB
names are the fixed list below. Standard library only (Quarto post-render).
"""
from __future__ import annotations

import html
import json
from collections.abc import Callable
from pathlib import Path

import summarize as sm

MLB = {"ATH": "Athletics", "ATL": "Atlanta Braves", "AZ": "Arizona Diamondbacks", "BAL": "Baltimore Orioles",
       "BOS": "Boston Red Sox", "CHC": "Chicago Cubs", "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians",
       "COL": "Colorado Rockies", "CWS": "Chicago White Sox", "DET": "Detroit Tigers", "HOU": "Houston Astros",
       "KC": "Kansas City Royals", "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins",
       "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYM": "New York Mets", "NYY": "New York Yankees",
       "PHI": "Philadelphia Phillies", "PIT": "Pittsburgh Pirates", "SD": "San Diego Padres", "SEA": "Seattle Mariners",
       "SF": "San Francisco Giants", "STL": "St. Louis Cardinals", "TB": "Tampa Bay Rays", "TEX": "Texas Rangers",
       "TOR": "Toronto Blue Jays", "WSH": "Washington Nationals"}
PREFIX = "baseball/teams/"


def e(x) -> str:
    return html.escape(str(x), quote=True)


def clubs(data: Path, season: int) -> dict[str, dict]:
    """Triple-A club -> {name, org}, from the latest season farm_orgs.json lists (at most this one)."""
    out: dict[str, dict] = {}
    for r in json.loads((data / "farm_orgs.json").read_text(encoding="utf-8")):
        if r["season"] <= season and (r["club"] not in out or r["season"] >= out[r["club"]]["season"]):
            out[r["club"]] = {"name": r["club_name"], "org": r["org"], "season": r["season"]}
    return out


def path(level: str, club: str) -> str:
    return f"{PREFIX}{level.lower()}/{club.lower()}"


def roster(players: list[dict], level: str, club: str) -> tuple[list, list]:
    """(hitters, pitchers) as (player, final) with a real sample at this level for this club."""
    hit, pit = [], []
    for p in players:
        fin = sm.final(p)
        roles = sm.roles(fin)
        if level == "AAA":
            if "hitter" in roles and fin["hitter"].get("team") == club:
                hit.append((p, fin))
            if "AAA" in roles and fin["AAA"].get("team") == club:
                pit.append((p, fin))
        elif "MLB" in roles and fin["MLB"].get("team") == club:
            pit.append((p, fin))
    return hit, pit


def _link(p: dict, href: Callable[[dict], str]) -> str:
    return f'<a href="{href(p)}">{e(p["name"])}</a>'


def _table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th{' class=l' if i == 0 else ''}>{e(h)}</th>" for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f"<td{' class=l' if i == 0 else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="s-table-wrap"><table class="s-table static"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def hitters_table(hit, href) -> str:
    rows = [[_link(p, href), str(f["hitter"]["pa"]), sm.woba(f["hitter"].get("xwoba")), sm.pct(f["hitter"].get("chase_pct")),
             sm.pct(f["hitter"].get("hard_hit_pct")), sm.num(f["hitter"].get("ev90")), sm.pct(f["hitter"].get("k_pct")),
             sm.pct(f["hitter"].get("bb_pct"))] for p, f in sorted(hit, key=lambda x: (-x[1]["hitter"]["pa"], x[0]["name"]))]
    return _table(["Hitter", "PA", "xwOBA", "Chase", "Hard-hit", "EV90", "K%", "BB%"], rows)


def pitchers_table(pit, level, href) -> str:
    k = level
    if level == "AAA":
        rows = [[_link(p, href), e(f[k].get("role") or "—"), str(f[k]["bf"]), sm.pct(f[k].get("k_pct")), sm.pct(f[k].get("bb_pct")),
                 sm.num(f[k].get("stuff_plus"), 0), sm.num(f[k].get("loc_plus"), 0), sm.num(f[k].get("pitching_plus"), 0)]
                for p, f in sorted(pit, key=lambda x: (-(x[1][k].get("bf") or 0), x[0]["name"]))]
        return _table(["Pitcher", "Role", "BF", "K%", "BB%", "Stuff+", "Location+", "Pitching+"], rows)
    rows = [[_link(p, href), e(f[k].get("role") or "—"), sm.intc(f[k]["pitches"]), sm.num(f[k].get("stuff_plus"), 0),
             sm.num(f[k].get("loc_plus"), 0), sm.num(f[k].get("pitching_plus"), 0), sm.num(f[k].get("rv_plus"), 0),
             sm.pct(f[k].get("k_pct")), sm.pct(f[k].get("bb_pct"))]
            for p, f in sorted(pit, key=lambda x: (-x[1][k]["pitches"], x[0]["name"]))]
    return _table(["Pitcher", "Role", "Pitches", "Stuff+", "Location+", "Pitching+", "RV+", "K%", "BB%"], rows)


def best(rows, key: str, field: str, need: Callable[[dict], bool]):
    c = [(p, f) for p, f in rows if f[key].get(field) is not None and need(f[key])]
    return max(c, key=lambda x: (x[1][key][field], x[0]["name"]), default=None)


def summary(level: str, name: str, season: int, hit, pit) -> str:
    """By rule: roster size, the top xwOBA among qualified hitters, the top Stuff+ among pitchers with a full sample."""
    if level == "AAA":
        s = [f"Sparvi has {season} profiles for {len(hit)} hitters and {len(pit)} pitchers whose most recent Triple-A team is the {name}."]
        b = best(hit, "hitter", "xwoba", lambda r: r.get("qualified"))
        if b:
            s.append(f"Among qualified hitters, {b[0]['name']} has the highest xwOBA ({sm.woba(b[1]['hitter']['xwoba'])}).")
        b = best(pit, "AAA", "stuff_plus", lambda r: (r.get("bf") or 0) >= 100)
        if b:
            s.append(f"Among pitchers with 100+ batters faced, {b[0]['name']} has the highest Stuff+ ({sm.num(b[1]['AAA']['stuff_plus'], 0)}).")
    else:
        s = [f"Sparvi has {season} profiles for {len(pit)} pitchers whose most recent MLB team is the {name}, each with {sm.MIN_MLB_PITCHES}+ pitches."]
        b = best(pit, "MLB", "stuff_plus", lambda r: r["pitches"] >= 500)
        if b:
            s.append(f"Among those with 500+ pitches, {b[0]['name']} has the highest Stuff+ ({sm.num(b[1]['MLB']['stuff_plus'], 0)}), "
                     "where 100 is average.")
    return " ".join(s)


def flags_html(players: list[dict], club: str, href) -> str:
    items = []
    for p in sorted(players, key=lambda p: p["name"]):
        for fl in p.get("flags") or []:
            txt = "; ".join(x for x in (fl.get("flag_discipline"), fl.get("flag_contact")) if x)
            if txt and fl.get("team") == club:
                items.append(f"<li>{_link(p, href)}: {e(txt)}</li>")
    return ('<div class="flag-list"><strong>Final 14 days</strong><ul>' + "".join(items) + "</ul></div>") if items else ""


def page_body(level, club, info, season, hit, pit, players, href, through: str, affiliate_link: str) -> str:
    name = info["name"]
    out = [f'<p class="kicker">{season} · {"Triple-A" if level == "AAA" else "MLB"}</p>',
           f'<p class="page-summary">{e(summary(level, name, season, hit, pit))}</p>',
           f'<p class="note">Season to date through {e(through)}. {affiliate_link} · <a href="/baseball/teams/">All teams</a> · '
           + ('<a href="/baseball/tools/triple-a.html">Triple-A explorer</a>' if level == "AAA"
              else '<a href="/baseball/tools/stuff-leaderboard.html">Pitch-grade leaderboard</a>') + "</p>"]
    if hit:
        out += [f"<h2>Hitters ({len(hit)})</h2>", hitters_table(hit, href)]
    if pit:
        out += [f"<h2>Pitchers ({len(pit)})</h2>", pitchers_table(pit, level, href)]
    out.append(flags_html([p for p, _ in hit + pit], club, href) if level == "AAA" else "")
    out.append('<p class="note">Players are listed with their most recent team this season, with at least '
               f"{sm.MIN_PA} PA, {sm.MIN_BF} batters faced, or {sm.MIN_MLB_PITCHES} MLB pitches. Stuff+, Location+, Pitching+: "
               "100 is an average qualified MLB pitcher. No team totals: a player's whole season counts toward his latest team only.</p>")
    return "\n".join(x for x in out if x)


def titles(level: str, name: str, season: int) -> tuple[str, str]:
    budget = 60 - len(" – Sparvi")
    if level == "AAA":
        t = [f"{name} {season} roster: Statcast profiles", f"{name} {season} Statcast roster", f"{name} {season}"]
        d = [f"{name} {season} Triple-A roster: every hitter's swing decisions and contact quality and every pitcher's Stuff+, with links to full profiles.",
             f"{name} {season} Triple-A roster: hitters' swing decisions and contact quality, pitchers' Stuff+, and full player profiles."]
    else:
        t = [f"{name} {season} pitching staff: Stuff+ grades", f"{name} {season} pitchers: Stuff+", f"{name} {season} pitchers"]
        d = [f"{name} {season} pitching staff: Stuff+, Location+, Pitching+, and results for every pitcher with 100+ MLB pitches, with full profiles.",
             f"{name} {season} pitchers: Stuff+, Location+, Pitching+, and results for every pitcher with 100+ MLB pitches."]
    title = next(x for x in t if len(x) <= budget)
    desc = next((x for x in d if 120 <= len(x) <= 155), d[-1])
    return title, desc


def build(players: list[dict], data: Path, season: int, href: Callable[[dict], str]) -> list[dict]:
    """Every team page as {path, title, h1, desc, body}, plus the index (path 'baseball/teams')."""
    cl = clubs(data, season)
    aff = {v["org"]: k for k, v in cl.items()}
    dates = {lvl: max((p["timeline"][lvl]["dates"][-1] for p in players if lvl in (p.get("timeline") or {})), default=None)
             for lvl in ("AAA", "MLB")}
    pages = []
    for club, info in sorted(cl.items()):
        hit, pit = roster(players, "AAA", club)
        org = info["org"]
        link = f'Affiliate of the <a href="/{path("MLB", org)}/">{e(MLB.get(org, org))}</a>'
        title, desc = titles("AAA", info["name"], season)
        pages.append({"path": path("AAA", club), "title": title, "h1": info["name"], "desc": desc, "hit": hit, "pit": pit,
                      "body": page_body("AAA", club, info, season, hit, pit, players, href, sm_short(dates["AAA"]), link)})
    for org, name in sorted(MLB.items()):
        _, pit = roster(players, "MLB", org)
        club = aff.get(org)
        link = (f'Triple-A affiliate: <a href="/{path("AAA", club)}/">{e(cl[club]["name"])}</a>' if club else "")
        title, desc = titles("MLB", name, season)
        pages.append({"path": path("MLB", org), "title": title, "h1": name, "desc": desc, "hit": [], "pit": pit,
                      "body": page_body("MLB", org, {"name": name}, season, [], pit, players, href, sm_short(dates["MLB"]), link)})
    rows = "".join(
        f'<li><a href="/{path("MLB", org)}/">{e(name)}</a>'
        + (f' <span class="muted">· Triple-A:</span> <a href="/{path("AAA", aff[org])}/">{e(cl[aff[org]]["name"])}</a>' if org in aff else "")
        + "</li>" for org, name in sorted(MLB.items(), key=lambda kv: kv[1]))
    pages.append({"path": PREFIX.rstrip("/"), "title": f"Baseball teams {season}: MLB and Triple-A rosters", "h1": "Baseball teams",
                  "desc": f"Every MLB club's {season} pitching staff and its Triple-A affiliate's roster, with Stuff+, swing decisions, "
                          "and links to every player profile.",
                  "hit": [], "pit": [],
                  "body": f'<p>Each MLB club links to its pitching staff; its Triple-A affiliate links to that club\'s hitters and pitchers. '
                          f'Every player links to a full profile; see also the <a href="/baseball/players/">list of player profiles</a>.</p>'
                          f'<ul class="team-index">{rows}</ul>'})
    return pages


def sm_short(iso: str | None) -> str:
    if not iso:
        return "—"
    import datetime
    d = datetime.date.fromisoformat(iso)
    return f"{d.strftime('%b')} {d.day}"
