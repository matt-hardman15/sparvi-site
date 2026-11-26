"""Static college basketball team pages (CBB-PAGES-1b; PLAN-seo S2). Called by player_pages.py, which owns the shell.

- /cbb/teams/<slug>/: one page per Division I team: its rating and ranks this morning, its results, and the games
  still on its schedule, as plain HTML. Every opponent in Division I links to its own page.
- /cbb/teams/: every team by conference, with its record and net-rating rank.

Reads site/data/cbb (written by cbb/export_site_data.py). No charts and no forecasts here. Summaries are built by
rule (tests/test_team_pages.py checks their numbers are on the page). Standard library only (Quarto post-render).
"""
from __future__ import annotations

import datetime
import html
import json
from pathlib import Path

import jsonld

PREFIX = "cbb/teams/"
SITE_WORD = {"H": "Home", "A": "Away", "N": "Neutral"}
TITLE_BUDGET = 60 - len(" – Sparvi")


def e(x) -> str:
    return html.escape(str(x), quote=True)


def sgn(v: float) -> str:
    return ("+" if v > 0 else "−" if v < 0 else "") + f"{abs(v):.1f}"


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def day(iso: str) -> str:
    d = datetime.date.fromisoformat(iso)
    return f"{d.strftime('%b')} {d.day}"


def long_day(iso: str) -> str:
    d = datetime.date.fromisoformat(iso)
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def path(slug: str) -> str:
    return f"{PREFIX}{slug}"


def _table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th{' class=l' if i < 2 else ''}>{e(h)}</th>" for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f"<td{' class=l' if i < 2 else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="s-table-wrap"><table class="s-table static"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def summary(t: dict, n: int, label: str, through: str) -> str:
    """By rule: record, net rating and rank, then the stronger side of the ball by rank."""
    side = ("offense", t["off"], t["off_rank"]) if t["off_rank"] <= t["def_rank"] else ("defense", t["def"], t["def_rank"])
    return (f"{t['team']} ({t['conf']}) is {t['w']}–{t['l']} in {label} through {long_day(through)}. Its net rating is {sgn(t['net'])} "
            f"points per 100 possessions against an average Division I team, {ordinal(t['rank'])} of {n}. "
            f"Its {side[0]} ranks higher: {sgn(side[1])}, {ordinal(side[2])}.")


def titles(t: dict, label: str) -> tuple[str, str]:
    name = t["team"]
    opts = [f"{name} basketball {label}: rating and results", f"{name} basketball {label} rating", f"{name} basketball {label}", name]
    title = next(x for x in opts if len(x) <= TITLE_BUDGET)
    descs = [f"{name} men's basketball in {label}: net, offensive and defensive rating with Division I ranks, tempo, every result, and the games still to play.",
             f"{name} men's basketball {label}: net, offensive and defensive rating with Division I ranks, tempo, results, and remaining schedule.",
             f"{name} men's basketball {label}: ratings with Division I ranks, tempo, every result so far, and the games still on the schedule."]
    return title, next((x for x in descs if 120 <= len(x) <= 155), descs[-1])


def opp_link(g: list, slugs: dict) -> str:
    opp, name = g[1], g[2]
    return f'<a href="/{path(slugs[opp])}/">{e(name)}</a>' if opp in slugs else e(name)


def body(t: dict, games: list, slugs: dict, mates: list[dict], n: int, label: str, through: str) -> str:
    played = [g for g in games if g[4] in ("final", "no score")]
    ahead = [g for g in games if g[4] in ("scheduled", "postponed")]
    out = [f'<p class="kicker">{e(label)} · {e(t["conf"])}</p>',
           f'<p class="page-summary">{e(summary(t, n, label, through))}</p>',
           '<p class="note"><a href="/cbb/tools/team-ratings.html">All team ratings</a> · <a href="/cbb/teams/">All teams</a> · '
           '<a href="/cbb/tools/matchup.html?a=' + t["slug"] + '">Match this team up</a> · '
           '<a href="/cbb/tools/team-ratings.html#method">How the ratings work</a></p>',
           "<h2>Ratings</h2>",
           _table(["Measure", "", "Rating", "Rank"],
                  [["Net", "Points per 100 possessions better than an average team", sgn(t["net"]), f"{t['rank']} of {n}"],
                   ["Offense", "Points scored above average per 100 possessions", sgn(t["off"]), f"{t['off_rank']} of {n}"],
                   ["Defense", "Points prevented above average per 100 possessions", sgn(t["def"]), f"{t['def_rank']} of {n}"],
                   ["Tempo", "Possessions per 40 minutes against an average opponent", f"{t['tempo']:.1f}", "—"]])]
    if played:
        rows = []
        for g in played:
            res = "Score not recorded" if g[4] == "no score" else f"{'W' if g[5] > g[6] else 'L'} {g[5]}–{g[6]}"
            rows.append([e(day(g[0])), opp_link(g, slugs), SITE_WORD[g[3]], res])
        out += [f"<h2>Results ({t['w']}–{t['l']})</h2>", _table(["Date", "Opponent", "Site", "Result"], rows)]
    if ahead:
        rows = [[e(day(g[0])), opp_link(g, slugs), SITE_WORD[g[3]], "Postponed" if g[4] == "postponed" else ""] for g in ahead]
        out += [f"<h2>Still to play ({len(ahead)})</h2>", _table(["Date", "Opponent", "Site", ""], rows),
                '<p class="note">The regular-season schedule as the source lists it today. Conference tournament and postseason '
                "games appear once they are set.</p>"]
    if mates:
        out.append(f'<h2>{e(t["conf"])}</h2><ul class="team-index">' + "".join(
            f'<li><a href="/{path(m["slug"])}/">{e(m["team"])}</a> <span class="muted">{m["w"]}–{m["l"]} · {ordinal(m["rank"])} in net rating</span></li>'
            for m in mates) + "</ul>")
    out.append(f'<p class="note">Ratings and records run through the games of {e(long_day(through))} and are refit each morning. '
               f"The ratings count games between two Division I teams with possessions recorded ({t['games']} of this team's games so far); "
               "the record counts every game with a score. Opponents outside Division I have no page. "
               'Data provided by <a href="https://collegebasketballdata.com">CollegeBasketballData.com</a>.</p>')
    return "\n".join(out)


def build(data: Path) -> list[dict]:
    """Every team page as {path, title, h1, desc, body, ld, og}, plus the index (path 'cbb/teams')."""
    ratings = json.loads((data / "ratings.json").read_text(encoding="utf-8"))
    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))
    games = json.loads((data / "team_games.json").read_text(encoding="utf-8"))["teams"]
    teams = sorted(ratings["teams"], key=lambda t: t["team"])
    n, label, through = len(teams), meta["season_label"], ratings["through"]
    slugs = {t["id"]: t["slug"] for t in teams}
    by_conf: dict[str, list[dict]] = {}
    for t in teams:
        by_conf.setdefault(t["conf"], []).append(t)
    pages = []
    for t in teams:
        title, desc = titles(t, label)
        mates = sorted((m for m in by_conf[t["conf"]] if m["id"] != t["id"]), key=lambda m: m["rank"])
        u = f"{jsonld.BASE}{path(t['slug'])}/"
        pages.append({"path": path(t["slug"]), "title": title, "h1": f"{t['team']} basketball", "desc": desc,
                      "body": body(t, games.get(str(t["id"]), []), slugs, mates, n, label, through),
                      "ld": jsonld.sports_team(u, t["team"], "Basketball", t["conf"]),
                      "og": {"title": t["team"], "kicker": f"College basketball · {label}", "sub": f"{t['conf']} · {t['w']}–{t['l']}",
                             "stats": [("Net rating", sgn(t["net"])), ("Rank", f"{t['rank']} of {n}")]}})
    blocks = "".join(
        f"<h2>{e(conf)}</h2><ul class=\"team-index\">" + "".join(
            f'<li><a href="/{path(t["slug"])}/">{e(t["team"])}</a> <span class="muted">{t["w"]}–{t["l"]} · {ordinal(t["rank"])} in net rating</span></li>'
            for t in sorted(ts, key=lambda t: t["rank"])) + "</ul>"
        for conf, ts in sorted(by_conf.items()))
    pages.append({"path": PREFIX.rstrip("/"), "title": f"College basketball teams {label}", "h1": "College basketball teams",
                  "desc": f"Every Division I men's basketball team in {label} by conference, with its record and net-rating rank, "
                          "and a page of ratings and results for each.",
                  "og": {"title": "College basketball teams", "kicker": f"College basketball · {label}",
                         "sub": f"All {n} Division I teams by conference, with ratings and results."},
                  "body": f'<p>All {n} Division I men\'s teams, by conference, in order of net rating through {e(long_day(through))}. '
                          'Each team has a page with its ratings and results; the '
                          '<a href="/cbb/tools/team-ratings.html">ratings table</a> compares them all.</p>' + blocks})
    return pages
