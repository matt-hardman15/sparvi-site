"""Static baseball player pages (PLAN-seo S2). Quarto post-render step, before seo.py.

For every player in site/data/baseball/players/ it writes /baseball/players/<slug>-<id>/index.html:
the site's page shell (rendered by Quarto from site/baseball/players/template/), the player's own title,
description, and canonical URL, a data-driven summary, the season's numbers as plain HTML tables, and
the player's data embedded in the page for the interactive charts (site/assets/js/player.js). One file
per player; the per-player JSON files are removed from the output.

- Indexed pages: players at or above the PLAN-seo S2 thresholds (summarize.indexable). The rest keep a
  page, since tools link to every player, but it is noindex and left out of the sitemap.
- Old paths (site/data/baseball/player_urls.json, which only grows): a noindex redirect to the current
  page. A player in the registry with no data this season keeps his page, built from his last season in
  site/data/baseball/archive/players_<season>.json (sparvi/player_archive.py): static numbers, no charts.
- Team pages (baseball_teams.py): /baseball/teams/, 30 Triple-A rosters, 30 MLB staffs; sitemap-baseball-teams.xml.
- sitemap-baseball-players.xml: lastmod is the last commit that changed the player's data file or the
  page code, not the build time.
- /baseball/players/ (the A-Z index) gets its list here.

Standard library only: the deploy job runs Quarto without the project's virtualenv.
"""
from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseball_teams as bt  # noqa: E402
import summarize as sm  # noqa: E402

SITE = Path(__file__).resolve().parents[1]
REPO = SITE.parent
OUT = Path(os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", REPO / "docs"))
if not OUT.is_absolute():
    OUT = SITE / OUT
DATA = SITE / "data" / "baseball"
BASE = "https://sparvi.io/"
PREFIX = "baseball/players/"
SHELL = OUT / PREFIX / "template" / "index.html"
CODE = ["site/baseball/players/template/index.qmd", "site/assets/js/player.js", "site/build/summarize.py",
        "site/build/player_pages.py", "site/build/baseball_teams.py"]
MONTHS = ["Jan.", "Feb.", "March", "April", "May", "June", "July", "Aug.", "Sept.", "Oct.", "Nov.", "Dec."]
TITLE_BUDGET = 60 - len(" – Sparvi")
DESC_MIN, DESC_MAX = 120, 155


def e(x) -> str:
    return html.escape(str(x), quote=True)


def ap(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


def short(iso: str) -> str:            # Sparvi.dateLabel: "Sep 20"
    d = date.fromisoformat(iso)
    return f"{d.strftime('%b')} {d.day}"


def url(path: str) -> str:
    return f"{BASE}{PREFIX}{path}/"


def first_fit(options: list[str], lo: int, hi: int) -> str | None:
    return next((o for o in options if lo <= len(o) <= hi), None)


def titles(p: dict, fin: dict, season: int, dup: bool) -> tuple[str, str]:
    """(page title, description) within the S1 limits. Duplicate names carry the team."""
    name = p["name"]
    team = next(iter(sm.teams(p, fin)), "").split(" (")[0]
    who = f"{name} ({team})" if dup and team else name
    pitcher = bool({"AAA", "MLB"} & set(sm.roles(fin)))
    if pitcher:
        t = [f"{who} {season} stats: Stuff+ and pitch grades", f"{who} {season} Stuff+ and pitch grades",
             f"{who} {season} stats and Stuff+", f"{who} {season} stats"]
    else:
        t = [f"{who} {season} Triple-A stats and percentiles", f"{who} {season} Triple-A stats", f"{who} {season} stats"]
    title = first_fit(t, 1, TITLE_BUDGET) or who[:TITLE_BUDGET]
    where = "; ".join(t.replace(" (", ", ").rstrip(")") for t in sm.teams(p, fin)) or "MLB and Triple-A"
    if pitcher:
        tails = [f" {season} Statcast profile: Stuff+, Location+, and Pitching+ by pitch, strikeout and walk rates, and how each moved through the season.",
                 f" {season} Statcast profile: Stuff+, Location+, and Pitching+ by pitch, and how each grade moved through the season.",
                 f" {season} Statcast profile: Stuff+, Location+, and Pitching+ by pitch, plus season trends.",
                 f" {season}: Stuff+, Location+, Pitching+ by pitch."]
    else:
        tails = [f" {season} Statcast profile: chase rate, zone contact, EV90, xwOBA, Triple-A percentiles, and how each moved through the season.",
                 f" {season} Statcast profile: chase rate, zone contact, EV90, xwOBA, and Triple-A percentiles through the season.",
                 f" {season} Statcast profile: swing decisions, contact quality, and Triple-A percentiles.",
                 f" {season}: swing decisions and contact quality."]
    desc = first_fit([f"{name} ({where}){t}" for t in tails], DESC_MIN, DESC_MAX) \
        or first_fit([f"{name}{t}" for t in tails], DESC_MIN, DESC_MAX) \
        or f"{name} ({where}){tails[-1]}"
    return title, desc




def kicker(p: dict, season: int) -> str:
    bio = p.get("hitter_aaa") or p.get("pitcher_aaa")
    lv = lambda lvl: "Triple-A" if lvl == "AAA" else "MLB"  # noqa: E731
    teams = list(dict.fromkeys(([f"{bio['team']} (Triple-A)"] if bio else [])
                               + [f"{d['team']} ({lv(d['level'])})" for d in p.get("stuff") or []]))
    age = f" · Age {bio['age']}" if bio and bio.get("age") else ""
    return f"{season} · {' · '.join(teams)}{age}"


def through(p: dict) -> dict[str, str]:
    tl = p.get("timeline") or {}
    return {lvl: short(tl[lvl]["dates"][-1]) for lvl in ("AAA", "MLB") if (tl.get(lvl) or {}).get("dates")}


def body(p: dict, season: int, fin: dict, current: int | None = None) -> str:
    """The page body. With `current` (an archived player with no data that season): static numbers only."""
    th = through(p)
    latest = max((p["timeline"][lvl]["dates"][-1] for lvl in th), default=None)
    # The player's team pages (baseball_teams.py), then the tools.
    team_links = [(f"/{bt.path(lvl, fin[k]['team'])}/", f"{fin[k]['team']} ({'Triple-A' if lvl == 'AAA' else 'MLB'})")
                  for k, lvl in (("hitter", "AAA"), ("AAA", "AAA"), ("MLB", "MLB")) if k in sm.roles(fin) and fin[k].get("team")
                  and not current]   # an archived season's clubs are not this season's team pages
    links = " · ".join(f'<a href="{h}">{e(t)}</a>' for h, t in list(dict.fromkeys(team_links)) + [
        ("/baseball/tools/stuff-leaderboard.html", "Pitch-grade leaderboard"), ("/baseball/tools/triple-a.html", "Triple-A explorer"),
        ("/baseball/players/", "All player profiles"), ("/glossary.html", "Glossary")])
    data = json.dumps(p, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    when = (f"No {current} data for him yet: these are his {season} numbers, his latest season in this site's data. "
            if current else ("Data through " + e(ap(latest)) + ". " if latest else ""))
    out = [f'<p class="kicker">{e(kicker(p, season))}</p>',
           f'<p class="player-summary">{e(sm.summary(p, season))}</p>',
           f'<p class="note">{when}See also: {links}</p>',
           '<div id="root">', sm.static_body(p, th), "</div>"]
    if not current:   # the interactive version needs the full season's timeline, which the archive doesn't keep
        out += [f'<script type="application/json" id="page-data">{data}</script>', '<script src="/assets/js/player.js"></script>']
    return "\n".join(out)


# Slots in the shell (site/baseball/players/template/index.qmd). Letters only: Pandoc would read
# punctuation such as @ in the front matter as Markdown.
SLOTS = ("ZZTITLEZZ", "ZZHONEZZ", "ZZDESCRIPTIONZZ", "ZZBODYZZ")


def fill(shell: str, title: str, h1: str, desc: str, body_html: str, canonical: str, noindex: bool) -> str:
    out = shell.replace("ZZTITLEZZ", e(title)).replace("ZZHONEZZ", e(h1)).replace("ZZDESCRIPTIONZZ", e(desc))
    out = out.replace("ZZBODYZZ", body_html)
    head = f'<link rel="canonical" href="{canonical}">' + ('\n<meta name="robots" content="noindex">' if noindex else "")
    out = re.sub(r'<link rel="canonical"[^>]*>\n?', "", out)
    return out.replace("</head>", head + "\n</head>", 1)


def redirect(to: str) -> str:
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n<title>Moved · Sparvi</title>\n"
            f'<meta name="robots" content="noindex">\n<link rel="canonical" href="{to}">\n'
            f'<meta http-equiv="refresh" content="0; url={to}">\n'
            f"<script>location.replace({json.dumps(to)} + location.search + location.hash)</script>\n"
            f'</head><body><p>This page moved to <a href="{to}">{to}</a>.</p></body></html>\n')


def git_dates() -> dict[str, str]:
    """Last commit date per player data file, in one git call."""
    r = subprocess.run(["git", "log", "--format=@%cs", "--name-only", "--", "site/data/baseball/players"],
                       cwd=REPO, capture_output=True, text=True, encoding="utf-8")
    out, cur = {}, None
    for line in r.stdout.splitlines():
        if line.startswith("@"):
            cur = line[1:]
        elif line.strip():
            out.setdefault(line.strip(), cur)
    return out


def git_date(paths: list[str]) -> str | None:
    r = subprocess.run(["git", "log", "-1", "--format=%cs", "--", *paths], cwd=REPO, capture_output=True, text=True)
    return r.stdout.strip() or None


def code_date() -> str | None:
    return git_date(CODE)


def last_name(name: str) -> str:
    parts = [x for x in name.split() if x.rstrip(".").lower() not in {"jr", "sr", "ii", "iii", "iv"}]
    return (parts[-1] if parts else name).lower()


def index_html(rows: list[tuple[str, str, str]]) -> str:
    """rows: (name, path, team line), already sorted. Grouped by the first letter of the last name."""
    groups: dict[str, list] = {}
    for r in rows:
        k = last_name(r[0])[:1].upper()
        groups.setdefault(k if k.isalpha() else "#", []).append(r)
    nav = " ".join(f'<a href="#letter-{k}">{k}</a>' for k in groups)
    parts = [f'<p class="letter-nav">{nav}</p>']
    for k, rs in groups.items():
        parts.append(f'<h2 id="letter-{k}">{k}</h2><ul class="player-index">'
                     + "".join(f'<li><a href="/{PREFIX}{p}/">{e(n)}</a> <span class="muted">{e(t)}</span></li>' for n, p, t in rs)
                     + "</ul>")
    return "\n".join(parts)


def main() -> int:
    if os.environ.get("QUARTO_PROJECT_RENDER_ALL") not in (None, "1") or not SHELL.exists():
        return 0                                        # quarto preview / single-file renders
    shell = SHELL.read_text(encoding="utf-8")
    # Root-relative links, so the shell works at any depth (team pages sit one level deeper than player pages).
    shell = re.sub(r'((?:href|src)=")(?:\.\./)+', r"\1/", shell)
    lost = [s for s in SLOTS if s not in shell]
    if lost:
        print(f"player_pages: the rendered shell lost its slots {lost}", file=sys.stderr)
        return 1
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    season = meta["season"]
    reg: dict[str, list[str]] = json.loads((DATA / "player_urls.json").read_text(encoding="utf-8"))
    ids = json.loads((DATA / "players_index.json").read_text(encoding="utf-8"))
    players = [json.loads((DATA / "players" / f"{i}.json").read_text(encoding="utf-8")) for i in ids]
    missing = [p["id"] for p in players if str(p["id"]) not in reg]
    if missing:
        print(f"player_pages: {len(missing)} players missing from player_urls.json (rerun the export): {missing[:5]}",
              file=sys.stderr)
        return 1
    name_n = Counter(p["name"] for p in players)
    dates, cdate = git_dates(), code_date()
    entries, index_rows, n_noindex = [], [], 0

    shutil.rmtree(OUT / PREFIX / "template")
    for p in players:
        fin = sm.final(p)
        path = reg[str(p["id"])][-1]
        ok = sm.indexable(fin)
        title, desc = titles(p, fin, season, name_n[p["name"]] > 1)
        page = fill(shell, title, p["name"], desc, body(p, season, fin), url(path), noindex=not ok)
        d = OUT / PREFIX / path
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.html").write_text(page, encoding="utf-8", newline="\n")
        if ok:
            lm = max(x for x in (dates.get(f"site/data/baseball/players/{p['id']}.json"), cdate, "1970-01-01") if x)
            entries.append((url(path), lm))
            label = {"hitter": "Triple-A hitter", "AAA": "Triple-A pitcher", "MLB": "MLB pitcher"}
            roles = [label[k] for k in sm.roles(fin)]
            teams = ", ".join(dict.fromkeys(fin[k]["team"] for k in sm.roles(fin) if fin[k].get("team")))
            index_rows.append((p["name"], path, f"{teams} · {', '.join(roles)}"))
        else:
            n_noindex += 1
    current = {str(p["id"]) for p in players}
    # Players with no data this season: their last season's page, from the newest archive that has them.
    archives = sorted(DATA.glob("archive/players_*.json"), key=lambda f: -int(f.stem.split("_")[1]))
    loaded = {f: json.loads(f.read_text(encoding="utf-8")) for f in archives}
    n_archived = 0
    for pid, paths in reg.items():
        if pid in current:
            continue
        f = next((f for f in archives if pid in loaded[f] and int(f.stem.split("_")[1]) < season), None)
        if f is None:
            continue
        p = loaded[f][pid]
        fin, last = sm.final(p), p["season"]
        ok = sm.indexable(fin)
        title, desc = titles(p, fin, last, False)
        page = fill(shell, title, p["name"], desc, body(p, last, fin, current=season), url(paths[-1]), noindex=not ok)
        d = OUT / PREFIX / paths[-1]
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.html").write_text(page, encoding="utf-8", newline="\n")
        current.add(pid)
        n_archived += 1
        if ok:
            entries.append((url(paths[-1]), max(x for x in (git_date([f.relative_to(REPO).as_posix()]), cdate, "1970-01-01") if x)))
            index_rows.append((p["name"], paths[-1], f"{last} season"))
    n_redirects = 0
    for pid, paths in reg.items():
        target = url(paths[-1]) if pid in current else f"{BASE}{PREFIX}"
        for old in paths[:-1] if pid in current else paths:
            d = OUT / PREFIX / old
            d.mkdir(parents=True, exist_ok=True)
            (d / "index.html").write_text(redirect(target), encoding="utf-8", newline="\n")
            n_redirects += 1

    # One file per player page: the per-player JSON is embedded, so it isn't published separately.
    shutil.rmtree(OUT / "data" / "baseball" / "players", ignore_errors=True)
    shutil.rmtree(OUT / "data" / "baseball" / "archive", ignore_errors=True)
    (OUT / "data" / "baseball" / "players_index.json").unlink(missing_ok=True)

    # The A-Z index, sorted by last name then first.
    idx = OUT / PREFIX / "index.html"
    index_rows.sort(key=lambda r: (last_name(r[0]), r[0].lower(), r[1]))
    if idx.exists():                                    # absent only in a partial render
        idx.write_text(idx.read_text(encoding="utf-8").replace("ZZPLAYERINDEXZZ", index_html(index_rows)), encoding="utf-8", newline="\n")

    # The shell isn't a page: drop it from Quarto's sitemap and search index.
    sm_path = OUT / "sitemap.xml"
    s = sm_path.read_text(encoding="utf-8")
    sm_path.write_text(re.sub(r"\s*<url>\s*<loc>[^<]*/baseball/players/template/[^<]*</loc>.*?</url>", "", s, flags=re.S),
                       encoding="utf-8", newline="\n")
    search = OUT / "search.json"
    if search.exists():
        docs = [d for d in json.loads(search.read_text(encoding="utf-8")) if "players/template/" not in d.get("href", "")]
        search.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8", newline="\n")

    entries.sort()
    (OUT / "sitemap-baseball-players.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join(f"  <url>\n    <loc>{u}</loc>\n    <lastmod>{d}</lastmod>\n  </url>\n" for u, d in entries)
        + "</urlset>\n", encoding="utf-8", newline="\n")
    print(f"player_pages: {len(players)} pages ({len(entries)} indexed, {n_noindex} noindex), {n_archived} from last season, "
          f"{n_redirects} redirects")

    # Team pages: same shell, their own sitemap.
    by_id = {str(p["id"]): reg[str(p["id"])][-1] for p in players}
    team_entries = []
    tdate = max(x for x in (git_date(["site/data/baseball"]), cdate, "1970-01-01") if x)
    for t in bt.build(players, DATA, season, lambda p: f"/{PREFIX}{by_id[str(p['id'])]}/"):
        u = f"{BASE}{t['path']}/"
        d = OUT / t["path"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.html").write_text(fill(shell, t["title"], t["h1"], t["desc"], t["body"], u, noindex=False), encoding="utf-8", newline="\n")
        team_entries.append((u, tdate))
    team_entries.sort()
    (OUT / "sitemap-baseball-teams.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join(f"  <url>\n    <loc>{u}</loc>\n    <lastmod>{d}</lastmod>\n  </url>\n" for u, d in team_entries)
        + "</urlset>\n", encoding="utf-8", newline="\n")
    print(f"player_pages: {len(team_entries)} team pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
