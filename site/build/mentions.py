"""Entity links and "Mentioned in" (PLAN-seo S5). Quarto post-render step, after player_pages.py and social.py,
before seo.py. Standard library only.

1. First-mention links. The rule is that articles link every player and team on first mention. Much of the
   prose naming players is generated ({{< meta >}} values from the exports), so this step applies the rule after
   render: in each article (pages with a dcterms.date), the first plain-text mention of a profiled baseball
   player (exact full name, only when one player has it), a baseball club, an NFL team, or a CFB team in that
   article's sport becomes a link to its page. Text inside links, headings, scripts, styles, code, buttons, and
   labels is left alone; longer names win ("Georgia Tech" before "Georgia").
2. "Mentioned in": every player and team page that an article links to (by hand or by step 1) gets a list of
   those articles, newest first.
Deterministic: the same pages give the same output.
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseball_teams as bt  # noqa: E402

SITE = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", SITE.parent / "docs"))
if not OUT.is_absolute():
    OUT = SITE / OUT
DATA = SITE / "data"
SKIP = {"a", "h1", "h2", "h3", "h4", "h5", "h6", "script", "style", "code", "pre", "button", "label", "title", "select", "option"}
TOKEN = re.compile(r"(<[^>]+>)")
TAGNAME = re.compile(r"<\s*(/?)\s*([a-zA-Z0-9]+)")
ENTITY = re.compile(r'href="(?:\.\./)*/?((?:baseball/players/[a-z0-9-]+-\d+|baseball/teams/(?:aaa|mlb)/[a-z]+|nfl/teams/[a-z]+|cfb/teams/[a-z0-9-]+)/)"')


def entities() -> dict[str, dict[str, str]]:
    """sport -> {display name: page path}. Baseball players only when the name is unique among current players."""
    out: dict[str, dict[str, str]] = {"baseball": {}, "nfl": {}, "cfb": {}}
    reg = json.loads((DATA / "baseball" / "player_urls.json").read_text(encoding="utf-8"))
    names = {}
    for f in (DATA / "baseball" / "players").glob("*.json"):
        p = json.loads(f.read_text(encoding="utf-8"))
        names[str(p["id"])] = p["name"]
    n = Counter(names.values())
    for pid, name in names.items():
        if n[name] == 1 and len(name) > 4 and pid in reg:
            out["baseball"][name] = f"baseball/players/{reg[pid][-1]}/"
    season = json.loads((DATA / "baseball" / "meta.json").read_text(encoding="utf-8"))["season"]
    for club, info in bt.clubs(DATA / "baseball", season).items():
        out["baseball"][info["name"]] = f"{bt.path('AAA', club)}/"
    for abbr, name in bt.MLB.items():
        out["baseball"][name] = f"{bt.path('MLB', abbr)}/"
    for sport in ("nfl", "cfb"):
        for t in json.loads((DATA / sport / "teams.json").read_text(encoding="utf-8")):
            if len(t["name"]) > 3:
                out[sport][t["name"]] = f"{sport}/teams/{t['slug']}/"
    return out


def link_first_mentions(h: str, names: dict[str, str]) -> tuple[str, list[str]]:
    """Link the first plain-text mention of each name inside <main>; returns (html, names linked)."""
    if not names or "<main" not in h:
        return h, []
    a, b = h.index("<main"), h.index("</main>")
    already = set(ENTITY.findall(h[a:b]))
    alts = sorted(names, key=lambda s: (-len(s), s))
    pat = re.compile(r"(?<![\w-])(" + "|".join(re.escape(html.escape(x, quote=False)) for x in alts) + r")(?![\w-])")
    by_escaped = {html.escape(x, quote=False): x for x in alts}
    parts, depth, done = TOKEN.split(h[a:b]), Counter(), []
    for i, part in enumerate(parts):
        if part.startswith("<"):
            m = TAGNAME.match(part)
            if m and m[2].lower() in SKIP and not part.endswith("/>"):
                depth[m[2].lower()] += -1 if m[1] else 1
            continue
        if not part.strip() or any(v > 0 for v in depth.values()):
            continue

        def sub(m):
            name = by_escaped[m[1]]
            path = names[name]
            if path in already:
                return m[0]
            already.add(path)
            done.append(name)
            return f'<a href="/{path}" class="entity-link">{m[0]}</a>'
        parts[i] = pat.sub(sub, part)
    return h[:a] + "".join(parts) + h[b:], done


def article_info(h: str, rel: str) -> tuple[str, str, str]:
    title = re.search(r'<h1 class="title">(.*?)</h1>', h, re.S)
    date = re.search(r'<meta name="dcterms.date" content="([^"]*)"', h)
    url = "/" + rel[: -len("index.html")] if rel.endswith("index.html") else "/" + rel
    return (re.sub(r"<[^>]+>", "", title[1]).strip() if title else url, date[1] if date else "", url)


def mentions_html(items: list[tuple[str, str, str]]) -> str:
    li = "".join(f'<li><a href="{u}">{html.escape(t)}</a></li>' for t, _, u in items)
    return f'<section class="mentioned-in"><h2>Mentioned in</h2><ul>{li}</ul></section>'


def main() -> int:
    if os.environ.get("QUARTO_PROJECT_RENDER_ALL") not in (None, "1"):
        return 0
    ents = entities()
    mentioned: dict[str, list] = defaultdict(list)
    n_links = 0
    for page in sorted(OUT.glob("*/*/index.html")):
        rel = page.relative_to(OUT).as_posix()
        sport = rel.split("/")[0]
        h = page.read_text(encoding="utf-8")
        if sport not in ents or 'name="dcterms.date"' not in h:
            continue
        h, done = link_first_mentions(h, ents[sport])
        n_links += len(done)
        page.write_text(h, encoding="utf-8", newline="\n")
        info = article_info(h, rel)
        main_html = h[h.index("<main"):h.index("</main>")]
        for path in sorted(set(ENTITY.findall(main_html))):
            mentioned[path].append(info)
    n_pages = 0
    for path, items in sorted(mentioned.items()):
        page = OUT / path / "index.html"
        if not page.exists():
            continue
        items = sorted(set(items), key=lambda x: (x[1], x[0]), reverse=True)
        h = page.read_text(encoding="utf-8")
        h = re.sub(r'<section class="mentioned-in">.*?</section>', "", h, flags=re.S)   # rerun-safe
        page.write_text(h.replace("</main>", mentions_html(items) + "\n</main>", 1), encoding="utf-8", newline="\n")
        n_pages += 1
    print(f"mentions: {n_links} first-mention links added; {n_pages} player/team pages list the articles that mention them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
