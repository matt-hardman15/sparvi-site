"""Post-render search step (PLAN-seo S1). Runs after every full `quarto render` (see _quarto.yml).

1. Canonical URLs: every page in the sitemap gets <link rel="canonical"> to its clean URL
   (https://sparvi.io/x/ rather than .../x/index.html), and the sitemap lists the same URLs.
2. lastmod: from the last git commit that touched the page's inputs (its source, plus the sport's
   data and prose numbers for data pages), not the build time, so it changes only when content does.
3. sitemap-index.xml (sitemap.xml plus each generated sitemap-*.xml) and robots.txt pointing at it.
   Pages marked noindex (redirects such as the old player.html) are left out of sitemap.xml.
4. Listing thumbnails get alt="" (decorative: the headline is right next to them).
5. Structured data (PLAN-seo S3, site/build/jsonld.py): a BreadcrumbList on every sitemap page except the home
   page, built from the URL's ancestors that are pages themselves (named by their <h1>); Article on articles,
   Dataset on tools, SportsTeam on NFL/CFB team pages, WebSite + Organization on the home page. The baseball
   generators add ProfilePage and SportsTeam themselves.
6. Checks that fail the build: every page in every sitemap has a title <= 60 characters, a description of
   120-155 characters, both unique across the site, a canonical URL, exactly one <h1>, an og:image,
   no <img> without alt, and JSON-LD that parses and includes a BreadcrumbList (home page excepted).

Standard library only (social.py is the one post-render step that needs the project's virtualenv).
"""
from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jsonld  # noqa: E402

SITE = Path(__file__).resolve().parents[1]
REPO = SITE.parent
OUT = Path(os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", REPO / "docs"))
if not OUT.is_absolute():
    OUT = SITE / OUT
BASE = "https://sparvi.io/"
TITLE_MAX, DESC_MIN, DESC_MAX = 60, 120, 155

# Inputs per section beyond the page's own source: its data and the prose numbers it quotes.
SECTION_INPUTS = {
    "baseball": ["site/data/baseball", "site/_generated/baseball.yml"],
    "nfl": ["site/data/nfl", "site/_generated/nfl.yml"],
    # Explainers quote every sport's numbers and chart its data, so any of them changing is a change to the page.
    "learn": ["site/data/baseball", "site/_generated/baseball.yml", "site/data/cfb", "site/_generated/cfb.yml",
              "site/_generated/nfl.yml", "site/_generated/models.yml"],
}


def clean_url(loc: str) -> str:
    return loc[: -len("index.html")] if loc.endswith("/index.html") or loc == BASE + "index.html" else loc


def html_path(loc: str) -> Path:
    rel = loc[len(BASE):]
    return OUT / (rel + "index.html" if rel == "" or rel.endswith("/") else rel)   # idempotent on rerun


def source_for(rel_html: str) -> Path:
    return SITE / (rel_html[: -len(".html")] + ".qmd")


def git_date(paths: list[str]) -> str | None:
    r = subprocess.run(["git", "log", "-1", "--format=%cs", "--", *paths], cwd=REPO,
                       capture_output=True, text=True)
    return r.stdout.strip() or None


def lastmod(rel_html: str) -> str:
    src = source_for(rel_html)
    paths = [src.relative_to(REPO).as_posix()]
    section = rel_html.split("/")[0]
    paths += SECTION_INPUTS.get(section, [])
    if rel_html == "index.html":                      # the home page lists every article
        paths += [p.relative_to(REPO).as_posix() for p in SITE.glob("*/*/index.qmd")]
    return git_date(paths) or git_date(["."]) or "1970-01-01"


def prose_numbers() -> dict[str, str]:
    """Every value in site/_generated/*.yml as 'sport.key' -> string (the files are flat: sport -> key: "value")."""
    out = {}
    for f in sorted((SITE / "_generated").glob("*.yml")):
        sport = None
        for line in f.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith((" ", "#")) and line.endswith(":"):
                sport = line[:-1]
            elif sport and line.startswith("  ") and ": " in line:
                k, v = line.strip().split(": ", 1)
                out[f"{sport}.{k}"] = v.strip().strip('"').replace('\\"', '"')
    return out


CARD_STAT = re.compile(r'(<span class="card-stat-v" data-key="([\w.]+)">)([^<]*)…([^<]*)(</span>)')


def fill_card_stats(h: str, numbers: dict[str, str], problems: list[str], where: str) -> str:
    """Listing cards (site/_listing-cards.ejs) name a prose-number key; put its value in."""
    def sub(m):
        if m[2] not in numbers:
            problems.append(f"{where}: card-key {m[2]} is not a prose number")
            return m[0]
        return f"{m[1]}{html.escape(m[3])}{html.escape(numbers[m[2]])}{html.escape(m[4])}{m[5]}"
    return CARD_STAT.sub(sub, h)


def tag(h: str, pattern: str) -> str | None:
    m = re.search(pattern, h, re.S)
    return html.unescape(m[1].strip()) if m else None


NOINDEX = '<meta name="robots" content="noindex">'
LD_MARK = '<script type="application/ld+json" data-ld="seo">'
SPORTS = {"baseball": "Baseball", "nfl": "American football", "cfb": "American football"}


def h1_of(f: Path, cache: dict) -> str | None:
    if f not in cache:
        cache[f] = tag(f.read_text(encoding="utf-8"), r'<h1 class="title">(.*?)</h1>') if f.exists() else None
        if cache[f]:
            cache[f] = re.sub(r"<[^>]+>", "", cache[f]).strip()
    return cache[f]


def crumbs(url: str, h: str, cache: dict) -> dict | None:
    """Home, then each ancestor that is a page, then this page; None on the home page."""
    rel = url[len(BASE):]
    if not rel or rel == "index.html":
        return None
    parts = rel.rstrip("/").split("/")
    items = [("Sparvi", BASE)]
    for i in range(1, len(parts)):
        anc = "/".join(parts[:i]) + "/"
        name = h1_of(OUT / anc / "index.html", cache)
        if name:
            items.append((name, BASE + anc))
    me = tag(h, r'<h1 class="title">(.*?)</h1>') or tag(h, r"<title>(.*?)</title>") or rel
    items.append((re.sub(r"<[^>]+>", "", me).strip(), url))
    return jsonld.breadcrumbs(items)


def football_teams() -> dict[str, dict]:
    """'nfl/teams/kc' -> its teams.json entry (NFL and CFB)."""
    out = {}
    for sport in ("nfl", "cfb"):
        f = SITE / "data" / sport / "teams.json"
        if f.exists():
            for t in json.loads(f.read_text(encoding="utf-8")):
                out[f"{sport}/teams/{t['slug']}"] = t
    return out


def structured(h: str, url: str, rel: str, modified: str, numbers: dict, teams: dict, cache: dict) -> list[dict]:
    out = []
    bc = crumbs(url, h, cache)
    if bc:
        out.append(bc)
    sport = rel.split("/")[0]
    title = re.sub(r"<[^>]+>", "", tag(h, r'<h1 class="title">(.*?)</h1>') or "").strip()
    desc = tag(h, r'<meta name="description" content="(.*?)"') or ""
    if rel == "index.html":
        out += jsonld.website()
    elif 'name="dcterms.date"' in h:
        image = tag(h, r'<meta property="og:image" content="(.*?)"') or f"{BASE}assets/og-default.png"
        out.append(jsonld.article(url, title, desc, image, tag(h, r'<meta name="dcterms.date" content="(.*?)"'), modified))
    elif "/tools/" in rel and sport in jsonld.SOURCES:
        first, last = numbers.get(f"{sport}.first_season"), numbers.get(f"{sport}.season")
        coverage = f"{first}/{last}" if first and last and first != last else last
        out.append(jsonld.dataset(url, title, desc, sport, coverage, modified))
    elif rel.rsplit("/", 1)[0] in teams:
        t = teams[rel.rsplit("/", 1)[0]]
        out.append(jsonld.sports_team(url, t["name"], SPORTS[sport], t.get("division") or t.get("conf")))
    return out


def set_structured(h: str, objs: list[dict]) -> str:
    h = re.sub(re.escape(LD_MARK) + r".*?</script>\n?", "", h, flags=re.S)       # rerun-safe
    if not objs:
        return h
    graph = objs[0] if len(objs) == 1 else {"@graph": objs}
    block = jsonld.script(graph).replace('<script type="application/ld+json">', LD_MARK, 1)
    return h.replace("</head>", block + "\n</head>", 1)


def ld_problems(h: str, url: str) -> list[str]:
    blocks = re.findall(r'<script type="application/ld\+json"[^>]*>(.*?)</script>', h, re.S)
    types = []
    for b in blocks:
        try:
            d = json.loads(b)
        except json.JSONDecodeError as e:
            return [f"JSON-LD does not parse: {e}"]
        for x in d.get("@graph", [d]):
            types.append(x.get("@type"))
    if url != BASE and "BreadcrumbList" not in types:
        return ["no BreadcrumbList JSON-LD"]
    return []


def check_page(h: str, url: str, titles: dict, descs: dict) -> list[str]:
    """The S1 metadata rules for one page; also records its title and description for the duplicate check."""
    title = tag(h, r"<title>(.*?)</title>") or ""
    desc = tag(h, r'<meta name="description" content="(.*?)"') or ""
    titles[title].append(url)
    descs[desc].append(url)
    p = []
    if not title or len(title) > TITLE_MAX:
        p.append(f"title {len(title)} chars (max {TITLE_MAX}): {title!r}")
    if not DESC_MIN <= len(desc) <= DESC_MAX:
        p.append(f"description {len(desc)} chars (want {DESC_MIN}-{DESC_MAX})")
    n_h1 = len(re.findall(r"<h1[\s>]", h))
    if n_h1 != 1:
        p.append(f"{n_h1} <h1> elements (want 1)")
    if 'property="og:image"' not in h:
        p.append("no og:image")
    if f'<link rel="canonical" href="{url}">' not in h:
        p.append("canonical URL missing or not this page")
    if NOINDEX in h:
        p.append("in a sitemap but marked noindex")
    bad_img = [m for m in re.findall(r"<img\b[^>]*>", h) if not re.search(r"\balt=", m)]
    if bad_img:
        p.append(f"{len(bad_img)} <img> without alt")
    p += ld_problems(h, url)
    return [f"{url}: {x}" for x in p]


def main() -> int:
    if os.environ.get("QUARTO_PROJECT_RENDER_ALL") not in (None, "1"):
        return 0                                       # quarto preview / single-file renders
    sm = OUT / "sitemap.xml"
    locs = re.findall(r"<loc>(.*?)</loc>", sm.read_text(encoding="utf-8"))
    entries, problems = [], []
    titles, descs = defaultdict(list), defaultdict(list)
    numbers = prose_numbers()
    teams, h1s = football_teams(), {}

    for loc in sorted({clean_url(x) for x in locs}):   # partial renders can list a page twice
        f = html_path(loc)
        rel = f.relative_to(OUT).as_posix()
        url = clean_url(loc)
        h = f.read_text(encoding="utf-8")

        # canonical (replace one Quarto may have written, else insert before </head>)
        link = f'<link rel="canonical" href="{url}">'
        h, n = re.subn(r'<link rel="canonical"[^>]*>', link, h)
        if n == 0:
            h = h.replace("</head>", link + "\n</head>", 1)
        h = fill_card_stats(h, numbers, problems, url)
        # decorative listing thumbnails
        h = re.sub(r'<img (?![^>]*\balt=)([^>]*class="thumbnail-image[^"]*")', r'<img alt="" \1', h)
        if NOINDEX in h:                               # redirect pages (the old player.html) stay out of the sitemap
            f.write_text(h, encoding="utf-8", newline="\n")
            continue
        mod = lastmod(rel)
        h = set_structured(h, structured(h, url, rel, mod, numbers, teams, h1s))
        f.write_text(h, encoding="utf-8", newline="\n")
        problems += check_page(h, url, titles, descs)
        entries.append((url, mod))

    # Generated pages (site/build/player_pages.py) come with their own sitemap, canonical URL, and lastmod.
    extra = sorted(x for x in OUT.glob("sitemap-*.xml") if x.name != "sitemap-index.xml")
    n_extra = 0
    for x in extra:
        for loc in re.findall(r"<loc>(.*?)</loc>", x.read_text(encoding="utf-8")):
            f = html_path(loc)
            h = f.read_text(encoding="utf-8")
            bc = crumbs(loc, h, h1s)                    # the generators add their own entity JSON-LD
            h = set_structured(h, [bc] if bc else [])
            f.write_text(h, encoding="utf-8", newline="\n")
            problems += check_page(h, loc, titles, descs)
            n_extra += 1

    problems += [f"duplicate title {t!r}: {', '.join(u)}" for t, u in titles.items() if len(u) > 1]
    problems += [f"duplicate description on: {', '.join(u)}" for d, u in descs.items() if len(u) > 1 and d]

    body = "".join(f"  <url>\n    <loc>{u}</loc>\n    <lastmod>{d}</lastmod>\n  </url>\n" for u, d in entries)
    sm.write_text('<?xml version="1.0" encoding="UTF-8"?>\n'
                  '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + body + "</urlset>\n",
                  encoding="utf-8", newline="\n")
    maps = [("sitemap.xml", max(d for _, d in entries))]
    maps += [(x.name, max(re.findall(r"<lastmod>(.*?)</lastmod>", x.read_text(encoding="utf-8")), default=maps[0][1]))
             for x in extra]
    (OUT / "sitemap-index.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join(f"  <sitemap>\n    <loc>{BASE}{n}</loc>\n    <lastmod>{d}</lastmod>\n  </sitemap>\n" for n, d in maps)
        + "</sitemapindex>\n", encoding="utf-8", newline="\n")
    (OUT / "robots.txt").write_text(f"User-agent: *\nAllow: /\n\nSitemap: {BASE}sitemap-index.xml\n",
                                    encoding="utf-8", newline="\n")
    print(f"seo: {len(entries)} pages + {n_extra} generated, canonical + lastmod set, sitemap-index.xml and robots.txt written")

    if problems:
        print("seo: page metadata problems (see PLAN-seo S1):", file=sys.stderr)
        for x in problems:
            print("  " + x, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
