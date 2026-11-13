"""Post-render search step (PLAN-seo S1). Runs after every full `quarto render` (see _quarto.yml).

1. Canonical URLs: every page in the sitemap gets <link rel="canonical"> to its clean URL
   (https://sparvi.io/x/ rather than .../x/index.html), and the sitemap lists the same URLs.
2. lastmod: from the last git commit that touched the page's inputs (its source, plus the sport's
   data and prose numbers for data pages), not the build time, so it changes only when content does.
3. sitemap-index.xml and robots.txt pointing at it.
4. Listing thumbnails get alt="" (decorative: the headline is right next to them).
5. Checks that fail the build: every sitemap page has a title <= 60 characters, a description of
   120-155 characters, both unique across the site, a canonical URL, exactly one <h1>, an og:image,
   and no <img> without alt.

Standard library only: the deploy job runs Quarto without the project's virtualenv.
"""
from __future__ import annotations

import html
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

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


def main() -> int:
    if os.environ.get("QUARTO_PROJECT_RENDER_ALL") not in (None, "1"):
        return 0                                       # quarto preview / single-file renders
    sm = OUT / "sitemap.xml"
    locs = re.findall(r"<loc>(.*?)</loc>", sm.read_text(encoding="utf-8"))
    entries, problems = [], []
    titles, descs = defaultdict(list), defaultdict(list)
    numbers = prose_numbers()

    for loc in sorted(set(locs)):
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
        f.write_text(h, encoding="utf-8", newline="\n")

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
        bad_img = [m for m in re.findall(r"<img\b[^>]*>", h) if not re.search(r"\balt=", m)]
        if bad_img:
            p.append(f"{len(bad_img)} <img> without alt")
        problems += [f"{url}: {x}" for x in p]
        entries.append((url, lastmod(rel)))

    problems += [f"duplicate title {t!r}: {', '.join(u)}" for t, u in titles.items() if len(u) > 1]
    problems += [f"duplicate description on: {', '.join(u)}" for d, u in descs.items() if len(u) > 1 and d]

    body = "".join(f"  <url>\n    <loc>{u}</loc>\n    <lastmod>{d}</lastmod>\n  </url>\n" for u, d in entries)
    sm.write_text('<?xml version="1.0" encoding="UTF-8"?>\n'
                  '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + body + "</urlset>\n",
                  encoding="utf-8", newline="\n")
    newest = max(d for _, d in entries)
    (OUT / "sitemap-index.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"  <sitemap>\n    <loc>{BASE}sitemap.xml</loc>\n    <lastmod>{newest}</lastmod>\n  </sitemap>\n"
        "</sitemapindex>\n", encoding="utf-8", newline="\n")
    (OUT / "robots.txt").write_text(f"User-agent: *\nAllow: /\n\nSitemap: {BASE}sitemap-index.xml\n",
                                    encoding="utf-8", newline="\n")
    print(f"seo: {len(entries)} pages, canonical + lastmod set, sitemap-index.xml and robots.txt written")

    if problems:
        print("seo: page metadata problems (see PLAN-seo S1):", file=sys.stderr)
        for x in problems:
            print("  " + x, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
