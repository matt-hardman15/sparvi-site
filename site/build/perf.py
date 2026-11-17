"""Non-blocking page scripts (PERF-1). Quarto post-render step, before seo.py. Standard library only.

Quarto loads its site_libs scripts (navbar, Bootstrap, tooltips, listings, clipboard) in <head> as plain
<script src>, which blocks the first paint: Lighthouse put the cost at ~1.5 s per page on mobile. Their setup
code runs on DOMContentLoaded, and deferred scripts run, in order, before that event, so `defer` keeps
behavior and lets the page paint first. Module scripts are already deferred. The chart libraries (d3, Plot,
sparvi.js) are left alone here: tools draw their main content with them (see site/_partials/metadata.html and,
for player and team pages, site/build/player_pages.py).

With site search off (_quarto.yml), Quarto stops writing <meta name="quarto:offset">, which quarto.js still uses
to find listings.json from article pages (their category links). This step adds it back: the path from the page
to the site root.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", SITE.parent / "docs"))
if not OUT.is_absolute():
    OUT = SITE / OUT
SCRIPT = re.compile(r'<script src="([^"]*site_libs/[^"]+\.js)"></script>')


def defer_site_libs(h: str) -> tuple[str, int]:
    return SCRIPT.subn(r'<script defer src="\1"></script>', h)


def ensure_offset(h: str, rel: str) -> str:
    if 'name="quarto:offset"' in h or "<head>" not in h:
        return h
    offset = "../" * rel.count("/") or "./"
    return h.replace("<head>", f'<head>\n<meta name="quarto:offset" content="{offset}">', 1)


def main() -> int:
    if os.environ.get("QUARTO_PROJECT_RENDER_ALL") not in (None, "1"):
        return 0
    n_pages = n_tags = 0
    for f in OUT.rglob("*.html"):
        h = f.read_text(encoding="utf-8")
        h2, n = defer_site_libs(ensure_offset(h, f.relative_to(OUT).as_posix()))
        if h2 != h:
            f.write_text(h2, encoding="utf-8", newline="\n")
            n_pages += 1
            n_tags += n
    print(f"perf: deferred {n_tags} Quarto scripts on {n_pages} pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
