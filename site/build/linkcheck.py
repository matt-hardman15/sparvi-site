"""Internal link check (PLAN-seo S5). Quarto post-render step, after seo.py. Standard library only.

Every href/src in every page of docs/ that points inside the site must resolve to a file: links, images,
scripts, stylesheets, preloads. A broken one fails the build. It also counts, for every page in a sitemap,
the pages that link to it (crawlable <a href> only; links drawn by JavaScript don't count): a page nothing
links to fails the build, and pages with fewer than 3 are reported (S5 asks for 3; most of those are player
pages reached from the A-Z index and their team page).

External links aren't fetched here (the check.yml lychee step reports them without failing).
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

SITE = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", SITE.parent / "docs"))
if not OUT.is_absolute():
    OUT = SITE / OUT
BASE = "https://sparvi.io/"
ATTR = re.compile(r'<(a|link|script|img|source)\b[^>]*?\s(href|src)="([^"]*)"', re.S)
MIN_IN = 3


def resolve(page: Path, ref: str) -> Path | None:
    """The file an internal reference points to, or None for external/anchor/data references."""
    ref = html_unescape(ref)
    if ref.startswith(BASE):
        ref = "/" + ref[len(BASE):]
    u = urlsplit(ref)
    if u.scheme or u.netloc or not u.path or ref.startswith("#"):
        return None
    path = unquote(u.path)
    target = (OUT / path.lstrip("/")) if path.startswith("/") else (page.parent / path)
    if path.endswith("/"):
        target = target / "index.html"
    return target


def html_unescape(s: str) -> str:
    return s.replace("&amp;", "&")


def norm(p: Path) -> Path:
    return Path(os.path.normpath(p))


def main() -> int:
    if os.environ.get("QUARTO_PROJECT_RENDER_ALL") not in (None, "1"):
        return 0
    pages = sorted(OUT.rglob("*.html"))
    broken: dict[str, list[str]] = {}
    inbound: Counter = Counter()
    for page in pages:
        h = page.read_text(encoding="utf-8")
        seen = set()
        for tag, _, ref in ATTR.findall(h):
            t = resolve(page, ref)
            if t is None:
                continue
            t = norm(t)
            if not t.exists() and not (t.suffix == "" and (t / "index.html").exists()):
                broken.setdefault(ref, []).append(page.relative_to(OUT).as_posix())
            elif tag == "a" and t != norm(page) and t not in seen:
                seen.add(t)
                inbound[t] += 1
    locs = []
    for sm in OUT.glob("sitemap*.xml"):
        if sm.name != "sitemap-index.xml":
            locs += re.findall(r"<loc>(.*?)</loc>", sm.read_text(encoding="utf-8"))
    targets = {loc: norm(resolve(OUT / "index.html", loc) or OUT) for loc in locs}
    orphans = sorted(loc for loc, t in targets.items() if inbound[t] == 0 and loc != BASE)
    thin = sorted(loc for loc, t in targets.items() if 0 < inbound[t] < MIN_IN)
    print(f"linkcheck: {len(pages):,} pages, {sum(len(v) for v in broken.values())} broken internal references, "
          f"{len(orphans)} sitemap pages with no inbound link, {len(thin)} with fewer than {MIN_IN}")
    if thin:
        print(f"linkcheck: fewer than {MIN_IN} inbound links (first 5): {thin[:5]}")
    if broken or orphans:
        for ref, where in sorted(broken.items())[:40]:
            print(f"  broken: {ref} (on {len(where)} pages, e.g. {where[0]})", file=sys.stderr)
        for loc in orphans[:40]:
            print(f"  orphan: {loc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
