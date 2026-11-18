"""Social preview images (PLAN-seo S3): 1200x630 cards drawn with Pillow in the site's fonts and palette.
Quarto post-render step, after player_pages.py and before seo.py. Needs the project virtualenv
(`uv run quarto render`), since Pillow isn't in the standard library.

Who gets a card of their own:
- articles: title and description beside the article's chart thumbnail (thumb.png stays the listing image);
- tools and hubs: title and description;
- baseball players above the S3 bar (500+ MLB pitches, or 150+ Triple-A PA or batters faced) and every
  baseball team page: name, team and level, and 2-3 headline numbers (from _og-manifest.json, written by
  player_pages.py);
- everyone else in baseball: one shared card per level. NFL and CFB team pages keep their preview.png
  charts (already 1200x630 with the team's numbers); the home page, glossary, about, and status keep
  assets/og-default.png.
No logos and no player photos. Cards are build output (docs/), never committed; Pillow output is
byte-identical on reruns. Also writes assets/logo.png (512x512), the Organization logo in the JSON-LD.
"""
from __future__ import annotations

import html
import io
import json
import os
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SITE = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", SITE.parent / "docs"))
if not OUT.is_absolute():
    OUT = SITE / OUT
FONTS = Path(__file__).resolve().parent / "fonts"
BASE = "https://sparvi.io/"
W, H = 1200, 630
TEAL, INK, PAPER, MUTED = (15, 82, 87), (251, 250, 247), (251, 250, 247), (178, 204, 205)
MANIFEST = "_og-manifest.json"
MAX_BYTES = 80_000


def font(name: str, size: int, weight: int | None = None) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(FONTS / f"{name}.ttf"), size)
    if weight is not None:
        try:
            f.set_variation_by_axes([weight])
        except (OSError, ValueError):   # a static font: nothing to set
            pass
    return f


def wrap(draw: ImageDraw.ImageDraw, text: str, f, width: int, max_lines: int) -> list[str]:
    words, lines = text.split(), [""]
    for w in words:
        trial = (lines[-1] + " " + w).strip()
        if draw.textlength(trial, font=f) <= width:
            lines[-1] = trial
        else:
            lines.append(w)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while lines[-1] and draw.textlength(lines[-1] + "…", font=f) > width:
            lines[-1] = lines[-1].rsplit(" ", 1)[0]
        lines[-1] += "…"
    return lines


def fit_title(draw, text: str, width: int, max_lines: int = 2) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    for size in (84, 76, 68, 60, 54, 48):
        f = font("serif-700", size)
        lines = wrap(draw, text, f, width, 99)
        if len(lines) <= max_lines:
            return f, lines
    f = font("serif-700", 48)
    return f, wrap(draw, text, f, width, max_lines)


def card(title: str, kicker: str = "", sub: str = "", stats: list[tuple[str, str]] | None = None,
         thumb: Path | None = None) -> bytes:
    im = Image.new("RGB", (W, H), TEAL)
    d = ImageDraw.Draw(im)
    x, right = 72, (W - 72 if thumb is None else 640)
    d.text((x, 52), "Sparvi", font=font("serif-700", 44), fill=INK)
    d.text((W - 72, 66), "sparvi.io", font=font("inter", 26, 500), fill=MUTED, anchor="ra")
    y = 150
    if kicker:
        d.text((x, y), kicker.upper(), font=font("inter", 26, 700), fill=MUTED)
        y += 52
    tf, lines = fit_title(d, title, right - x, 3 if thumb is not None else 2)
    for ln in lines:
        d.text((x, y), ln, font=tf, fill=INK)
        y += int(tf.size * 1.12)
    if sub:
        sf = font("inter", 32 if not stats else 30, 400)
        bottom = H - 190 if stats else H - 56     # keep clear of the stat boxes and the bottom edge
        for ln in wrap(d, sub, sf, right - x, 3 if not stats else 2):
            if y + 14 + sf.size > bottom:
                break
            y += 6
            d.text((x, y + 8), ln, font=sf, fill=MUTED)
            y += int(sf.size * 1.3)
    if stats:
        bx, by, bw = x, H - 170, (W - 2 * x - 2 * 24) // 3
        for label, value in stats[:3]:
            d.rounded_rectangle((bx, by, bx + bw, by + 120), radius=14, outline=(87, 133, 136), width=2)
            d.text((bx + 24, by + 14), value, font=font("serif-700", 56), fill=INK)
            d.text((bx + 24, by + 82), label, font=font("inter", 24, 500), fill=MUTED)
            bx += bw + 24
    if thumb is not None and thumb.exists():
        t = Image.open(thumb).convert("RGB")
        box = (680, 150, W - 64, H - 64)
        bw, bh = box[2] - box[0], box[3] - box[1]
        d.rounded_rectangle((box[0] - 12, box[1] - 12, box[2] + 12, box[3] + 12), radius=16, fill=PAPER)
        t.thumbnail((bw, bh), Image.Resampling.LANCZOS)
        im.paste(t, (box[0] + (bw - t.width) // 2, box[1] + (bh - t.height) // 2))
    q = im.quantize(128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    buf = io.BytesIO()
    q.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def logo() -> bytes:
    im = Image.new("RGB", (512, 512), TEAL)
    d = ImageDraw.Draw(im)
    d.text((256, 262), "Sparvi", font=font("serif-700", 124), fill=INK, anchor="mm")
    buf = io.BytesIO()
    im.quantize(64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def set_image(h: str, url: str) -> str:
    """Point og:image and twitter:image at url, at 1200x630, as a large-image card."""
    for prop, attr in (("og:image", "property"), ("twitter:image", "name")):
        tag = f'<meta {attr}="{prop}" content="{url}">'
        h, n = re.subn(rf'<meta {attr}="{prop}" content="[^"]*">', tag, h)
        if not n:
            h = h.replace("</head>", tag + "\n</head>", 1)
    for k, v in (("og:image:width", W), ("og:image:height", H)):
        tag = f'<meta property="{k}" content="{v}">'
        h, n = re.subn(rf'<meta property="{k}" content="[^"]*">', tag, h)
        if not n:
            h = h.replace("</head>", tag + "\n</head>", 1)
    tag = '<meta name="twitter:card" content="summary_large_image">'
    h, n = re.subn(r'<meta name="twitter:card" content="[^"]*">', tag, h)
    if not n:
        h = h.replace("</head>", tag + "\n</head>", 1)
    return h


def image_path(page: Path) -> Path:
    """docs/x/index.html -> docs/x/og.png; docs/x/y.html -> docs/x/y.og.png."""
    return page.parent / "og.png" if page.name == "index.html" else page.with_suffix(".og.png")


def url_of(p: Path) -> str:
    return BASE + p.relative_to(OUT).as_posix()


def tag(h: str, pattern: str) -> str:
    m = re.search(pattern, h, re.S)
    return html.unescape(re.sub(r"<[^>]+>", "", m[1]).strip()) if m else ""


SECTION = {"baseball": "Baseball", "nfl": "NFL", "cfb": "College football"}
HUBS = {"baseball/index.html", "nfl/index.html", "cfb/index.html", "baseball/players/index.html",
        "nfl/teams/index.html", "cfb/teams/index.html"}


def quarto_pages() -> list[tuple[Path, str, str, str, Path | None]]:
    """(page, title, kicker, description, thumb) for the Quarto-rendered pages that get their own card."""
    sm = (OUT / "sitemap.xml").read_text(encoding="utf-8")
    out = []
    for loc in sorted(set(re.findall(r"<loc>(.*?)</loc>", sm))):
        rel = loc[len(BASE):]
        rel = rel + "index.html" if rel.endswith("/") or rel == "" else rel
        page = OUT / rel
        if not page.exists():
            continue
        sec = rel.split("/")[0]
        h = page.read_text(encoding="utf-8")
        title = tag(h, r'<h1 class="title">(.*?)</h1>')
        desc = tag(h, r'<meta name="description" content="(.*?)"')
        is_article = 'name="dcterms.date"' in h and (page.parent / "thumb.png").exists()
        if is_article:
            out.append((page, title, f"{SECTION.get(sec, '')} · analysis", desc, page.parent / "thumb.png"))
        elif "/tools/" in rel:
            out.append((page, title, f"{SECTION.get(sec, '')} · tool", desc, None))
        elif rel in HUBS:
            out.append((page, title, SECTION.get(sec, ""), desc, None))
    return out


def main() -> int:
    if os.environ.get("QUARTO_PROJECT_RENDER_ALL") not in (None, "1"):
        return 0
    (OUT / "assets").mkdir(exist_ok=True)
    (OUT / "assets" / "logo.png").write_bytes(logo())
    big, n = [], 0

    def write(img_path: Path, data: bytes):
        nonlocal n
        img_path.write_bytes(data)
        n += 1
        if len(data) > MAX_BYTES:
            big.append(f"{img_path.relative_to(OUT).as_posix()} {len(data) // 1000} KB")

    for page, title, kicker, desc, thumb in quarto_pages():
        p = image_path(page)
        write(p, card(title, kicker, desc, thumb=thumb))
        page.write_text(set_image(page.read_text(encoding="utf-8"), url_of(p)), encoding="utf-8", newline="\n")

    mf = OUT / MANIFEST
    shared: dict[str, str] = {}
    if mf.exists():
        for e in json.loads(mf.read_text(encoding="utf-8")):
            page = OUT / e["page"]
            if e.get("shared"):
                key = e["shared"]
                if key not in shared:
                    p = OUT / "assets" / f"og-{key}.png"
                    write(p, card(e["title"], e.get("kicker", ""), e.get("sub", "")))
                    shared[key] = url_of(p)
                url = shared[key]
            else:
                p = image_path(page)
                write(p, card(e["title"], e.get("kicker", ""), e.get("sub", ""), [tuple(s) for s in e.get("stats", [])]))
                url = url_of(p)
            page.write_text(set_image(page.read_text(encoding="utf-8"), url), encoding="utf-8", newline="\n")
        mf.unlink()
    print(f"social: {n} preview images (1200x630) + logo")
    if big:
        print(f"social: {len(big)} images over {MAX_BYTES // 1000} KB: {big[:5]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
