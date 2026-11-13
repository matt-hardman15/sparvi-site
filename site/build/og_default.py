"""Write site/assets/og-default.png: the 1200x630 share image for the homepage and the section hubs
(articles keep their own thumbnails). Drawn with the site's own fonts in a headless browser.
Rerun only if the wording or palette changes; the PNG is committed.

    python site/build/og_default.py
"""
import base64

from playwright.sync_api import sync_playwright

from sparvi.config import ROOT

FONTS = ROOT / "site" / "assets" / "fonts"
OUT = ROOT / "site" / "assets" / "og-default.png"
def b64(name: str) -> str:
    return base64.b64encode((FONTS / name).read_bytes()).decode()


MARK = '<path d="M8 21c3-1 5-4 6-8 1 3 3 5 6 5l4-6 1 3-4 6c-3 3-9 3-13 0z" fill="#fbfaf7"/>'

HTML = f"""<!doctype html><html><head><style>
@font-face {{ font-family: Serif; src: url("data:font/woff2;base64,{b64('serif-700.woff2')}"); font-weight: 700; }}
@font-face {{ font-family: Inter; src: url("data:font/woff2;base64,{b64('inter.woff2')}"); font-weight: 100 900; }}
body {{ margin: 0; }}
.c {{ width: 1200px; height: 630px; background: #0f5257; color: #fbfaf7; position: relative; overflow: hidden; font-family: Inter; }}
.mark {{ position: absolute; right: -30px; bottom: -60px; opacity: .12; }}
.t {{ position: absolute; left: 96px; top: 170px; }}
.w {{ font-family: Serif; font-weight: 700; font-size: 132px; line-height: 1; letter-spacing: -2px; }}
.g {{ font-size: 46px; margin-top: 34px; font-weight: 500; max-width: 1060px; line-height: 1.2; }}
.u {{ font-size: 30px; margin-top: 40px; opacity: .78; }}
</style></head><body><div class="c">
<svg class="mark" viewBox="7.5 12.5 18 13" width="560" height="404">{MARK}</svg>
<div class="t"><div class="w">Sparvi</div><div class="g">Sports analytics with the data out in the open</div><div class="u">sparvi.io</div></div>
</div></body></html>"""

with sync_playwright() as p:
    try:
        b = p.chromium.launch()
    except Exception:                       # no bundled Chromium: use the installed Chrome
        b = p.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": 1200, "height": 630})
    pg.set_content(HTML)
    pg.evaluate("document.fonts.ready")
    pg.wait_for_timeout(300)
    pg.screenshot(path=str(OUT))
    b.close()
print(f"wrote {OUT.relative_to(ROOT)}")
