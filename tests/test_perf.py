"""Load time on a throttled connection (opt-in: SPARVI_PERF=1; timing is too noisy for shared CI runners).

Serves docs/ with gzip like GitHub Pages, applies Chrome DevTools' "Fast 4G" profile with the cache
off, and measures from navigation start until the page's main view has rendered.
    SPARVI_PERF=1 SPARVI_BROWSER_CHANNEL=chrome uv run pytest tests/test_perf.py -s
"""
import functools
import gzip
import http.server
import io
import os
import threading
from pathlib import Path

import pytest

from sparvi.config import ROOT

DOCS = ROOT / "docs"
pytestmark = pytest.mark.skipif(os.environ.get("SPARVI_PERF") != "1" or not (DOCS / "index.html").exists(),
                                reason="opt-in: SPARVI_PERF=1 with docs/ built")
# Chrome DevTools "Fast 4G": 9 Mbps down, 1.5 Mbps up, 60 ms x 2.75 latency, throughput x 0.9.
FAST_4G = {"offline": False, "downloadThroughput": 9e6 / 8 * 0.9, "uploadThroughput": 1.5e6 / 8 * 0.9, "latency": 165}
BUDGET_S = 2.0


class GzipHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_head(self):
        path = Path(self.translate_path(self.path))
        if path.is_dir():
            path = path / "index.html"
        if path.suffix in {".json", ".js", ".css", ".html", ".svg"} and path.exists() \
                and "gzip" in self.headers.get("Accept-Encoding", ""):
            body = gzip.compress(path.read_bytes(), 6)
            self.send_response(200)
            self.send_header("Content-Type", self.guess_type(str(path)))
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return io.BytesIO(body)
        return super().send_head()


@pytest.fixture(scope="module")
def base_url():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(GzipHandler, directory=str(DOCS)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.mark.parametrize("path,ready", [
    ("/baseball/tools/triple-a.html?side=pitchers", "#t table tbody tr"),   # the largest file
    ("/baseball/tools/triple-a.html", "#t table tbody tr"),
    ("/baseball/tools/stuff-leaderboard.html", "#t table tbody tr"),
    ("/baseball/player.html?id=691769", "#root .stat-row"),
])
def test_interactive_within_budget(base_url, path, ready):
    sync = pytest.importorskip("playwright.sync_api")
    with sync.sync_playwright() as pw:
        b = pw.chromium.launch(channel=os.environ.get("SPARVI_BROWSER_CHANNEL") or None)
        times = []
        for _ in range(3):
            ctx = b.new_context()
            page = ctx.new_page()
            page.add_init_script(f"""new MutationObserver((_, o) => {{ if (document.querySelector('{ready}')) {{
                window.__ready = performance.now(); o.disconnect(); }} }}).observe(document, {{childList: true, subtree: true}});""")
            cdp = ctx.new_cdp_session(page)
            cdp.send("Network.enable")
            cdp.send("Network.setCacheDisabled", {"cacheDisabled": True})
            cdp.send("Network.emulateNetworkConditions", FAST_4G)
            page.goto(base_url + path, wait_until="commit")
            page.wait_for_function("window.__ready !== undefined", timeout=20000)
            times.append(page.evaluate("window.__ready") / 1000)
            ctx.close()
        b.close()
    median = sorted(times)[1]
    print(f"{path}: {', '.join(f'{t:.2f}' for t in times)} s (median {median:.2f})")
    assert median < BUDGET_S, f"{path}: median {median:.2f}s on Fast 4G (budget {BUDGET_S}s)"
