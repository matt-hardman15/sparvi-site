"""PERF-1: page scripts that don't block the first paint (no render needed; tests/test_site.py checks pages still work)."""
import importlib.util

from sparvi.config import ROOT


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "site" / "build" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


perf, pp = load("perf"), load("player_pages")


def test_quarto_scripts_get_defer_and_others_are_left_alone():
    h = ('<script src="../site_libs/bootstrap/bootstrap.min.js"></script>'
         '<script src="site_libs/quarto-html/quarto.js" type="module"></script>'
         '<script src="/assets/js/d3.min.js"></script><script>inline()</script>')
    out, n = perf.defer_site_libs(h)
    assert n == 1 and '<script defer src="../site_libs/bootstrap/bootstrap.min.js"></script>' in out
    assert 'type="module"' in out and '<script src="/assets/js/d3.min.js"></script>' in out
    assert perf.defer_site_libs(out)[1] == 0          # rerun-safe


def test_player_pages_defer_chart_libraries_and_team_pages_drop_them():
    shell = "".join(pp.CHART_LIBS)
    player = pp.libs(shell, team=False)
    assert player.count("<script defer src=") == 3 and player.index("d3.min.js") < player.index("plot.umd") < player.index("sparvi.js")
    assert pp.libs(shell, team=True) == ""


def test_quarto_offset_meta_is_restored_without_search():
    h = "<html><head><title>x</title></head></html>"
    assert '<meta name="quarto:offset" content="../../">' in perf.ensure_offset(h, "baseball/stuff-plus/index.html")
    assert '<meta name="quarto:offset" content="./">' in perf.ensure_offset(h, "index.html")
    kept = '<head><meta name="quarto:offset" content="../">'
    assert perf.ensure_offset(kept, "a/b/index.html") == kept
