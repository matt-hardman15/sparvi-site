"""College football matchup tool (CFB-MATCHUP): the arithmetic, the published terms, and plausibility checks that can
fail. tests/test_site.py holds the page's numbers equal to matchup.expected."""
import json
import shutil

import numpy as np

from cfb.adjusted_ratings.matchup import Z80, expected
from plausibility import cfb as suite_cfb
from plausibility.common import Suite
from sparvi.config import ROOT

DATA = ROOT / "site" / "data" / "cfb"
MODEL = {"home_points": 2.5, "margin_sd": 16.0}


def test_margin_is_the_rating_gap_plus_home_field():
    a, b = {"power": 14.0}, {"power": 3.5}
    at_a, at_b, neutral = (expected(a, b, s, MODEL) for s in ("a", "b", "n"))
    assert np.isclose(neutral["margin"], 10.5) and np.isclose(at_a["margin"], 13.0) and np.isclose(at_b["margin"], 8.0)
    assert np.isclose(at_a["hi80"] - at_a["lo80"], 2 * Z80 * 16) and 0.5 < at_b["p_a"] < neutral["p_a"] < at_a["p_a"] < 1
    flipped = expected(b, a, "b", MODEL)
    assert np.isclose(flipped["margin"], -at_a["margin"]) and np.isclose(flipped["p_a"], 1 - at_a["p_a"])
    assert np.isclose(expected(a, a, "n", MODEL)["p_a"], 0.5)


def test_published_spread_is_the_backtests():
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    bt = json.loads((ROOT / "cfb/adjusted_ratings/output/metrics.json").read_text(encoding="utf-8"))["backtest"]
    assert meta["power_model"]["margin_sd"] == round(bt["power"]["rmse"], 2)
    assert meta["power_model"]["home_points"] == json.loads((ROOT / "cfb/adjusted_ratings/output/metrics.json").read_text(encoding="utf-8"))["power_model"]["home_points"]


def test_plausibility_checks_pass_and_fail_on_doctored_terms(tmp_path):
    def run(root):
        s = Suite("cfb", root)
        suite_cfb.matchup_tool(s)
        return {r["name"]: r["ok"] for r in s.results}
    assert len(run(ROOT)) == 3 and all(run(ROOT).values())
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    root = tmp_path / "root"
    (root / "site" / "data" / "cfb").mkdir(parents=True)
    for name in ("teams.json", f"team_week_{meta['season']}.json"):
        shutil.copy(DATA / name, root / "site" / "data" / "cfb" / name)
    wk = json.loads((DATA / f"team_week_{meta['season']}.json").read_text(encoding="utf-8"))
    wk["teams"][0]["series"]["no_gt"]["power"][-1] = 400.0            # a 400-point team
    wk["teams"][1]["id"] = "No Such School"                            # a rated team without a page
    (root / "site" / "data" / "cfb" / f"team_week_{meta['season']}.json").write_text(json.dumps(wk), encoding="utf-8")
    meta["power_model"]["margin_sd"] = 4.0
    (root / "site" / "data" / "cfb" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert not any(run(root).values())
