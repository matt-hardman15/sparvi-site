"""NFL matchup tool (NFL-MATCHUP): the arithmetic, the committed site file, and plausibility checks that can fail.
tests/test_site.py holds the page's numbers equal to matchup.expected."""
import json
import shutil

import numpy as np

from nfl.team_efficiency.matchup import Z80, expected, site_data
from plausibility import nfl as suite_nfl
from plausibility.common import Suite
from sparvi.config import ROOT

DATA = ROOT / "site" / "data" / "nfl"
LINE = {"home_points": 2.0, "points_per_net": 240.0, "margin_sd": 13.0}


def test_margin_is_home_field_plus_the_rating_gap():
    a, b = {"net": 0.03}, {"net": -0.01}
    at_a, at_b, neutral = (expected(a, b, s, LINE) for s in ("a", "b", "n"))
    assert np.isclose(neutral["margin"], 240 * 0.04) and np.isclose(at_a["margin"], neutral["margin"] + 2) and np.isclose(at_b["margin"], neutral["margin"] - 2)
    assert np.isclose(at_a["hi80"] - at_a["lo80"], 2 * Z80 * 13) and at_a["lo80"] < at_a["margin"] < at_a["hi80"]
    assert 0.5 < neutral["p_a"] < at_a["p_a"] < 1
    flipped = expected(b, a, "b", LINE)                       # the same game, named the other way
    assert np.isclose(flipped["margin"], -at_a["margin"]) and np.isclose(flipped["p_a"], 1 - at_a["p_a"])
    even = expected(a, a, "n", LINE)
    assert np.isclose(even["margin"], 0) and np.isclose(even["p_a"], 0.5)


def test_site_data_ranks_teams_and_names_them():
    res = {"season": 2026, "through_week": 5, "line": LINE,
           "fit": {"seasons": [2016, 2025], "games": 2160, "mae_out_of_sample": 10.38, "mae_home_only": 11.1},
           "teams": [{"id": "AAA", "net": 0.01, "off": 0.02, "def": 0.01}, {"id": "BBB", "net": 0.03, "off": 0.01, "def": -0.02},
                     {"id": "CCC", "net": -0.02, "off": -0.01, "def": 0.01}]}
    teams = [{"id": k, "name": f"Team {k}"} for k in ("AAA", "BBB", "CCC")]
    data, N = site_data(res, teams)
    assert [t["id"] for t in data["teams"]] == ["AAA", "BBB", "CCC"] and {t["id"]: t["rank"] for t in data["teams"]} == {"BBB": 1, "AAA": 2, "CCC": 3}
    assert N["mu_best"] == "Team BBB" and N["mu_worst"] == "Team CCC" and N["mu_best_vs_worst"] == "12.0" and N["mu_points_per_hundredth"] == "2.4"
    assert N["mu_week"] == "5" and N["mu_seasons"] == "2016–2025"


def test_committed_matchup_file_passes_and_doctored_copies_fail(tmp_path):
    d = json.loads((DATA / "matchup.json").read_text(encoding="utf-8"))
    assert len(d["teams"]) == 32 and d["through_week"] == json.loads((DATA / "meta.json").read_text(encoding="utf-8"))["last_week"]

    def run(root):
        s = Suite("nfl", root)
        suite_nfl.matchup_tool(s)
        return {r["name"]: r["ok"] for r in s.results}
    assert all(run(ROOT).values())
    root = tmp_path / "root"
    (root / "site" / "data").mkdir(parents=True)
    shutil.copytree(DATA, root / "site" / "data" / "nfl", ignore=shutil.ignore_patterns("team_*"))
    p = root / "site" / "data" / "nfl" / "matchup.json"
    d["line"].update(home_points=7.5, margin_sd=4.0)
    d["teams"][0]["net"] += 0.4                               # no longer offense less defense, out of rank order, and a 100-point favorite
    d["teams"].pop()
    d["through_week"] += 1
    p.write_text(json.dumps(d), encoding="utf-8")
    failed = {k for k, ok in run(root).items() if ok is False}
    assert len(failed) == 6, failed
