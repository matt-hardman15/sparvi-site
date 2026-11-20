"""Baseball plausibility tests (plausibility/baseball.py): the committed site data passes, and each kind of implausible
number is caught on a doctored copy: a "+" metric off its stated scale, Triple-A graded like the majors, a Pitching+
that isn't its components, percentiles that don't follow the values, a grade that differs between two tools."""
import json
import shutil

import pytest

from plausibility import baseball as bb
from plausibility.common import Suite
from sparvi.config import ROOT

DATA = ROOT / "site" / "data" / "baseball"


def run(root, tmp_path):
    suite = Suite("baseball", root)
    bb.run(suite)
    ok = suite.finish(write=False, status_dir=tmp_path / "status")
    return ok, {r["name"]: r for r in suite.results}


@pytest.fixture()
def root(tmp_path):
    """A copy of the committed baseball site data (top-level JSON only) the test may rewrite."""
    d = tmp_path / "root" / "site" / "data" / "baseball"
    d.mkdir(parents=True)
    for f in DATA.glob("*.json"):
        shutil.copy(f, d / f.name)
    return tmp_path / "root"


def edit(root, name, fn):
    p = root / "site" / "data" / "baseball" / f"{name}.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    fn(obj)
    p.write_text(json.dumps(obj), encoding="utf-8")


def shift(field, by, window="std", scale=1.0):
    def fn(snap):
        for p in snap["players"]:
            s = p["series"][window]
            s[field] = [None if v is None else round(100 + (v - 100) * scale + by, 1) for v in s[field]]
    return fn


def failed(results):
    return [n for n, r in results.items() if r["ok"] is False]


def test_committed_data_passes_and_writes_nothing(tmp_path):
    ok, results = run(ROOT, tmp_path)
    assert ok, failed(results)
    assert len(results) >= 80 and all(r["ok"] is True for r in results.values())
    assert {r["tool"] for r in results.values()} >= {bb.LB, bb.AAA, bb.R5, bb.PP, bb.FARM, bb.TR, bb.FA, bb.TWO}
    assert not (tmp_path / "status").exists()          # write=False


def test_suite_runs_through_the_module_entry_point(tmp_path, monkeypatch):
    from plausibility import common
    monkeypatch.setattr(common, "STATUS", tmp_path)
    assert common.main(["baseball"]) == 0
    st = json.loads((tmp_path / "baseball_plausibility.json").read_text(encoding="utf-8"))
    assert st["ok"] and st["n_failed"] == 0 and st["n_checks"] == len(st["checks"])
    first = (tmp_path / "baseball_plausibility.json").read_bytes()
    assert common.main(["baseball"]) == 0 and (tmp_path / "baseball_plausibility.json").read_bytes() == first   # rerun-safe


def test_mlb_metric_off_its_scale_fails(root, tmp_path):
    edit(root, "snapshots_grades_mlb", shift("loc_plus", -6))
    ok, results = run(root, tmp_path)
    assert not ok
    assert "MLB league mean of Location+ is 100 on the MLB scale (final date, pitch-weighted)" in failed(results)


def test_triple_a_recentred_to_its_own_average_fails(root, tmp_path):
    """Triple-A grades put on a Triple-A scale (league mean 100) while the page says MLB scale."""
    edit(root, "snapshots_grades_aaa", shift("stuff_plus", 9))
    ok, results = run(root, tmp_path)
    f = failed(results)
    assert not ok and "Triple-A league mean of Stuff+ is below MLB's and inside 75-98 (MLB scale)" in f
    assert "Triple-A league means stay inside their bands at every weekly snapshot" in f


def test_pitching_plus_that_is_not_its_components_fails(root, tmp_path):
    """A Triple-A Pitching+ whose gap to 100 is twice what the model gives: far more than stuff plus location."""
    edit(root, "snapshots_grades_aaa", shift("pitching_plus", 0, scale=1.0))     # unchanged: still passes
    assert run(root, tmp_path)[0]
    # Every pitcher's gap to 100 doubled, so the league's doubles whatever the data. (A fixed 8-point shift stopped
    # tripping the check when Location+ 3.0.0 moved the true ratio from 1.14 to 0.97: 8 points then left it at 1.47.)
    edit(root, "snapshots_grades_aaa", shift("pitching_plus", 0, scale=2.0))
    ok, results = run(root, tmp_path)
    name = "Triple-A Pitching+ shortfall equals the Stuff+ and Location+ shortfalls combined, in runs"
    assert not ok and name in failed(results) and "ratio" in results[name]["detail"]


def test_wrong_sd_fails(root, tmp_path):
    edit(root, "snapshots_grades_mlb", shift("stuff_plus", 0, scale=1.5))
    ok, results = run(root, tmp_path)
    assert not ok and "Qualified MLB pitchers' Stuff+ has an SD near 10 (10 points = 1 SD)" in failed(results)


def test_reversed_and_out_of_range_percentiles_fail(root, tmp_path):
    def flip(snap):
        for p in snap["players"]:
            s = p["series"]["std"]
            s["k_pct_pctl"] = [None if v is None else 100 - v for v in s["k_pct_pctl"]]
    edit(root, "snapshots_pitchers_aaa", flip)
    ok, results = run(root, tmp_path)
    assert not ok and "Percentiles point the right way: higher is better for every column" in failed(results)

    def scramble(snap):
        qualified = [p for p in snap["players"] if p["series"]["std"]["qualified"][-1]]
        a, b = qualified[0]["series"]["std"], qualified[1]["series"]["std"]
        a["xwoba_pctl"][-1], b["xwoba_pctl"][-1] = 150, b["xwoba_pctl"][-1]
    edit(root, "snapshots_hitters_aaa", scramble)
    ok, results = run(root, tmp_path)
    f = failed(results)
    assert "Hitters: percentiles are between 0 and 100 (season to date)" in f
    assert "Hitters: percentiles follow the sorted values among qualified hitters (season to date)" in f


def test_grade_that_differs_between_tools_fails(root, tmp_path):
    def bump(rows):
        rows[0]["stuff_plus"] += 5
    edit(root, "rule5_pitchers", bump)
    ok, results = run(root, tmp_path)
    assert not ok
    assert "Pitchers' Stuff+, Location+, Pitching+ equal their Triple-A leaderboard grades (MLB scale)" in failed(results)


def test_duplicates_decreasing_counts_and_bad_intervals_fail(root, tmp_path):
    edit(root, "stuff_pitchers", lambda rows: rows.append(dict(rows[0])))

    def drop(snap):
        s = next(p for p in snap["players"] if p["series"]["std"]["pitches"][-1])["series"]["std"]
        s["pitches"][-1] = 0
    edit(root, "snapshots_grades_mlb", drop)

    def widen(rows):
        rows[0]["score"] = rows[0]["hi"] + 1
    edit(root, "farm_orgs", widen)
    ok, results = run(root, tmp_path)
    f = failed(results)
    assert not ok and "One row per pitcher and level in the season table" in f
    assert "MLB season-to-date pitches, games, and batters faced never decrease from week to week" in f
    assert "Every organization's score is inside its own interval" in f


def test_missing_player_page_fails(root, tmp_path):
    edit(root, "player_urls", lambda urls: urls.pop(next(iter(urls))))
    ok, results = run(root, tmp_path)
    assert not ok and "Every player listed in a tool has a player page" in failed(results)
