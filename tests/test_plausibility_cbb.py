"""The basketball plausibility suite passes on the committed ratings and fails on doctored copies (each test can fail)."""
from __future__ import annotations

import json
import shutil

import pytest

from plausibility import cbb as suite_cbb
from plausibility.common import Suite
from sparvi.config import ROOT

DATA = ROOT / "site" / "data" / "cbb"


def run(root, tmp):
    s = Suite("cbb", root)
    suite_cbb.run(s)
    ok = s.finish(write=False, status_dir=tmp / "status")
    return ok, {r["name"]: r for r in s.results}


@pytest.fixture()
def root(tmp_path):
    d = tmp_path / "root" / "site" / "data" / "cbb"
    d.mkdir(parents=True)
    for f in DATA.glob("*.json"):
        shutil.copy(f, d / f.name)
    return tmp_path / "root"


def edit(root, name, fn):
    p = root / "site" / "data" / "cbb" / f"{name}.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    fn(obj)
    p.write_text(json.dumps(obj), encoding="utf-8")


def failed(results):
    return [n for n, r in results.items() if r["ok"] is False]


def test_committed_ratings_pass(tmp_path):
    ok, results = run(ROOT, tmp_path)
    assert ok, failed(results)
    assert len(results) >= 10 and all(r["ok"] is True for r in results.values())


def test_ratings_that_do_not_add_up_or_rank_fail(root, tmp_path):
    def bump(d):
        d["teams"][0]["net"] -= 30          # the top team: no longer offense plus defense, and ranked above better teams
        d["teams"].append(dict(d["teams"][1]))
    edit(root, "ratings", bump)
    ok, results = run(root, tmp_path)
    f = failed(results)
    assert not ok and "Net rating is offense plus defense" in f and "Rank follows the net rating" in f and "One row per Division I team" in f


def test_team_pages_that_disagree_with_the_ratings_fail(root, tmp_path):
    def break_games(d):
        k = next(k for k, v in d["teams"].items() if sum(g[4] == "final" for g in v) >= 2)
        final = [g for g in d["teams"][k] if g[4] == "final"]
        final[0][5], final[0][6] = final[0][6], final[0][5]        # a win becomes a loss: no longer the published record
        final[1][5] = 400                                          # not a basketball score
        final[1][1] = -5                                           # an opponent with no page
    edit(root, "team_games", break_games)
    edit(root, "ratings", lambda d: d["teams"][3].update(slug=d["teams"][2]["slug"]))
    ok, results = run(root, tmp_path)
    f = failed(results)
    for name in ("Every team has its own page address", "Every team's record is the results listed on its page",
                 "Every final score is a basketball score (10 to 180 points a side)", "Every linked opponent is a team with a page"):
        assert name in f, name
    assert not ok


def test_unshrunk_or_shifted_ratings_and_bad_league_terms_fail(root, tmp_path):
    def blow_up(d):
        for t in d["teams"]:
            t["off"], t["def"] = round(t["off"] * 4 + 10, 1), round(t["def"] * 4 + 10, 1)
            t["net"] = round(t["off"] + t["def"], 1)
            t["tempo"] = 95.0
            t["games"] = t["w"] + t["l"] + 3
    edit(root, "ratings", blow_up)
    edit(root, "meta", lambda m: m.update(home_court_points=9.5, model={"state": "blocked"}, matchup={**m["matchup"], "margin_sd": 40}))
    ok, results = run(root, tmp_path)
    f = failed(results)
    for name in ("The average Division I team's net rating is near zero", "Net ratings spread like team strength does (4 to 25 points per 100 possessions)",
                 "No team is implausibly far from average", "Every team's tempo is between 55 and 82 possessions",
                 "No team has more rated games than games played", "Home court is worth between 1 and 6 points",
                 "The model's version and state are published",
                 "The league terms are basketball-sized (efficiency 90 to 120, tempo 60 to 76, margin spread 7 to 16)"):
        assert name in f, name
    assert not ok
