"""Plausibility tests (plausibility/): the framework, the NFL suite, the known-truth simulation behind the NFL
opponent adjustment, and the publishing gate. The baseball and college football suites have their own files.

Each suite must pass on the committed site data (so a pull request that would publish implausible numbers can't
merge), and each test must fail when the thing it guards is broken in a doctored copy.
"""
import json

import numpy as np
import pytest
import yaml

import plausibility.nfl as pn
from football.common.adjust import cell_cv, pool_cv, ridge_ratings
from football.common.simulate import recovery, season, true_schedule
from plausibility.common import SPORTS, Suite, main, run_sport
from sparvi.config import ROOT, nfl

SEASON = nfl().season
FILES = ["site/data/nfl"]


def results(root=ROOT) -> dict[str, dict]:
    s = Suite("nfl", root)
    pn.efficiency(s); pn.game_logs(s); pn.fourth_downs(s); pn.matchups(s)
    return {r["name"]: r for r in s.results}


def failed(root) -> list[str]:
    return [n for n, r in results(root).items() if r["ok"] is False]


@pytest.fixture()
def root(tmp_path):
    """The files the NFL suite reads, as links to the real ones until a test rewrites one."""
    for f in (ROOT / "site/data/nfl").glob("*.json"):
        (tmp_path / "site/data/nfl").mkdir(parents=True, exist_ok=True)
        (tmp_path / "site/data/nfl" / f.name).symlink_to(f)
    return tmp_path


def edit(root, rel, fn):
    p = root / rel
    d = json.loads(p.read_text(encoding="utf-8"))
    fn(d)
    p.unlink()
    p.write_text(json.dumps(d), encoding="utf-8")


# ---- the framework ---------------------------------------------------------------------------------------------

def test_every_suite_passes_on_the_committed_site_data(tmp_path):
    """The publishing gate for pull requests: what is committed is what the site shows."""
    for sport in SPORTS:
        assert run_sport(sport, write=False), f"{sport}: see the FAILED lines above"


def test_committed_status_files_match_the_committed_data(tmp_path):
    """/status/ shows site/data/status/<sport>_plausibility.json: it must be what the suites say about the committed
    data (tests that need a full rebuild keep the result the last rebuild recorded)."""
    for sport in SPORTS:
        committed = ROOT / "site/data/status" / f"{sport}_plausibility.json"
        (tmp_path / committed.name).write_text(committed.read_text(encoding="utf-8"), encoding="utf-8")
        run_sport(sport, write=True, status_dir=tmp_path)
        fresh, old = (json.loads(p.read_text(encoding="utf-8")) for p in (tmp_path / committed.name, committed))
        # A test that can't run here says which inputs are missing, which differs by machine: compare the rest.
        ran = [[c for c in d["checks"] if not c["detail"].startswith("not run here")] for d in (fresh, old)]
        assert ran[0] == ran[1], f"{sport}: rerun `python -m plausibility {sport}` and commit the status file"
        assert old["ok"] is True


def test_a_failure_fails_the_run_and_is_written(tmp_path):
    s = Suite("nfl")
    s.check("Tool", "passes", True, "fine")
    s.check("Tool", "fails", False, "42 is not 100")
    assert s.finish(status_dir=tmp_path) is False
    d = json.loads((tmp_path / "nfl_plausibility.json").read_text(encoding="utf-8"))
    assert (d["ok"], d["n_checks"], d["n_failed"]) == (False, 2, 1)
    assert [c["ok"] for c in d["checks"]] == [True, False]


def test_a_test_that_cannot_run_here_keeps_its_last_result(tmp_path):
    """Pull-request CI has no dbt database: a build-only test must not erase (or forge) what a refresh recorded."""
    first = Suite("nfl")
    first.check("Tool", "needs the build", False, "3 plays credited to a team that didn't play")
    assert first.finish(status_dir=tmp_path) is False
    later = Suite("nfl", tmp_path)
    assert later.needs("Tool", "needs the build", tmp_path / "football.duckdb") is False
    assert later.finish(status_dir=tmp_path) is False          # still failing: the last real result stands
    never = Suite("cfb", tmp_path)
    never.needs("Tool", "needs the build", tmp_path / "football.duckdb")
    assert never.finish(status_dir=tmp_path) is True           # never run: neither a pass nor a failure
    assert json.loads((tmp_path / "cfb_plausibility.json").read_text(encoding="utf-8"))["n_not_run"] == 1


def test_check_mode_writes_nothing(tmp_path, monkeypatch):
    import plausibility.common as common
    monkeypatch.setattr(common, "STATUS", tmp_path)
    assert main(["nfl", "--check"]) == 0 and not list(tmp_path.iterdir())
    assert main(["nfl"]) == 0 and (tmp_path / "nfl_plausibility.json").exists()
    first = (tmp_path / "nfl_plausibility.json").read_bytes()
    main(["nfl"])
    assert (tmp_path / "nfl_plausibility.json").read_bytes() == first      # no clock times: reruns are identical


def test_ranks_match_values():
    rows = [{"v": 3.0, "r": 1}, {"v": 2.0, "r": 2}, {"v": 2.0, "r": 2}, {"v": 1.0, "r": 4}]
    assert Suite("nfl").ranks_match("T", "competition ranking", rows, "r", "v")
    assert not Suite("nfl").ranks_match("T", "dense ranks", [*rows[:3], {"v": 1.0, "r": 3}], "r", "v")
    assert not Suite("nfl").ranks_match("T", "a tie split", [{"v": 2.0, "r": 1}, {"v": 2.0, "r": 2}], "r", "v")
    assert not Suite("nfl").ranks_match("T", "out of order", [{"v": 1.0, "r": 1}, {"v": 2.0, "r": 2}], "r", "v")
    assert Suite("nfl").ranks_match("T", "lower is better", [{"v": 1.0, "r": 1}, {"v": 2.0, "r": 2}], "r", "v", descending=False)
    # Rounded values, or part of a ranked list: only the order can be tested.
    part = [{"v": 0.21, "r": 3}, {"v": 0.21, "r": 4}, {"v": 0.05, "r": 17}]
    assert Suite("nfl").ranks_match("T", "part of a list", part, "r", "v", exact=False)
    assert not Suite("nfl").ranks_match("T", "part, out of order", [{"v": 0.21, "r": 9}, {"v": 0.05, "r": 4}], "r", "v", exact=False)
    assert not Suite("nfl").ranks_match("T", "nothing to rank", [], "r", "v")


def test_league_mean_on_its_stated_scale():
    assert Suite("baseball").mean_near("T", "100 on the MLB scale", [90, 110, 100], 100, 2)
    assert Suite("baseball").mean_near("T", "weighted", [90, 110], 100, 2, weights=[1, 1])
    assert not Suite("baseball").mean_near("T", "Triple-A on the MLB scale is not 100", [91, 89, 85], 100, 2)
    assert not Suite("baseball").mean_near("T", "no values", [], 100, 2)


# ---- the NFL suite ---------------------------------------------------------------------------------------------

def test_nfl_committed_data_passes():
    bad = {n: r["detail"] for n, r in results().items() if r["ok"] is not True}
    assert not bad, bad


def _scale_adjustment(d, k):
    """Move every adjusted number to raw + k * (adjusted - raw)."""
    for t in d["teams"]:
        for s in t["series"].values():
            for side in ("off", "def"):
                s[f"{side}_epa_adj"] = [None if a is None else r + k * (a - r)
                                        for a, r in zip(s[f"{side}_epa_adj"], s[f"{side}_epa"], strict=True)]


def test_an_adjustment_that_barely_moves_anyone_fails(root):
    """The Nov 2026 defect: corrections a fifth of their proper size (largest 0.008 EPA/play) passed every check."""
    edit(root, f"site/data/nfl/team_week_{SEASON}.json", lambda d: _scale_adjustment(d, 0.2))
    assert failed(root) == ["opponent adjustment has a non-trivial, bounded size"]


def test_an_oversized_adjustment_fails(root):
    edit(root, f"site/data/nfl/team_week_{SEASON}.json", lambda d: _scale_adjustment(d, 6))
    assert "opponent adjustment has a non-trivial, bounded size" in failed(root)


def test_a_correction_that_shifts_the_league_fails(root):
    def shift(d):
        for t in d["teams"]:
            for s in t["series"].values():
                s["off_epa_adj"] = [None if a is None else a + 0.03 for a in s["off_epa_adj"]]
    edit(root, f"site/data/nfl/team_week_{SEASON}.json", shift)
    assert failed(root) == ["the correction averages zero across teams"]


def test_a_correction_with_the_forecast_penalty_or_no_simulation_fails(root):
    def old(d):
        d["adjustment"].pop("lambda_schedule"); d["adjustment"].pop("simulation_recovered")
    edit(root, "site/data/nfl/meta.json", old)
    assert failed(root) == ["the schedule correction has its own penalty, lighter than the forecast penalty",
                            "in simulation the correction recovers most of a known schedule effect"]


def test_offense_credited_without_a_defense_fails(root):
    def drop(d):
        s = d["teams"][0]["series"]["no_gt"]
        s["off_plays"] = [n + 40 if n else n for n in s["off_plays"]]
    edit(root, f"site/data/nfl/team_week_{SEASON}.json", drop)
    assert failed(root) == ["league offense and defense totals agree (each play counted once on each side)"]


def test_a_game_only_one_team_played_fails(root):
    def swap(d):
        team, other = sorted(d["teams"])[:2]
        row = d["teams"][team][0]
        row["opp"] = other if row["opp"] != other else sorted(d["teams"])[2]       # an opponent that played someone else
    edit(root, f"site/data/nfl/team_games_{SEASON}.json", swap)
    assert failed(root) == ["every game is in both teams' logs, mirrored (scores, home team, offense = opponent's defense)"]


def test_offense_and_defense_exchanged_in_a_game_fails(root):
    def flip(d):
        row = d["teams"][sorted(d["teams"])[0]][0]
        for k in ("epa", "plays", "ng_epa", "ng_plays"):
            row[f"off_{k}"], row[f"def_{k}"] = row[f"def_{k}"], row[f"off_{k}"]
    edit(root, f"site/data/nfl/team_games_{SEASON}.json", flip)
    assert failed(root) == ["every game is in both teams' logs, mirrored (scores, home team, offense = opponent's defense)"]


@pytest.mark.parametrize("break_it", [
    lambda g: g.update(t=g["y"] + 1),                              # 4th and 12 from the 11
    lambda g: g.update(go=1.2),                                    # a probability above 1
    lambda g: g.update(call="kick" if g["call"] == "go" else "go"),    # a call its interval doesn't support
    lambda g: g.update(m=g["m"] + 0.05),                           # margin that isn't go minus the best kick
    lambda g: g.update(kick="punt" if g["kick"] == "fg" else "fg"),    # the worse (or an unavailable) kick named best
])
def test_an_impossible_fourth_down_decision_fails(root, break_it):
    def doctor(d):
        break_it(next(g for g in d["grid"] if g["call"] != "toss-up" and g["fg"] is not None))
    edit(root, "site/data/nfl/fourth_downs.json", doctor)
    assert "no impossible decision in the grid (distance, options offered, probabilities, call vs. interval)" in failed(root)


def test_a_field_goal_from_midfield_or_a_punt_at_the_goal_line_fails(root):
    name = "no impossible decision in the grid (distance, options offered, probabilities, call vs. interval)"
    edit(root, "site/data/nfl/fourth_downs.json", lambda d: next(g for g in d["grid"] if g["y"] == 60).update(fg=0.4))
    assert name in failed(root)


def test_a_longer_kick_that_is_likelier_fails(root):
    def bend(d):
        for g in d["grid"]:
            if g["y"] == 30 and g["fg"] is not None:
                g["pf"] = 0.99
    edit(root, "site/data/nfl/fourth_downs.json", bend)
    assert failed(root) == ["longer kicks and longer conversions are never likelier"]


def test_team_fourth_down_totals_that_do_not_add_up_fail(root):
    edit(root, "site/data/nfl/fourth_downs.json", lambda d: d["teams"][0].update(clear_go=d["teams"][0]["clear_go"] + 1))
    assert failed(root) == ["team totals are coherent and add up to the league's"]


def test_a_rank_that_does_not_follow_its_value_fails(root):
    def worst_first(d):
        units = [g[u]["all"] for g in d["games"] for u in ("away_off", "home_off")]
        best = max(units, key=lambda m: m["off"])
        best["off_rank"] = 33
    edit(root, "site/data/nfl/offense_defense.json", worst_first)
    assert failed(root) == ["unit ranks follow the adjusted EPA/play shown"]


def test_a_play_credited_to_a_team_that_did_not_play_fails(tmp_path, monkeypatch):
    """Build-only test: plays against the schedule. Not run without the dbt database; fails on a stray team."""
    import duckdb
    import pandas as pd
    s = Suite("nfl", tmp_path)
    pn.attribution(s)
    assert s.results[0]["ok"] is None and "not run here" in s.results[0]["detail"]

    (tmp_path / "data/raw/nfl").mkdir(parents=True)
    pd.DataFrame({"game_id": ["g1", "g2"], "home_team": ["A", "C"], "away_team": ["B", "D"]}).to_parquet(
        tmp_path / "data/raw/nfl/schedules.parquet")
    db = tmp_path / "football.duckdb"
    monkeypatch.setenv("FOOTBALL_DB_PATH", str(db))

    def run(rows):
        db.unlink(missing_ok=True)
        con = duckdb.connect(str(db))
        con.execute("create table stg_nfl_plays (game_id varchar, offense varchar, defense varchar)")
        con.executemany("insert into stg_nfl_plays values (?, ?, ?)", rows)
        con.close()
        s = Suite("nfl", tmp_path)
        pn.attribution(s)
        return s.results[0]["ok"]

    good = [("g1", "A", "B"), ("g1", "B", "A"), ("g2", "D", "C")]
    assert run(good) is True
    assert run([*good, ("g2", "A", "C")]) is False       # A's offense in a game between C and D
    assert run([*good, ("g1", "A", "A")]) is False       # a team against itself
    assert run([*good, ("g9", "A", "B")]) is False       # a game that isn't on the schedule


# ---- the opponent adjustment against known truth -------------------------------------------------------------------

def test_the_schedule_correction_recovers_most_of_a_known_schedule_effect():
    """NFL-SCHED sanity test: true strengths are known and schedules are unequal (division rivals twice, most other
    teams never). At the published penalty the correction must recover most of each team's true schedule effect over
    a full season; at the forecast penalty it replaced (3,200) it does not, which is the defect this guards."""
    published = json.loads((ROOT / "nfl/team_efficiency/output/adjustment.json").read_text(encoding="utf-8"))
    lam, lam_s = published["lambda"], tuple(published["lambda_schedule"])
    new = recovery(lam, lam_s, seasons=12, seed=3)
    old = recovery(lam, None, seasons=12, seed=3)
    assert new["recovered"] > 0.6 and 0.5 < new["slope"] < 1.1, new
    assert old["recovered"] < 0.5 and old["slope"] < 0.35, old
    # ...and correcting helps: adjusted numbers are closer to the truth than raw ones, and closer than with 3,200.
    assert new["rmse_adjusted"] < old["rmse_adjusted"] < new["rmse_raw"]
    # More unequal schedules (strong and weak divisions) are still recovered.
    assert recovery(lam, lam_s, seasons=12, seed=3, strength_by_division=0.5)["recovered"] > 0.6
    # Early in the season there is less to go on: some is recovered, not most of it, and never negative.
    assert 0.2 < recovery(lam, lam_s, seasons=12, seed=3, through_week=5)["recovered"] < new["recovered"]


def test_with_no_noise_and_no_penalty_the_correction_is_exact():
    """The estimator itself is right: with noiseless plays and almost no shrinkage it returns the true schedule."""
    plays, truth = season(np.random.default_rng(5), sd_play=1e-9)
    r = ridge_ratings(plays, 3200, lam_schedule=1e-6).set_index("team")
    ts = true_schedule(plays, truth).set_index("team").reindex(r.index)
    for side in ("off", "def"):
        est, tru = r[f"{side}_sched"] - r[f"{side}_sched"].mean(), ts[f"{side}_sched"] - ts[f"{side}_sched"].mean()
        assert np.abs(est - tru).max() < 1e-4
        true_strength = truth.set_index("team").reindex(r.index)[side]
        assert np.abs((r[f"{side}_adj"] - r[f"{side}_adj"].mean()) - (true_strength - true_strength.mean())).max() < 1e-4


def test_the_schedule_penalty_leaves_the_forecast_ratings_alone():
    plays, _ = season(np.random.default_rng(1))
    one, two = ridge_ratings(plays, 3200), ridge_ratings(plays, 3200, lam_schedule=400)
    for c in ("off_ridge", "def_ridge", "off_raw", "def_raw"):
        assert (one[c] == two[c]).all()
    assert (two["off_adj"] - two["off_raw"]).std() > 2 * (one["off_adj"] - one["off_raw"]).std()
    pair = ridge_ratings(plays, 3200, lam_schedule=(300, 800))
    assert not np.allclose(pair["off_adj"], two["off_adj"])


def test_held_out_games_pick_a_penalty_near_the_truth():
    """`cell_cv` on simulated seasons prefers a penalty near noise variance / team variance (about 350 plays here),
    not the forecast penalty, in both the leave-one-game-out and the forward form."""
    rng = np.random.default_rng(11)
    grid = [50, 400, 3200]
    seasons = [season(rng)[0] for _ in range(4)]
    logo = pool_cv([cell_cv(p, grid) for p in seasons])
    fwd = pool_cv([cell_cv(p, grid, forward_from=3) for p in seasons])
    assert max(logo, key=lambda r: r["skill"])["lambda"] == 400
    assert max(fwd, key=lambda r: r["skill"])["lambda"] == 400
    assert all(0 < r["skill"] < 0.5 for r in logo)


# ---- the publishing gate ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("wf, sport", [("nfl.yml", "nfl"), ("cfb.yml", "cfb"), ("refresh-baseball.yml", "baseball")])
def test_refresh_publishes_only_what_passes_plausibility(wf, sport):
    """A failing plausibility test blocks publishing the way a failing data check does: the suite runs after the
    build, the rebuilt data is committed only if it passed, its status file is committed either way, and the run
    ends red."""
    steps = yaml.safe_load((ROOT / ".github/workflows" / wf).read_text(encoding="utf-8"))["jobs"]["refresh"]["steps"]
    names = [s.get("name") for s in steps]
    by = {s.get("name"): s for s in steps}
    test, commit = by["Plausibility tests"], by["Commit refreshed data"]
    build = next(n for n in names if n and n.startswith("Rebuild"))
    assert names.index(build) < names.index("Plausibility tests") < names.index("Commit refreshed data")
    assert f"python -m plausibility {sport}" in test["run"] and "steps.build.outcome == 'success'" in test["if"]
    run = commit["run"]
    gate = run.index('steps.plausibility.outputs.ok }}" = "true"')
    assert gate < run.index(f"git add site/data/{sport}")                       # data: only inside the gate
    assert run.index(f"site/data/status/{sport}_plausibility.json") < gate      # status: before it, pass or fail
    fail = by["Fail the run if a plausibility test failed"]
    assert fail["if"] == "steps.plausibility.outputs.ok == 'false'" and "exit 1" in fail["run"]


def test_deploy_and_pull_requests_run_the_suites():
    deploy = yaml.safe_load((ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8"))["jobs"]["build"]["steps"]
    names = [s.get("name") for s in deploy]
    assert names.index("Plausibility tests on the committed site data") < names.index("Render")
    assert "python -m plausibility all --check" in deploy[names.index("Plausibility tests on the committed site data")]["run"]
    check = (ROOT / ".github/workflows/check.yml").read_text(encoding="utf-8")
    for f in ("tests/test_plausibility.py", "tests/test_plausibility_baseball.py", "tests/test_plausibility_cfb.py"):
        assert f in check
