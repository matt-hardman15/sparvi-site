"""The college football plausibility suite (plausibility/cfb.py): it passes on the committed site data, and each test
fails when the thing it guards is broken in a doctored copy."""
import json

import pytest

import plausibility.cfb as pc
from plausibility.common import Suite
from sparvi.config import ROOT, cfb

SEASON = cfb().season
FILES = ["site/data/cfb", "site/data/status", "site/_generated/cfb.yml", "cfb/adjusted_ratings/output/metrics.json"]


def results(root=ROOT) -> dict[str, dict]:
    s = Suite("cfb", root)
    pc.run(s)
    return {r["name"]: r for r in s.results}


@pytest.fixture()
def root(tmp_path):
    """A copy-on-write view of the files the suite reads: links to the real ones until a test rewrites one."""
    for rel in FILES:
        src, dst = ROOT / rel, tmp_path / rel
        if src.is_dir():
            for f in src.rglob("*"):
                if f.is_file():
                    (tmp_path / f.relative_to(ROOT)).parent.mkdir(parents=True, exist_ok=True)
                    (tmp_path / f.relative_to(ROOT)).symlink_to(f)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src)
    return tmp_path


def edit(root, rel, fn):
    p = root / rel
    d = json.loads(p.read_text(encoding="utf-8"))
    fn(d)
    p.unlink()
    p.write_text(json.dumps(d), encoding="utf-8")


def failed(root) -> set[str]:
    return {n for n, r in results(root).items() if r["ok"] is False}


def test_committed_data_passes():
    res = results()
    bad = {n: r["detail"] for n, r in res.items() if r["ok"] is False}
    assert not bad, bad
    assert len(res) >= 18 and {r["tool"] for r in res.values()} == {pc.RATINGS, pc.GAMES, pc.PBP, pc.PORTAL}


def test_view_matches_the_real_files(root):
    assert not failed(root)


WEEK = f"site/data/cfb/team_week_{SEASON}.json"


def test_rank_out_of_order_fails(root):
    def f(d):
        a, b = d["teams"][0]["series"]["no_gt"]["rank"], d["teams"][1]["series"]["no_gt"]["rank"]
        a[-1], b[-1] = b[-1], a[-1]
    edit(root, WEEK, f)
    assert "power ranks follow the power rating (ranked on the unrounded value; ties share a rank)" in failed(root)


def test_skipped_rank_fails(root):
    def f(d):
        for t in d["teams"]:
            r = t["series"]["no_gt"]["rank"]
            if r[-1] is not None and r[-1] >= 10:
                r[-1] += 1
    edit(root, WEEK, f)
    assert "power ranks follow the power rating (ranked on the unrounded value; ties share a rank)" in failed(root)


def test_uncentered_power_fails(root):
    def f(d):
        for t in d["teams"]:
            p = t["series"]["no_gt"]["power"]
            p[-1] = None if p[-1] is None else p[-1] + 3
    edit(root, WEEK, f)
    assert "power rating is centered on the average FBS team every week" in failed(root)


def test_parts_that_do_not_sum_fail(root):
    def f(d):
        d["fields"] += ["pw_epa", "pw_elo"]
        for t in d["teams"]:
            s = t["series"]["no_gt"]
            s["pw_epa"] = [None if v is None else round(v / 2, 1) for v in s["power"]]
            s["pw_elo"] = [None if v is None else round(v / 2, 1) + 1 for v in s["power"]]
    edit(root, WEEK, f)
    assert "power rating equals its EPA part plus its Elo part" in failed(root)


def test_wrong_or_missing_weights_fail(root):
    edit(root, "site/data/cfb/meta.json", lambda d: d["power_model"].update(points_per_elo=0.05))
    assert "published blend weights are the fitted ones" in failed(root)
    edit(root, "site/data/cfb/meta.json", lambda d: d.pop("power_model"))
    assert "blend weights are published and positive" in failed(root)


def test_an_epa_part_that_is_not_the_efficiency_rating_fails(root):
    """Since the Nov 20 2026 refresh the files carry the two parts themselves (before it the check derived them from
    the published weights, so wrong weights tripped it). A part that still sums to the rating but runs against the
    efficiency rating must fail."""
    def f(d):
        for t in d["teams"]:
            for s in t["series"].values():
                if "pw_epa" in s:
                    epa = [None if v is None else -v for v in s["pw_epa"]]
                    s["pw_elo"] = [None if p is None or e is None else round(p - e, 1) for p, e in zip(s["power"], epa, strict=True)]
                    s["pw_epa"] = epa
    edit(root, WEEK, f)
    bad = failed(root)
    assert "the EPA part of the power rating is the efficiency rating" in bad
    assert "power rating equals its EPA part plus its Elo part" not in bad


def test_trivial_and_oversized_adjustment_fail(root):
    name = "opponent adjustment of offense EPA/play has a non-trivial, bounded size"

    def flat(d):                 # adjusted = raw: no adjustment at all
        for t in d["teams"]:
            s = t["series"]["no_gt"]
            s["off_epa_adj"] = list(s["off_epa"])
    edit(root, WEEK, flat)
    assert name in failed(root)

    def wild(d):                 # an adjustment larger than the thing it adjusts
        for i, t in enumerate(d["teams"]):
            s = t["series"]["no_gt"]
            s["off_epa_adj"] = [None if v is None else v + (0.5 if i % 2 else -0.5) for v in s["off_epa"]]
    edit(root, WEEK, wild)
    assert name in failed(root)


def test_game_in_one_log_only_fails(root):
    def f(d):
        team = next(t for t, rows in d["teams"].items() if rows and rows[0]["opp"] in d["teams"])
        d["teams"][team][0]["opp"] = next(t for t in d["teams"] if t not in (team, d["teams"][team][0]["opp"]))
    edit(root, f"site/data/cfb/team_games_{SEASON}.json", f)
    assert "every FBS-vs-FBS game is in both teams' logs with the same score" in failed(root)


def test_undecided_and_unrepaired_games_fail(root):
    rel = "site/data/status/cfb_pbp_integrity.json"
    edit(root, rel, lambda d: d["games"].pop())
    assert "every score-gap game has a documented decision" in failed(root)
    edit(root, rel, lambda d: d.update(built=True))           # after a build, nothing may still be pending
    assert "every score-gap game has a documented decision" in failed(root)

    def keep_swapped(d):
        for g in d["games"]:
            if g["class"] == "swapped":
                g["decision"] = "keep"
    edit(root, rel, keep_swapped)
    assert "no game with home and away exchanged is used as it came" in failed(root)


def test_portal_window_problems_fail(root):
    edit(root, "site/data/cfb/portal/index.json", lambda d: [w.pop("opens", None) for w in d["windows"] if w["players"] == 0])
    assert "a window with no entrants is labeled with when it opens" in failed(root)
    edit(root, "site/data/cfb/portal/index.json", lambda d: d["windows"][-1].update(players=d["windows"][-1]["players"] + 1))
    assert "entrant counts match the boards, and shares and percentiles are in range" in failed(root)


def test_stray_team_in_the_plays_fails(root, tmp_path, monkeypatch):
    """Build only: a play crediting a team that isn't in its game."""
    import duckdb
    import pandas as pd
    raw = root / "data" / "raw" / "cfb"
    raw.mkdir(parents=True)
    pd.DataFrame({"id": [1], "homeTeam": ["Alpha"], "awayTeam": ["Bravo"]}).to_parquet(raw / "games_2026.parquet")
    db = tmp_path / "f.duckdb"
    con = duckdb.connect(str(db))
    con.sql("create table stg_cfb_plays as select * from (values (1, 'a', 'Alpha', 'Bravo'), (1, 'b', 'Bravo', 'Alpha')) t(game_id, play_id, offense, defense)")
    con.close()
    monkeypatch.setenv("FOOTBALL_DB_PATH", str(db))
    name = "no team's offense or defense is credited in a game it didn't play"
    assert results(root)[name]["ok"] is True
    con = duckdb.connect(str(db))
    con.sql("insert into stg_cfb_plays values (1, 'c', 'Zulu', 'Alpha')")
    con.close()
    assert name in failed(root)
