"""Ingest data checks (PLAN-scout A0): the synthetic fixtures (shaped by the contracts) pass, and each
kind of bad pull is caught: a dropped column or changed type, lost rows, duplicates, a game without
plays, stale data. Also: a failed run keeps the last accepted row counts, and reruns are byte-identical."""
import importlib.util
import json
import shutil
from datetime import date

import duckdb
import pytest
from synthetic_nfl import TEAMS as NFL_TEAMS
from synthetic_nfl import write_nfl_fixture
from synthetic_statcast import TEAMS as MLB_TEAMS
from synthetic_statcast import write_fixture

import ingest.baseball.checks as bb
import ingest.common.checks as common
import ingest.nfl.checks as nflc
from sparvi.config import ROOT, baseball, nfl

OFF_SEASON = date(baseball().season, 12, 15)


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    write_fixture(d, baseball().season)
    write_nfl_fixture(d, nfl().season)
    return d


@pytest.fixture()
def raw(fixture_dir, tmp_path, monkeypatch):
    """A fresh copy of the fixtures per test (tests rewrite files)."""
    d = tmp_path / "raw"
    shutil.copytree(fixture_dir, d)
    monkeypatch.setattr(bb, "RAW", d)
    monkeypatch.setattr(nflc, "RAW", d)
    monkeypatch.setattr(common, "STATUS", d / "status")
    monkeypatch.setattr(nflc, "nfl", lambda: type(nfl())(season=nfl().season, first_season=nfl().season - 1))
    return d


def run_bb(today=OFF_SEASON):
    return bb.run(today=today, n_orgs=len(MLB_TEAMS))


def run_nfl(today=None):
    return nflc.run(today=today or date(nfl().season + 1, 3, 1), n_teams=len(NFL_TEAMS))


def status(raw, sport):
    return json.loads((raw / "status" / f"{sport}.json").read_text(encoding="utf-8"))


def failed(raw, sport):
    return [c["name"] for c in status(raw, sport)["checks"] if not c["ok"]]


def rewrite(path, sql):
    """Replace a parquet file with `select ... from t` over its current contents."""
    con = duckdb.connect()
    con.sql(f"create table t as select * from read_parquet('{path.as_posix()}')")
    con.sql(f"copy ({sql}) to '{path.as_posix()}' (format parquet)")
    con.close()


def test_fixtures_pass_and_reruns_are_identical(raw):
    assert run_bb(), failed(raw, "baseball")
    assert run_nfl(), failed(raw, "nfl")
    first = {s: (raw / "status" / f"{s}.json").read_bytes() for s in ("baseball", "nfl")}
    run_bb(), run_nfl()                         # second run: compares against the stored counts
    second = {s: (raw / "status" / f"{s}.json").read_bytes() for s in ("baseball", "nfl")}
    run_bb(), run_nfl()
    assert second == {s: (raw / "status" / f"{s}.json").read_bytes() for s in ("baseball", "nfl")}
    assert first.keys() == second.keys()


def test_changed_type_or_missing_column_fails(raw):
    season = baseball().season
    rewrite(raw / f"statcast_mlb_{season}.parquet", "select * exclude (release_speed), release_speed::varchar as release_speed from t")
    rewrite(raw / f"statcast_aaa_{season}.parquet", "select * exclude (pitch_type) from t")
    assert not run_bb()
    assert {f"statcast_mlb_{season}: columns and types", f"statcast_aaa_{season}: columns and types"} <= set(failed(raw, "baseball"))


def test_lost_rows_fail_and_keep_the_baseline(raw):
    assert run_nfl()
    s = nfl().season
    before = status(raw, "nfl")["rows"][f"pbp_{s}"]
    rewrite(raw / "nfl" / f"pbp_{s}.parquet", "select * from t where game_id not in (select min(game_id) from t)")
    assert not run_nfl()
    assert f"pbp_{s}: row count" in failed(raw, "nfl")
    assert "every completed game has play-by-play" in failed(raw, "nfl")
    assert status(raw, "nfl")["rows"][f"pbp_{s}"] == before      # the failed pull didn't lower the baseline


def test_duplicates_fail(raw):
    s = nfl().season
    rewrite(raw / "nfl" / f"pbp_{s}.parquet", "select * from t union all (select * from t limit 5)")
    assert not run_nfl()
    assert f"pbp_{s}: one row per game_id + play_id" in failed(raw, "nfl")


def test_too_many_nulls_fail(raw):
    s = baseball().season
    rewrite(raw / f"statcast_mlb_{s}.parquet",
            "select * exclude (plate_x), case when pitch_number % 5 = 0 then null else plate_x end as plate_x from t")
    assert not run_bb()
    assert f"statcast_mlb_{s}: plate_x missing" in failed(raw, "baseball")


def test_stale_data_fails_in_season(raw):
    s = baseball().season
    lo, hi = baseball().season_windows[s]
    assert not run_bb(today=date.fromisoformat(hi))     # the fixture's season ends long before
    assert f"statcast_mlb_{s}: fresh" in failed(raw, "baseball")


def test_unscored_past_game_fails(raw):
    rewrite(raw / "nfl" / "schedules.parquet",
            "select * exclude (home_score), case when game_id = (select max(game_id) from t) then null else home_score end as home_score from t")
    assert not run_nfl()
    assert "schedule results are current" in failed(raw, "nfl")


def test_nfl_pull_keeps_exactly_the_contract_columns():
    spec = importlib.util.spec_from_file_location("nfl_pull", ROOT / "ingest" / "nfl" / "pull.py")
    pull = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pull)
    assert list(pull.PBP_COLS) == list(common.contract("nfl")["pbp"]["columns"])
