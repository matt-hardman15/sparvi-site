"""College football (F2) on synthetic data: the cfb_* dbt models and their tests (including the score
check), garbage time the same in SQL and Python, the ingest checks, and the snapshot table set."""
import os
import shutil
import subprocess
from datetime import date

import duckdb
import pytest
from synthetic_cfb import TEAMS, write_cfb_fixture

import ingest.cfb.checks as cchecks
import ingest.common.checks as common
from football.common import filters
from sparvi.config import ROOT, cfb, nfl

PROJECT = ROOT / "football" / "dbt"
DBT = shutil.which("dbt")
SEASON = cfb().season


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    d = tmp_path_factory.mktemp("cfb")
    write_cfb_fixture(d, SEASON)
    return d


def dbt_build(raw, db):
    env = os.environ | {"DBT_PROFILES_DIR": ".", "SPARVI_RAW_DIR": raw.as_posix(), "FOOTBALL_DB_PATH": db.as_posix()}
    return subprocess.run(
        [DBT, "build", "--select", "stg_cfb_plays+", "cfb_reconcile_scores", "--vars",
         f"{{nfl_season: {nfl().season}, nfl_first_season: {nfl().season}, cfb_season: {SEASON}, cfb_first_season: {SEASON - 1}}}",
         "--target-path", (raw / "target").as_posix(), "--log-path", (raw / "logs").as_posix()],
        cwd=PROJECT, env=env, capture_output=True, text=True)


@pytest.fixture(scope="module")
def built(raw):
    db = raw / "football.duckdb"
    r = dbt_build(raw, db)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    return duckdb.connect(str(db), read_only=True)


@pytest.mark.skipif(DBT is None, reason="dbt not installed")
def test_efficiency_plays_and_bowl_week(built):
    types = {t for (t,) in built.sql("select distinct is_pass::varchar from stg_cfb_plays").fetchall()}
    assert types == {"true", "false"}
    n_kicks = built.sql("select count(*) from stg_cfb_plays where epa is null").fetchone()[0]
    assert n_kicks == 0
    last = built.sql(f"select max(week) from stg_cfb_plays where season = {SEASON} and season_type = 'regular'").fetchone()[0]
    bowl = built.sql(f"select distinct week from stg_cfb_plays where season = {SEASON} and season_type = 'postseason'").fetchall()
    assert bowl == [(last + 1,)], "the postseason is the week after the last regular-season week"


@pytest.mark.skipif(DBT is None, reason="dbt not installed")
def test_garbage_time_same_in_sql_and_python(built, raw):
    df = built.sql("select qtr, offense, is_garbage, game_id, play_id from stg_cfb_plays").df()
    p = duckdb.sql(f"select id as play_id, gameId as game_id, period, offenseScore - defenseScore as margin "
                   f"from read_parquet('{(raw / 'cfb' / 'plays_*.parquet').as_posix()}')").df()
    m = df.merge(p, on=["game_id", "play_id"])
    assert len(m) == len(df)
    assert (m["is_garbage"].to_numpy() == filters.cfb_garbage_time(m["period"], m["margin"])).all()


@pytest.mark.skipif(DBT is None, reason="dbt not installed")
def test_score_check_catches_a_wrong_result(raw, tmp_path):
    bad = tmp_path / "raw"
    shutil.copytree(raw / "cfb", bad / "cfb")
    g = duckdb.sql(f"select * from read_parquet('{(bad / 'cfb' / f'games_{SEASON}.parquet').as_posix()}')").df()
    g["homePoints"] = g["homePoints"] + 3          # every game's official score now disagrees with the plays
    g.to_parquet(bad / "cfb" / f"games_{SEASON}.parquet", index=False)
    r = dbt_build(bad, tmp_path / "f.duckdb")
    assert r.returncode != 0 and "min_share_true" in r.stdout


def test_ingest_checks_pass_on_the_fixture(raw, monkeypatch):
    monkeypatch.setattr(cchecks, "RAW", raw)
    monkeypatch.setattr(common, "STATUS", raw / "status")
    monkeypatch.setattr(cchecks, "cfb", lambda: type(cfb())(season=SEASON, first_season=SEASON - 1, call_reserve=100, approve_over=300))
    # A tiny league: the minimum row counts and the FBS team-count range describe real data, so those
    # are the only checks expected to fail here.
    cchecks.run(today=date(SEASON + 1, 3, 1))
    st = __import__("json").loads((raw / "status" / "cfb.json").read_text(encoding="utf-8"))
    failed = [c["name"] for c in st["checks"] if not c["ok"]]
    assert failed and all(n.endswith(("row count", "FBS team count")) for n in failed), failed
    assert any("final scores agree" in c["name"] and c["ok"] for c in st["checks"])
    assert st["score_gaps"] == []
    assert len(TEAMS) < 120


def test_snapshot_tables_cover_every_cfb_file():
    from ingest.snapshot import cfb_tables
    names = {t.name for t in cfb_tables()}
    for s in cfb().seasons:
        assert {f"plays_{s}", f"games_{s}", f"teams_{s}", f"rankings_{s}", f"lines_{s}"} <= names
