"""Build the real dbt project against synthetic data and check the marts the site reads."""
import os
import shutil
import subprocess

import duckdb
import pytest
from synthetic_statcast import write_fixture

from sparvi.config import ROOT, baseball

PIPELINE = ROOT / "baseball" / "pd_pipeline"
DBT = shutil.which("dbt")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pd")
    season = baseball().season
    write_fixture(tmp, season)
    db = tmp / "pd.duckdb"
    env = os.environ | {
        "DBT_PROFILES_DIR": ".",
        "SPARVI_RAW_DIR": tmp.as_posix(),
        "SPARVI_PQ_DIR": tmp.as_posix(),
        "PD_DB_PATH": db.as_posix(),
    }
    as_of = f"{season}-08-28"  # last synthetic game day (150 days from April 1), both levels
    r = subprocess.run(
        [DBT, "build", "--vars", f"{{season: {season}, as_of_date: '{as_of}', as_of_date_mlb: '{as_of}'}}",
         "--target-path", (tmp / "target").as_posix(), "--log-path", (tmp / "logs").as_posix()],
        cwd=PIPELINE, env=env, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    return duckdb.connect(str(db), read_only=True)


@pytest.mark.skipif(DBT is None, reason="dbt not installed")
@pytest.mark.parametrize("table,min_rows", [
    ("fct_hitter_report", 90), ("fct_pitcher_report", 90), ("fct_pitcher_arsenal", 300), ("int_pitcher_games", 1000),
])
def test_marts_have_rows(built, table, min_rows):
    assert built.sql(f"select count(*) from {table}").fetchone()[0] >= min_rows


@pytest.mark.skipif(DBT is None, reason="dbt not installed")
def test_percentiles_oriented_and_bounded(built):
    lo, hi = built.sql("select min(chase_pct_pctl), max(chase_pct_pctl) from fct_hitter_report where qualified").fetchone()
    assert 0 <= lo < 10 and 90 < hi <= 100


@pytest.mark.skipif(DBT is None, reason="dbt not installed")
def test_primary_fastball_is_one_pitch_type(built):
    # The recent velocity must be measured on the same pitch as the season velocity.
    bad = built.sql("""
        select count(*) from fct_pitcher_report r
        where recent_fb_velo is not null and abs(recent_fb_velo - fb_velo) > 4
    """).fetchone()[0]
    assert bad == 0


@pytest.mark.skipif(DBT is None, reason="dbt not installed")
def test_appearances_cover_both_levels(built):
    rows = dict(built.sql("select level, count(distinct season) from int_pitcher_appearances group by 1").fetchall())
    assert rows == {"AAA": 2, "MLB": 3}


@pytest.mark.skipif(DBT is None, reason="dbt not installed")
def test_starter_is_first_pitch_of_the_game(built):
    # Fixture: pitchers 0-4 of each team (id % 100 < 5) start; 5-11 only relieve.
    wrong = built.sql("""
        select count(*) from int_pitcher_appearances
        where started != (pitcher_id % 100 < 5)
    """).fetchone()[0]
    assert wrong == 0


@pytest.mark.skipif(DBT is None, reason="dbt not installed")
def test_roles_follow_the_rotation(built):
    roles = dict(built.sql("""
        select pitcher_id % 100 < 5 as rotation, string_agg(distinct role) from dim_pitcher_role group by 1
    """).fetchall())
    assert roles == {True: "SP", False: "RP"}


@pytest.mark.skipif(DBT is None, reason="dbt not installed")
def test_games_and_starts_add_up(built):
    bad = built.sql("""
        select count(*) from dim_pitcher_role
        where games != starts + relief_apps or start_share != starts / games
    """).fetchone()[0]
    assert bad == 0
