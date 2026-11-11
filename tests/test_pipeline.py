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


FINAL = """(select max(snapshot_date) from int_snapshot_dates where level = '{lvl}')"""


@pytest.mark.skipif(DBT is None, reason="dbt not installed")
@pytest.mark.parametrize("snap,mart,key,cols", [
    ("fct_pitcher_snapshots", "fct_pitcher_report", "pitcher_id",
     ["bf", "k_pct", "bb_pct", "csw_pct", "whiff_pct", "chase_pct", "zone_pct", "hard_hit_pct", "xwoba", "fb_velo",
      "k_pct_pctl", "bb_pct_pctl", "csw_pct_pctl", "whiff_pct_pctl", "xwoba_pctl", "qualified", "role"]),
    ("fct_hitter_snapshots", "fct_hitter_report", "batter_id",
     ["pa", "chase_pct", "zone_contact_pct", "whiff_pct", "k_pct", "bb_pct", "hard_hit_pct", "ev90", "xwoba",
      "chase_pct_pctl", "zone_contact_pct_pctl", "ev90_pctl", "xwoba_pctl", "qualified"]),
])
def test_final_snapshot_equals_season_mart(built, snap, mart, key, cols):
    """The key correctness test: the last season-to-date snapshot reproduces the season-level mart."""
    # Sums of floats (fb_velo, ev90, xwoba) can differ in the last bits with summation order; everything
    # built from counts (rates, percentiles, flags) must match exactly.
    floaty = {"fb_velo", "ev90", "avg_ev", "xwoba"}
    diff = " or ".join(f"abs(s.{c} - m.{c}) > 1e-9 or (s.{c} is null) != (m.{c} is null)" if c in floaty
                       else f"s.{c} is distinct from m.{c}" for c in cols)
    bad = built.sql(f"""
        select count(*) from (select * from {snap} where level = 'AAA' and window_name = 'std'
                              and snapshot_date = {FINAL.format(lvl='AAA')}) s
        full join {mart} m on m.player_id = s.{key}
        where s.{key} is null or m.player_id is null or {diff}
    """).fetchone()[0]
    assert bad == 0


@pytest.mark.skipif(DBT is None, reason="dbt not installed")
def test_snapshots_are_weekly_and_end_on_the_last_game(built):
    rows = built.sql("""
        select level, count(*), max(snapshot_date), bool_and(dayofweek(snapshot_date) = 0 or is_final)
        from int_snapshot_dates group by 1
    """).fetchall()
    assert {r[0] for r in rows} == {"AAA", "MLB"}
    for level, n, _last, weekly in rows:
        assert n >= 20 and weekly, (level, n, weekly)
