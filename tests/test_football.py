"""Football foundation: the dbt project on synthetic nflverse data, the garbage-time filter, and the
opponent adjustment. Never touches real data."""
import os
import shutil
import subprocess

import duckdb
import numpy as np
import pandas as pd
import pytest
import yaml
from synthetic_nfl import TEAMS, WEEKS, write_nfl_fixture

from football.common import filters
from football.common.adjust import ridge_ratings
from football.common.teams import nfl_team
from sparvi.config import ROOT, nfl

PROJECT = ROOT / "football" / "dbt"
DBT = shutil.which("dbt")
VARS = yaml.safe_load((PROJECT / "dbt_project.yml").read_text(encoding="utf-8"))["vars"]
SEASON = nfl().season


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("nfl")
    write_nfl_fixture(tmp, SEASON)
    return tmp


@pytest.fixture(scope="module")
def built(raw):
    db = raw / "football.duckdb"
    env = os.environ | {"DBT_PROFILES_DIR": ".", "SPARVI_RAW_DIR": raw.as_posix(), "FOOTBALL_DB_PATH": db.as_posix()}
    r = subprocess.run(
        [DBT, "build", "--vars", f"{{nfl_season: {SEASON}, nfl_first_season: {SEASON - 1}}}",
         "--target-path", (raw / "target").as_posix(), "--log-path", (raw / "logs").as_posix()],
        cwd=PROJECT, env=env, capture_output=True, text=True)
    # dbt build runs every schema test too: unique keys, the final week equals the season totals,
    # every team-game is scheduled, and the reconciliation against team stats.
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    return duckdb.connect(str(db), read_only=True)


# ---- definitions agree between Python and SQL -------------------------------------------------
def test_garbage_thresholds_match_dbt():
    assert (VARS["nfl_garbage_qtr"], VARS["nfl_garbage_wp_lo"], VARS["nfl_garbage_wp_hi"]) == (
        filters.NFL_GARBAGE_QTR, filters.NFL_GARBAGE_WP_LO, filters.NFL_GARBAGE_WP_HI)


def test_garbage_time_rule():
    qtr = [4, 4, 4, 3, 4, 4]
    wp = [0.05, 0.95, 0.5, 0.01, np.nan, 0.10]
    assert filters.nfl_garbage_time(qtr, wp).tolist() == [True, True, False, False, False, False]


@pytest.mark.skipif(DBT is None, reason="dbt not installed")
def test_garbage_time_same_in_sql_and_python(built, raw):
    pbp = pd.concat(pd.read_parquet(p) for p in sorted((raw / "nfl").glob("pbp_*.parquet")))
    s = built.sql("select game_id, play_id, is_garbage from stg_nfl_plays").df()
    m = s.merge(pbp[["game_id", "play_id", "qtr", "wp"]], on=["game_id", "play_id"])
    assert len(m) == len(s) and m["is_garbage"].any()
    assert (m["is_garbage"].to_numpy() == filters.nfl_garbage_time(m["qtr"], m["wp"])).all()


@pytest.mark.skipif(DBT is None, reason="dbt not installed")
def test_efficiency_plays_exclude_two_point_kneels_spikes(built, raw):
    pbp = pd.concat(pd.read_parquet(p) for p in sorted((raw / "nfl").glob("pbp_*.parquet")))
    s = built.sql("select game_id, play_id from stg_nfl_plays").df()
    m = s.merge(pbp, on=["game_id", "play_id"])
    assert m[["two_point_attempt", "qb_kneel", "qb_spike"]].sum().sum() == 0
    assert len(m) < len(pbp)


# ---- weekly windows ------------------------------------------------------------------------------
@pytest.mark.skipif(DBT is None, reason="dbt not installed")
def test_bye_week_carries_season_to_date_forward(built):
    w = built.sql(f"""select team, week, win, games, plays, epa_sum from nfl_team_week
                      where season = {SEASON} and side = 'off' and gt = 'all' order by team, win, week""").df()
    assert set(w["team"]) == set(TEAMS) and w["week"].max() == WEEKS
    std = w[w["win"] == "std"].pivot(index="team", columns="week", values="games")
    bye = std.index[std[WEEKS] == std[WEEKS - 1]]
    assert len(bye) == 2, "two teams have a week-7 bye in the fixture"
    for t in bye:
        a = w[(w.team == t) & (w.win == "std") & (w.week == WEEKS)].iloc[0]
        b = w[(w.team == t) & (w.win == "std") & (w.week == WEEKS - 1)].iloc[0]
        assert (a.plays, a.epa_sum) == (b.plays, b.epa_sum)
    l4 = w[(w["win"] == "last4") & (w["week"] == WEEKS)]
    assert l4["games"].max() == VARS["nfl_trailing_weeks"]


@pytest.mark.skipif(DBT is None, reason="dbt not installed")
def test_rates_match_a_direct_computation(built, raw):
    pbp = pd.read_parquet(raw / "nfl" / f"pbp_{SEASON}.parquet")
    keep = ((pbp["pass"] == 1) | (pbp["rush"] == 1)) & pbp["epa"].notna() & (pbp[["two_point_attempt", "qb_kneel", "qb_spike"]].sum(axis=1) == 0)
    direct = pbp[keep].groupby("posteam")["epa"].mean()
    mart = built.sql(f"select team, epa_play from nfl_team_season where season = {SEASON} and side = 'off' and gt = 'all'").df()
    got = mart.set_index("team")["epa_play"]
    assert np.allclose(got.loc[direct.index], direct, atol=1e-12)


# ---- opponent adjustment -------------------------------------------------------------------------
def _plays(schedule, true_off, true_def, n=400, seed=1, noise=1.2):
    rng = np.random.default_rng(seed)
    rows = []
    for off, dff in schedule:
        e = 0.0 + true_off[off] + true_def[dff] + rng.normal(0, noise, n)
        rows.append(pd.DataFrame({"offense": off, "defense": dff, "is_home": rng.random(n) < 0.5, "epa": e}))
    return pd.concat(rows, ignore_index=True)


def test_adjustment_removes_schedule_strength():
    # Team A's offense only faces the two best defenses; B's only the two worst. Equal true offenses.
    teams = list("ABCDEF")
    true_off = dict.fromkeys(teams, 0.0)
    true_def = {"A": 0, "B": 0, "C": -0.15, "D": -0.15, "E": 0.15, "F": 0.15}
    sched = [("A", "C"), ("A", "D"), ("B", "E"), ("B", "F")]
    # Everyone else plays everyone, so the defenses' strength is identifiable.
    sched += [(o, d) for o in "CDEF" for d in teams if o != d]
    sched += [(o, d) for o in "AB" for d in "AB" if o != d] + [("C", "A"), ("D", "B")]
    p = _plays(sched, true_off, true_def, n=3000, noise=0.5)
    r = ridge_ratings(p, lam=50).set_index("team")
    raw_gap = r.loc["B", "off_raw"] - r.loc["A", "off_raw"]
    adj_gap = r.loc["B", "off_adj"] - r.loc["A", "off_adj"]
    assert raw_gap > 0.2            # B looks much better on raw numbers (soft schedule)
    assert abs(adj_gap) < 0.05      # after adjustment they're about equal, as they truly are


def test_balanced_schedule_changes_little():
    teams = list("ABCD")
    true = dict(zip(teams, [0.1, 0.0, -0.05, -0.05], strict=True))
    sched = [(o, d) for o in teams for d in teams if o != d]
    p = _plays(sched, true, dict.fromkeys(teams, 0.0), n=2000)
    r = ridge_ratings(p, lam=200)
    assert (r["off_adj"] - r["off_raw"]).abs().max() < 0.02


def test_ridge_on_cells_equals_play_level_ridge():
    teams = list("ABCD")
    p = _plays([(o, d) for o in teams for d in teams if o != d], dict.fromkeys(teams, 0.0), dict.fromkeys(teams, 0.0), n=50)
    lam = 30.0
    r = ridge_ratings(p, lam)
    k = len(teams)
    X = np.zeros((len(p), 2 * k + 1))
    X[np.arange(len(p)), p["offense"].map({t: i for i, t in enumerate(teams)})] = 1
    X[np.arange(len(p)), k + p["defense"].map({t: i for i, t in enumerate(teams)})] = 1
    X[:, -1] = p["is_home"].astype(float)
    y = p["epa"].to_numpy() - p["epa"].mean()
    beta = np.linalg.solve(X.T @ X + lam * np.eye(2 * k + 1), X.T @ y)
    assert np.allclose(r["off_ridge"], p["epa"].mean() + beta[:k])
    assert np.allclose(r["def_ridge"], p["epa"].mean() + beta[k:2 * k])
    assert np.isclose(r.attrs["home"], beta[-1])


def test_ridge_needs_a_positive_penalty():
    with pytest.raises(ValueError):
        ridge_ratings(pd.DataFrame({"offense": ["A"], "defense": ["B"], "is_home": [True], "epa": [0.1]}), 0)


def test_team_codes_normalize_relocations():
    assert [nfl_team(c) for c in ("OAK", "SD", "STL", "LAR", "KC")] == ["LV", "LAC", "LA", "LA", "KC"]
