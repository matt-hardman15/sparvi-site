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
        [DBT, "build", "--select", "int_cfb_game_integrity+", "cfb_reconcile_scores", "--vars",
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
        assert {f"players_{t}_{s}" for t in ("roster", "stats", "usage", "ppa", "wepa", "recruits")} <= names
    assert f"players_portal_{cfb().season + 1}" in names and f"players_recruits_{cfb().season + 1}" in names


def test_player_checks_catch_unlinked_stats(raw, monkeypatch, tmp_path):
    """Season stats whose players aren't on the roster fail the link check."""
    import shutil as sh
    d = tmp_path / "cfb"
    sh.copytree(raw / "cfb", d)
    st = __import__("pandas").read_parquet(d / "players" / f"stats_{SEASON}.parquet")
    st["playerId"] = "x" + st["playerId"]
    st.to_parquet(d / "players" / f"stats_{SEASON}.parquet", index=False)
    monkeypatch.setattr(cchecks, "RAW", tmp_path)
    monkeypatch.setattr(common, "STATUS", tmp_path / "status")
    monkeypatch.setattr(cchecks, "cfb", lambda: type(cfb())(season=SEASON, first_season=SEASON - 1, call_reserve=100, approve_over=300))
    cchecks.run(today=date(SEASON + 1, 3, 1))
    st = __import__("json").loads((tmp_path / "status" / "cfb.json").read_text(encoding="utf-8"))
    bad = [c["name"] for c in st["checks"] if not c["ok"]]
    assert f"player_stats_{SEASON}: players on the roster" in bad


# ---- Play-by-play integrity (int_cfb_game_integrity): swapped, over, and unattributable plays ------------------

def _doctor(raw, tmp_path):
    """Five kinds of bad play-by-play in season SEASON's first games with unequal finals: team labels exchanged, score
    fields exchanged, exchanged with no scoring plays to say which, a running score above the final, a stray team."""
    import pandas as pd
    bad = tmp_path / "raw"
    shutil.copytree(raw / "cfb", bad / "cfb")
    f = bad / "cfb" / f"plays_{SEASON}.parquet"
    p, g = pd.read_parquet(f), pd.read_parquet(bad / "cfb" / f"games_{SEASON}.parquet")
    pick = g[(g["homePoints"] != g["awayPoints"]) & (g["homePoints"] + g["awayPoints"] >= 21)]["id"].tolist()[:5]
    teams, scores, both, over, stray = pick
    tds, nid = [], int(p["id"].astype("int64").max())
    for gid in (teams, scores):                        # touchdowns by the true offense, one per 7 points
        r = g[g["id"] == gid].iloc[0]
        base = p[(p["gameId"] == gid) & (p["playType"] == "Rush")].iloc[0]
        for off, dff, pts in ((r["homeTeam"], r["awayTeam"], r["homePoints"]), (r["awayTeam"], r["homeTeam"], r["awayPoints"])):
            for _ in range(int(pts // 7)):
                nid += 1
                tds.append({**base, "id": str(nid), "playType": "Rushing Touchdown",
                            "offense": off, "defense": dff, "offenseScore": 0.0, "defenseScore": 0.0, "period": 1})
    p = pd.concat([p, pd.DataFrame(tds)], ignore_index=True)
    m = p["gameId"].isin([teams, both])                # labels exchanged; the scores stay with the true offense
    p.loc[m, ["offense", "defense"]] = p.loc[m, ["defense", "offense"]].to_numpy()
    m = p["gameId"] == scores                          # score fields exchanged; labels right
    p.loc[m, ["offenseScore", "defenseScore"]] = p.loc[m, ["defenseScore", "offenseScore"]].to_numpy()
    i = p.index[(p["gameId"] == over) & (p["period"] == 4) & p["ppa"].notna()][0]
    p.loc[i, "offenseScore"] = 150.0
    j = p.index[(p["gameId"] == stray) & p["ppa"].notna() & (p["playType"] == "Rush")][0]
    p.loc[j, "offense"] = "Zulu"
    p.to_parquet(f, index=False)
    return bad, dict(teams=teams, scores=scores, both=both, over=over, stray=stray), (p.loc[i, "id"], p.loc[j, "id"])


@pytest.mark.skipif(DBT is None, reason="dbt not installed")
def test_pbp_integrity_rule_repairs_and_keeps(raw, built, tmp_path):
    bad, G, (over_play, stray_play) = _doctor(raw, tmp_path)
    r = dbt_build(bad, tmp_path / "f.duckdb")
    # Four bad games in a 29-game season fall under the 90% score-agreement bar; nothing else may fail.
    assert "Completed with 1 error" in r.stdout and "min_share_true_cfb_reconcile_scores" in r.stdout, r.stdout[-4000:]
    con = duckdb.connect(str(tmp_path / "f.duckdb"), read_only=True)
    dec = dict(con.sql("select game_id, pbp_class || ':' || decision from int_cfb_game_integrity").fetchall())
    assert dec[G["teams"]] == "swapped:repair_teams" and dec[G["scores"]] == "swapped:repair_scores"
    assert dec[G["both"]] == "swapped:keep_no_score" and dec[G["over"]] == "over:keep"
    cols = "game_id, play_id, offense, defense, is_home, is_garbage"
    good = built.sql(f"select {cols} from stg_cfb_plays where season = {SEASON}").df().set_index("play_id")
    new = con.sql(f"select {cols} from stg_cfb_plays where season = {SEASON}").df().set_index("play_id")
    both = good.merge(new, left_index=True, right_index=True, suffixes=("", "_n"))
    for k in ("teams", "scores"):                       # repaired games: exactly the plays of the clean build
        x = both[both["game_id"] == G[k]]
        assert len(x) and (x["offense"] == x["offense_n"]).all() and (x["is_home"] == x["is_home_n"]).all()
        assert (x["is_garbage"] == x["is_garbage_n"]).all()
    x = both[both["game_id"] == G["both"]]              # undetermined: labels as the feed has them, never garbage time
    assert (x["offense"] == x["defense_n"]).all() and not x["is_garbage_n"].any()
    assert not new.loc[over_play, "is_garbage"]         # impossible running score: no margin
    assert stray_play not in new.index and len(new) > len(good) - 2      # the stray play is dropped, nothing else
    assert con.sql("select count(*) from stg_cfb_plays where offense = 'Zulu' or defense = 'Zulu'").fetchone()[0] == 0
    # the SQL classes are the Python classes
    from ingest.cfb.integrity import classify, swap_decision
    for row in con.sql("""select home_points, away_points, pbp_home, pbp_away, pbp_class, est_home, est_away, decision
                          from int_cfb_game_integrity""").fetchall():
        assert classify(*row[:4]) == row[4]
        if row[4] == "swapped":
            assert swap_decision(row[0], row[1], row[5], row[6]) == row[7]
    con.close()
    # the /status/ table lists each of them with its decision
    from cfb.qa import pbp_integrity as PI
    t = {g["id"]: g for g in PI.from_db(tmp_path / "f.duckdb")["games"]}
    assert t[G["teams"]]["decision"] == "repair_teams" and t[G["over"]]["class"] == "over"


def test_integrity_thresholds_match_dbt():
    import yaml

    from ingest.cfb import integrity
    v = yaml.safe_load((PROJECT / "dbt_project.yml").read_text(encoding="utf-8"))["vars"]
    assert v["cfb_pbp_swap_evidence_points"] == integrity.SWAP_EVIDENCE_POINTS and v["cfb_pbp_big_gap"] == cchecks.BIG_GAP
