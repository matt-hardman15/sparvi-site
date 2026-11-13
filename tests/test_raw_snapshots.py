"""Append-only raw snapshots and as-of reads (A5a), on a local folder standing in for R2."""
import pandas as pd
import pytest

from ingest.common.asof import read_asof
from ingest.common.snapshots import LocalStore, SnapshotExists, Table, day_after, snapshot

T0 = pd.Timestamp("2026-05-01T12:00:00", tz="UTC")     # first snapshot
T1 = pd.Timestamp("2026-05-10T12:00:00", tz="UTC")     # second snapshot (a revision)


def games(rows):
    return pd.DataFrame(rows, columns=["game_pk", "pitch", "game_date", "speed"])


@pytest.fixture()
def setup(tmp_path):
    path = tmp_path / "raw.parquet"
    store = LocalStore(tmp_path / "bucket")
    t = Table("statcast_test", path, ("game_pk", "pitch"), day_after("game_date"))
    games([(1, 1, "2026-04-01", 94.0), (1, 2, "2026-04-01", 95.0), (2, 1, "2026-04-20", 88.0)]).to_parquet(path, index=False)
    s0 = snapshot(store, "baseball", t, T0)
    return store, t, path, s0


def test_first_snapshot_and_no_change(setup):
    store, t, path, s0 = setup
    assert s0["baseline"] and s0["inserts"] == 3 and len(s0["files"]) == 1
    s = snapshot(store, "baseball", t, T0 + pd.Timedelta(days=1))
    assert not s["files"] and (s["inserts"], s["updates"], s["deletes"]) == (0, 0, 0)
    assert len(store.get("snapshots/baseball/statcast_test/manifest.jsonl").decode().splitlines()) == 1


def test_revision_insert_delete_and_asof(setup):
    store, t, path, _ = setup
    # Source revises one pitch's speed, adds a game, and drops a pitch.
    games([(1, 1, "2026-04-01", 94.5), (2, 1, "2026-04-20", 88.0), (3, 1, "2026-05-08", 90.0)]).to_parquet(path, index=False)
    s1 = snapshot(store, "baseball", t, T1)
    assert (s1["inserts"], s1["updates"], s1["deletes"]) == (1, 1, 1)

    before, meta = read_asof(store, "baseball", "statcast_test", pd.Timestamp("2026-05-05"))
    assert meta["revision_safe"]
    assert before.set_index(["game_pk", "pitch"])["speed"].to_dict() == {(1, 1): 94.0, (1, 2): 95.0, (2, 1): 88.0}

    after, _ = read_asof(store, "baseball", "statcast_test", pd.Timestamp("2026-05-11"))
    assert after.set_index(["game_pk", "pitch"])["speed"].to_dict() == {(1, 1): 94.5, (2, 1): 88.0, (3, 1): 90.0}


def test_available_at_hides_games_not_yet_published(setup):
    store, *_ = setup
    df, _ = read_asof(store, "baseball", "statcast_test", pd.Timestamp("2026-05-01T13:00"))
    assert set(df["game_pk"]) == {1, 2}
    df, _ = read_asof(store, "baseball", "statcast_test", pd.Timestamp("2026-04-20T12:00"))   # game 2 is known on 4/21
    assert set(df["game_pk"]) == {1}


def test_history_before_the_first_snapshot_is_not_revision_safe(setup):
    store, *_ = setup
    df, meta = read_asof(store, "baseball", "statcast_test", pd.Timestamp("2026-04-15"))
    assert not meta["revision_safe"] and set(df["game_pk"]) == {1}


def test_snapshot_files_are_never_overwritten(setup):
    store, t, path, _ = setup
    games([(1, 1, "2026-04-01", 99.0), (1, 2, "2026-04-01", 95.0), (2, 1, "2026-04-20", 88.0)]).to_parquet(path, index=False)
    with pytest.raises(SnapshotExists):
        snapshot(store, "baseball", t, T0)       # same stamp as the first snapshot


def test_mass_change_is_refused(tmp_path):
    path, store = tmp_path / "r.parquet", LocalStore(tmp_path / "b")
    t = Table("big", path, ("game_pk", "pitch"))
    games([(g, 1, "2026-04-01", 90.0) for g in range(2000)]).to_parquet(path, index=False)
    snapshot(store, "baseball", t, T0)
    games([(g, 1, "2026-04-01", 91.0 if g < 600 else 90.0) for g in range(2000)]).to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="allow-mass-update"):
        snapshot(store, "baseball", t, T1)            # 600 of 2,000 rows (30%) at once
    assert snapshot(store, "baseball", t, T1, allow_mass_update=True)["updates"] == 600


def test_small_tables_may_change_wholesale(setup):
    store, t, path, _ = setup
    games([(1, 1, "2026-04-01", 1.0), (1, 2, "2026-04-01", 2.0), (2, 1, "2026-04-20", 3.0)]).to_parquet(path, index=False)
    assert snapshot(store, "baseball", t, T1)["updates"] == 3


def test_dropped_columns_are_not_versioned(tmp_path):
    path = tmp_path / "r.parquet"
    store = LocalStore(tmp_path / "b")
    t = Table("rosters", path, ("player_id",), drop=("snapshot_date",))
    pd.DataFrame({"player_id": [1, 2], "snapshot_date": "2026-11-11", "org": ["A", "B"]}).to_parquet(path, index=False)
    snapshot(store, "baseball", t, T0)
    pd.DataFrame({"player_id": [1, 2], "snapshot_date": "2026-11-18", "org": ["A", "B"]}).to_parquet(path, index=False)
    assert not snapshot(store, "baseball", t, T1)["files"]


def test_duplicate_keys_are_refused(tmp_path):
    path = tmp_path / "r.parquet"
    games([(1, 1, "2026-04-01", 94.0), (1, 1, "2026-04-01", 95.0)]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        snapshot(LocalStore(tmp_path / "b"), "baseball", Table("x", path, ("game_pk", "pitch")), T0)
