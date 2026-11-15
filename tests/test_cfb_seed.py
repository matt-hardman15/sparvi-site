"""The CFB raw-data seed (ingest/cfb/seed.py): upload writes a new dated archive, restore brings back every file
byte for byte from the newest one, and an upload never overwrites an existing object."""
import pytest

import ingest.cfb.seed as S
from ingest.common.snapshots import LocalStore, SnapshotExists


def test_seed_round_trip(tmp_path, monkeypatch):
    src, dst, store_dir = tmp_path / "src", tmp_path / "dst", tmp_path / "r2"
    (src / "cfb" / "cache").mkdir(parents=True)
    (src / "cfb" / "plays_2025.parquet").write_bytes(b"plays")
    (src / "cfb" / "cache" / "x.json.gz").write_bytes(b"\x1f\x8bcache")
    store = LocalStore(store_dir)
    monkeypatch.setattr(S, "store_from_env", lambda: store)
    monkeypatch.setattr(S, "RAW", src)
    assert S.upload() == 0
    keys = store.list(S.PREFIX)
    assert len(keys) == 1 and keys[0].startswith("seeds/cfb/raw-") and keys[0].endswith(".tar.gz")
    with pytest.raises(SnapshotExists):
        store.put_new(keys[0], b"other")
    monkeypatch.setattr(S, "RAW", dst)
    assert S.restore() == 0
    assert (dst / "cfb" / "plays_2025.parquet").read_bytes() == b"plays"
    assert (dst / "cfb" / "cache" / "x.json.gz").read_bytes() == b"\x1f\x8bcache"


def test_restore_without_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "store_from_env", lambda: LocalStore(tmp_path / "r2"))
    monkeypatch.setattr(S, "RAW", tmp_path / "raw")
    assert S.restore() == 1
