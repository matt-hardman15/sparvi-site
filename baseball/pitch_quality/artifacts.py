"""Trained pitch-quality models are stored, not refit (MODEL-ARTIFACTS).

Training the bags takes about an hour and the training seasons don't change during a season, so a refresh that
retrains every day spends that hour producing the model it already had. Here a fitted set of bags is one bundle (a zip
of LightGBM text models), named by a key made from everything training reads:

    key = sha256(set name, training seasons, train.py, features.py, lightgbm version,
                 each training season's raw file, the listed heights of the batters in those seasons)

`get()` returns the bags for a key from, in order: the local store (output/models/store/, gitignored), the private R2
bucket (models/pitch_quality/<set>/<key>.zip; the bucket is append-only, so a key is written once), or a fresh fit.
Whatever the route, the boosters handed back are parsed from the bundle's text, so scoring never depends on whether
the model was fitted in this process. A reloaded booster predicts bit for bit what the fitted one did
(tests/test_training.py).

Official runs (the refresh: SPARVI_MODELS=official) add two rules:
  * a fresh fit is uploaded, and `model_artifacts.json` (committed with the refreshed data) records each set's key and
    the bundle's sha256. The file is written only once the bundle is in the bucket.
  * a bundle whose key equals the recorded key must have the recorded sha256, or the run stops (`ModelMismatch`): the
    refresh never scores with a model other than the one on record. When the key itself is new (model code changed,
    or the season rolled over), the set is fitted again and the record replaced.
  * `verify=True` (the out-of-sample set) fits a second time when a fit was needed and requires an identical bundle:
    the repeatability check, moved here from the refresh so it runs when training runs.

Everywhere else (a laptop, a pull-request job) fitted sets are kept in the local store only and nothing is uploaded or
checked against the record: local raw files differ from CI's, so local keys differ too.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from collections.abc import Callable
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from sparvi.config import RAW

HERE = Path(__file__).resolve().parent
RECORD = HERE / "model_artifacts.json"
STORE = HERE / "output" / "models" / "store"
PREFIX = "models/pitch_quality"
KEY_CODE = ["train.py", "features.py"]


class ModelMismatch(RuntimeError):
    pass


def official() -> bool:
    return os.environ.get("SPARVI_MODELS") == "official"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _text_sha(p: Path) -> str:
    return _sha(p.read_bytes().replace(b"\r\n", b"\n"))    # a Windows checkout and CI must agree


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def inputs(name: str, seasons: list[int], raw: Path = RAW, code_dir: Path = HERE) -> dict:
    """Everything the key is made from, readable (the record keeps it so a changed key can be explained)."""
    files = {f"statcast_mlb_{y}.parquet": _file_sha(raw / f"statcast_mlb_{y}.parquet") for y in sorted(seasons)}
    # Heights are read for every batter, but only the training seasons' batters reach the fit: new call-ups are
    # added to the file all season and must not change the key.
    batters = sorted(set().union(*(pd.read_parquet(raw / f, columns=["batter"])["batter"].dropna().astype(int) for f in files)))
    h = pd.read_parquet(raw / "batter_heights.parquet").set_index("batter")["height_in"]
    heights = h.reindex(batters).astype(float)
    return {"set": name, "seasons": sorted(int(y) for y in seasons), "lightgbm": lgb.__version__,
            "code": {f: _text_sha(code_dir / f) for f in KEY_CODE}, "raw": files,
            "heights": _sha(json.dumps([[int(b), None if pd.isna(v) else float(v)] for b, v in heights.items()]).encode())}


def key_of(inp: dict) -> str:
    return _sha(json.dumps(inp, sort_keys=True).encode())


def bundle(models: dict) -> bytes:
    """{kind: {group: [boosters]}} -> zip bytes. Same models, same bytes: entries sorted, timestamps fixed."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for k in sorted(models):
            for g in sorted(models[k]):
                for f, m in enumerate(models[k][g]):
                    n = m.best_iteration if m.best_iteration and m.best_iteration > 0 else None
                    info = zipfile.ZipInfo(f"{k}_{g}_{f}.txt", date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    z.writestr(info, m.model_to_string(num_iteration=n))
    return buf.getvalue()


def unbundle(data: bytes) -> dict:
    """zip bytes -> {kind: {group: [boosters]}}, each booster holding exactly its best-iteration trees."""
    out: dict = {}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in sorted(z.namelist(), key=lambda s: (s.rsplit("_", 1)[0], int(s.rsplit("_", 1)[1].split(".")[0]))):
            head, f = name[: -len(".txt")].rsplit("_", 1)
            k, g = head.rsplit("_", 1)
            m = lgb.Booster(model_str=z.read(name).decode("utf-8"))
            m.best_iteration = m.num_trees()        # the text holds only the trees to use
            bag = out.setdefault(k, {}).setdefault(g, [])
            assert len(bag) == int(f), name
            bag.append(m)
    return out


def _record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def get(name: str, seasons: list[int], fit: Callable[[], dict], *, verify: bool = False, store="env",
        raw: Path = RAW, local: Path = STORE, record: Path = RECORD, log=print) -> dict:
    """The fitted bags for training set `name`: from the local store, the bucket, or `fit()` (see the module note)."""
    inp = inputs(name, seasons, raw)
    key = key_of(inp)
    path = local / f"{name}-{key[:20]}.zip"
    obj = f"{PREFIX}/{name}/{key}.zip"
    if store == "env":
        from ingest.common.snapshots import store_from_env
        store = store_from_env() if official() else None
    data, source = (path.read_bytes(), "local store") if path.exists() else (None, None)
    if data is None and store is not None:
        data = store.get(obj)
        source = "bucket" if data is not None else None
    in_bucket = source == "bucket"
    if data is None:
        log(f"models[{name}]: no stored fit for key {key[:12]}; training")
        data, source = bundle(fit()), "trained"
        if verify and official():
            log(f"models[{name}]: training again to check it repeats")
            if _sha(bundle(fit())) != _sha(data):
                raise ModelMismatch(f"{name}: two fits on the same data gave different models; training is not repeatable")
    rec = _record(record)
    have = rec.get(name)
    on_record = bool(have) and have["key"] == key and have["sha256"] == _sha(data)     # recorded only once uploaded
    if official() and store is not None and not in_bucket and not on_record:
        from ingest.common.snapshots import SnapshotExists
        try:
            store.put_new(obj, data)
            log(f"models[{name}]: uploaded {obj}")
        except SnapshotExists:                      # written by another run: the stored one is the model of record
            data, source = store.get(obj), "bucket"
        in_bucket = True
    sha = _sha(data)
    if official():
        if have and have["key"] == key and have["sha256"] != sha:
            raise ModelMismatch(f"{name}: the model for key {key[:12]} has sha256 {sha[:12]}, but {record.name} records "
                                f"{have['sha256'][:12]}. The refresh only scores with the model on record.")
        if in_bucket and (not have or have["key"] != key):
            rec[name] = {"key": key, "sha256": sha, "object": obj, "inputs": inp}
            record.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
            log(f"models[{name}]: recorded key {key[:12]} in {record.name}")
    if not path.exists():
        local.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    log(f"models[{name}]: {source}, key {key[:12]}, sha256 {sha[:12]}")
    return unbundle(data)
