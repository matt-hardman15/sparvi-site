"""The forecast ledger (PLAN-scout §4, L1a): one append-only JSONL file per sport and season.

    forecasts/<sport>/<season>.jsonl            one line per forecast, never changed
    forecasts/<sport>/<season>.resolved.jsonl   one line per resolution: scored, or void with a reason

Integrity rules, and where each is enforced:
- Only GitHub Actions writes, on main (`append_forecasts` / `append_resolutions` refuse anywhere else),
  so the push time on GitHub is the timestamp of record. The writer stamps `created_at` itself: a
  forecast can't be backdated, and it must lock before it can resolve.
- No model output enters without its publish gate (`backtest.gate.require_passing`); the line records
  whether the model was validated or experimental, and the exact version.
- Append-only: `check` compares every ledger file against a base commit and fails if any existing byte
  changed or a file disappeared. CI runs it on every pull request (where it also refuses new lines: they
  only arrive by an Actions push) and on every push to main that touches the ledger.
- Mistakes are voided with a reason, never edited. Each forecast resolves at most once.

    python -m sparvi.ledger check --base origin/main --pr     # pull requests
    python -m sparvi.ledger check --base <sha>                # after an Actions write, before/after a push
    python -m sparvi.ledger smoke                             # the full write path, in a scratch copy
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

from sparvi.config import ROOT

LEDGER = ROOT / "forecasts"
ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,159}$")
SPORT = re.compile(r"^[a-z][a-z0-9-]*$")
FILE = re.compile(r"^(?P<sport>[a-z][a-z0-9-]*)/(?P<season>\d{4})(?P<res>\.resolved)?\.jsonl$")
STAMP = "%Y-%m-%dT%H:%M:%SZ"

# Field order is the line's key order (lines are written once, so it never matters again; it keeps
# files readable). `None` in the allowed-types tuple means the field may be null or left out.
FORECAST = {
    "forecast_id": (str,), "created_at": (str,), "code_sha": (str,),
    "model": (str,), "model_version": (str,), "model_state": (str,),
    "entity_type": (str,), "entity_id": (str, int), "entity_name": (str, None),
    "target": (str,), "horizon": (str,), "resolves_after": (str,), "min_sample": (str, None),
    "prob": (float, int, None), "point": (float, int, None), "lo80": (float, int, None), "hi80": (float, int, None),
    "baselines": (dict, None), "deserved_target": (bool, None), "detail": (dict, None),
}
RESOLUTION = {
    "forecast_id": (str,), "resolved_at": (str,), "code_sha": (str,), "status": (str,),
    "actual": (float, int, bool, None), "actual_deserved": (float, int, None), "sample": (str, None),
    "void_reason": (str, None),
}
WRITER_INPUT_FORECAST = set(FORECAST) - {"created_at", "code_sha", "model_state"}
WRITER_INPUT_RESOLUTION = set(RESOLUTION) - {"resolved_at", "code_sha"}


class LedgerError(ValueError):
    pass


class NotActions(PermissionError):
    pass


# --- format --------------------------------------------------------------------------------------------

def _ts(s: str, field: str) -> datetime:
    try:
        return datetime.strptime(s, STAMP).replace(tzinfo=UTC)
    except (TypeError, ValueError):
        raise LedgerError(f"{field} must be UTC like 2027-01-05T14:00:00Z, got {s!r}") from None


def _day(s: str, field: str) -> date:
    try:
        return date.fromisoformat(s)
    except (TypeError, ValueError):
        raise LedgerError(f"{field} must be a date like 2027-10-01, got {s!r}") from None


def _types(rec: dict, spec: dict, what: str) -> None:
    extra = set(rec) - set(spec)
    if extra:
        raise LedgerError(f"{what}: unknown fields {sorted(extra)} (type-specific fields go in `detail`)")
    for k, ok in spec.items():
        v = rec.get(k)
        if v is None:
            if None not in ok:
                raise LedgerError(f"{what}: `{k}` is required")
            continue
        types = tuple(t for t in ok if t is not None)
        if isinstance(v, bool) and bool not in types:
            raise LedgerError(f"{what}: `{k}` can't be true/false")
        if not isinstance(v, types):
            raise LedgerError(f"{what}: `{k}` has the wrong type ({type(v).__name__})")
        if isinstance(v, float) and not math.isfinite(v):
            raise LedgerError(f"{what}: `{k}` isn't a finite number")


def validate_forecast(rec: dict) -> None:
    fid = rec.get("forecast_id")
    what = f"forecast {fid!r}"
    _types(rec, FORECAST, what)
    if not ID.match(fid):
        raise LedgerError(f"{what}: id must be lowercase letters, digits, '.', '_', '-' (3-160 chars)")
    if rec["model_state"] not in ("validated", "experimental"):
        raise LedgerError(f"{what}: model_state must be validated or experimental")
    made, after = _ts(rec["created_at"], "created_at"), _day(rec["resolves_after"], "resolves_after")
    if after <= made.date():
        raise LedgerError(f"{what}: resolves_after {after} must be after the day it was made ({made.date()})")
    prob, interval = rec.get("prob"), [rec.get(k) for k in ("point", "lo80", "hi80")]
    if prob is not None:
        if any(v is not None for v in interval):
            raise LedgerError(f"{what}: a forecast is either a probability or a point with an interval, not both")
        if not 0 <= prob <= 1:
            raise LedgerError(f"{what}: prob must be in [0, 1]")
    elif any(v is None for v in interval):
        raise LedgerError(f"{what}: needs prob, or point with lo80 and hi80 (every projection ships with an interval)")
    elif not interval[1] <= interval[0] <= interval[2]:
        raise LedgerError(f"{what}: needs lo80 <= point <= hi80")


def validate_resolution(rec: dict, forecast: dict | None) -> None:
    what = f"resolution of {rec.get('forecast_id')!r}"
    _types(rec, RESOLUTION, what)
    if forecast is None:
        raise LedgerError(f"{what}: no such forecast in this sport and season")
    at = _ts(rec["resolved_at"], "resolved_at")
    if at < _ts(forecast["created_at"], "created_at"):
        raise LedgerError(f"{what}: resolved before the forecast was made")
    if rec["status"] == "void":
        if not (rec.get("void_reason") or "").strip():
            raise LedgerError(f"{what}: a void needs a reason")
        if rec.get("actual") is not None:
            raise LedgerError(f"{what}: a void has no actual")
    elif rec["status"] == "scored":
        if rec.get("void_reason") is not None:
            raise LedgerError(f"{what}: a scored resolution has no void_reason")
        if rec.get("actual") is None:
            raise LedgerError(f"{what}: a scored resolution needs an actual")
        if at.date() < _day(forecast["resolves_after"], "resolves_after"):
            raise LedgerError(f"{what}: scored before resolves_after ({forecast['resolves_after']})")
    else:
        raise LedgerError(f"{what}: status must be scored or void")


def dumps(rec: dict, spec: dict) -> str:
    """One ledger line: keys in the format's order, nulls left out, no NaN, UTF-8."""
    out = {k: rec[k] for k in spec if rec.get(k) is not None}
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def _lines(b: bytes, path: str) -> list[dict]:
    if b and not b.endswith(b"\n"):
        raise LedgerError(f"{path}: the last line has no newline (a partial write?)")
    out = []
    for i, ln in enumerate(b.decode("utf-8").split("\n")[:-1] if b else [], 1):
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError as e:
            raise LedgerError(f"{path}:{i}: not JSON ({e.msg})") from None
    return out


def validate_files(files: dict[str, bytes]) -> None:
    """Every ledger file: well-formed lines, unique ids, created_at never going backwards within a file,
    and each resolution pointing at one forecast in the same sport and season, at most once."""
    forecasts: dict[tuple[str, str], dict[str, dict]] = {}
    for path in sorted(files):
        m = FILE.match(path)
        if not m:
            raise LedgerError(f"forecasts/{path}: not a ledger file name (<sport>/<season>[.resolved].jsonl)")
        if m["res"]:
            continue
        seen, last = {}, None
        for i, rec in enumerate(_lines(files[path], path), 1):
            try:
                validate_forecast(rec)
            except LedgerError as e:
                raise LedgerError(f"{path}:{i}: {e}") from None
            if rec["forecast_id"] in seen:
                raise LedgerError(f"{path}:{i}: duplicate forecast_id {rec['forecast_id']}")
            if last and rec["created_at"] < last:
                raise LedgerError(f"{path}:{i}: created_at goes backwards")
            seen[rec["forecast_id"]], last = rec, rec["created_at"]
        forecasts[(m["sport"], m["season"])] = seen
    for path in sorted(files):
        m = FILE.match(path)
        if not m["res"]:
            continue
        fc, done = forecasts.get((m["sport"], m["season"]), {}), set()
        for i, rec in enumerate(_lines(files[path], path), 1):
            try:
                validate_resolution(rec, fc.get(rec.get("forecast_id")))
            except LedgerError as e:
                raise LedgerError(f"{path}:{i}: {e}") from None
            if rec["forecast_id"] in done:
                raise LedgerError(f"{path}:{i}: {rec['forecast_id']} is already resolved")
            done.add(rec["forecast_id"])


# --- reading -------------------------------------------------------------------------------------------

def _disk(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*.jsonl"))}


def read(sport: str, season: int, root: Path = LEDGER) -> tuple[list[dict], list[dict]]:
    """(forecasts, resolutions) for one sport and season."""
    f, r = root / sport / f"{season}.jsonl", root / sport / f"{season}.resolved.jsonl"
    return (_lines(f.read_bytes(), str(f)) if f.exists() else [],
            _lines(r.read_bytes(), str(r)) if r.exists() else [])


# --- writing (Actions only) ----------------------------------------------------------------------------

def _require_actions(root: Path) -> str:
    """The real ledger is written only by a GitHub Actions run on main. Returns the commit sha the run
    is building from. A scratch root (tests, `smoke`) has no such rule."""
    if root.resolve() != LEDGER.resolve():
        return os.environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD", cwd=ROOT).strip()
    env = os.environ
    if env.get("GITHUB_ACTIONS") != "true" or env.get("GITHUB_REF") != "refs/heads/main" \
            or env.get("GITHUB_EVENT_NAME") not in ("schedule", "workflow_dispatch", "push"):
        raise NotActions("The ledger is written only by a GitHub Actions run on main (PLAN-scout §4). "
                         "Locally, write to a scratch root (`python -m sparvi.ledger smoke`).")
    return env["GITHUB_SHA"]


def _append(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as fh:
        fh.write("".join(lines).encode("utf-8"))


def _inputs(rec: dict, allowed: set, spec: dict, what: str) -> None:
    own = (set(rec) - allowed) & set(spec)
    if own:
        raise LedgerError(f"{what}: the writer sets {sorted(own)} itself")
    if set(rec) - allowed:
        raise LedgerError(f"{what}: unknown fields {sorted(set(rec) - allowed)} (type-specific fields go in `detail`)")


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def append_forecasts(sport: str, season: int, recs: list[dict], root: Path = LEDGER, gate=None) -> int:
    """Append new forecasts. The writer stamps created_at, code_sha and the model's gate state; every
    record must name the model's current registered version. Returns the number written. Nothing is
    written unless every record passes."""
    if not SPORT.match(sport):
        raise LedgerError(f"bad sport {sport!r}")
    sha = _require_actions(root)
    if gate is None:
        from backtest.gate import require_passing as gate
    now = _now().strftime(STAMP)
    existing, _ = read(sport, season, root)
    ids = {r["forecast_id"] for r in existing}
    if existing and existing[-1]["created_at"] > now:
        raise LedgerError("the clock is behind the ledger's last line")
    lines, states = [], {}
    for rec in recs:
        _inputs(rec, WRITER_INPUT_FORECAST, FORECAST, f"forecast {rec.get('forecast_id')!r}")
        if rec["model"] not in states:
            states[rec["model"]] = gate(rec["model"])
        st = states[rec["model"]]
        if rec.get("model_version") != st["version"]:
            raise LedgerError(f"forecast {rec['forecast_id']!r}: model_version {rec.get('model_version')} isn't "
                              f"{rec['model']}'s current version {st['version']}")
        full = {**rec, "created_at": now, "code_sha": sha, "model_state": st["state"]}
        validate_forecast(full)
        if full["forecast_id"] in ids:
            raise LedgerError(f"forecast {full['forecast_id']!r} is already in the ledger (forecasts are never replaced)")
        ids.add(full["forecast_id"])
        lines.append(dumps(full, FORECAST))
    _append(root / sport / f"{season}.jsonl", lines)
    return len(lines)


def append_resolutions(sport: str, season: int, recs: list[dict], root: Path = LEDGER) -> int:
    """Append resolutions: {forecast_id, status: scored, actual[, actual_deserved, sample]} or
    {forecast_id, status: void, void_reason}. Each forecast resolves once."""
    sha = _require_actions(root)
    now = _now().strftime(STAMP)
    fc, res = read(sport, season, root)
    fc = {r["forecast_id"]: r for r in fc}
    done = {r["forecast_id"] for r in res}
    lines = []
    for rec in recs:
        _inputs(rec, WRITER_INPUT_RESOLUTION, RESOLUTION, f"resolution of {rec.get('forecast_id')!r}")
        full = {**rec, "resolved_at": now, "code_sha": sha}
        validate_resolution(full, fc.get(rec.get("forecast_id")))
        if full["forecast_id"] in done:
            raise LedgerError(f"{full['forecast_id']} is already resolved")
        done.add(full["forecast_id"])
        lines.append(dumps(full, RESOLUTION))
    _append(root / sport / f"{season}.resolved.jsonl", lines)
    return len(lines)


# --- the append-only check -----------------------------------------------------------------------------

def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
                          encoding="utf-8").stdout


def _at(base: str, repo: Path) -> dict[str, bytes]:
    names = _git("ls-tree", "-r", "--name-only", base, "--", "forecasts", cwd=repo).split("\n")
    out = {}
    for n in filter(None, names):
        if n.endswith(".jsonl"):
            out[n.removeprefix("forecasts/")] = subprocess.run(
                ["git", "show", f"{base}:{n}"], cwd=repo, check=True, capture_output=True).stdout
    return out


def check(base: str, repo: Path = ROOT, pr: bool = False) -> list[str]:
    """Compare the working tree's ledger with `base`. Returns what changed (empty when nothing did);
    raises LedgerError on any edit, deletion, invalid line, or (with pr=True) any new line at all."""
    before, after = _at(base, repo), _disk(repo / "forecasts")
    for path, old in before.items():
        new = after.get(path)
        if new is None:
            raise LedgerError(f"forecasts/{path} was deleted (the ledger is append-only)")
        if not new.startswith(old):
            n = next((i for i, (a, b) in enumerate(zip(old.split(b"\n"), new.split(b"\n"), strict=False), 1) if a != b),
                     old.count(b"\n"))
            raise LedgerError(f"forecasts/{path}: line {n} was changed or removed (the ledger is append-only; "
                              "void a forecast with a reason instead)")
    validate_files(after)
    added = [f"forecasts/{p}: +{after[p].count(b'\n') - before.get(p, b'').count(b'\n')} lines"
             for p in sorted(after) if after[p] != before.get(p)]
    if pr and added:
        raise LedgerError("ledger lines arrive only by an Actions push to main, never in a pull request: "
                          + "; ".join(added))
    return added


# --- smoke: the whole write path in a scratch copy -----------------------------------------------------

def smoke() -> None:
    """Copy the repo's ledger into a scratch git repo, write a forecast and its resolution through the
    real writer and gate, commit, and run the append-only check; then confirm it catches an edit. Proves
    the path without putting anything in the real ledger."""
    tmp = Path(tempfile.mkdtemp(prefix="ledger-smoke-"))
    try:
        if LEDGER.exists():
            shutil.copytree(LEDGER, tmp / "forecasts")
        (tmp / "forecasts").mkdir(exist_ok=True)
        for args in (("init", "-q"), ("config", "user.email", "smoke@localhost"), ("config", "user.name", "smoke"),
                     ("config", "core.autocrlf", "false"), ("add", "-A"), ("commit", "-q", "--allow-empty", "-m", "base")):
            _git(*args, cwd=tmp)
        from sparvi import registry
        m = registry.get("cfb_power_rating")
        today = _now().date()
        season = today.year
        n = append_forecasts("smoke", season, [{
            "forecast_id": f"smoke-{season}-1", "model": m.name, "model_version": m.version,
            "entity_type": "team", "entity_id": "smoke", "target": "home_margin", "horizon": "next-game",
            "resolves_after": date.fromordinal(today.toordinal() + 7).isoformat(),
            "point": 3.0, "lo80": -9.0, "hi80": 15.0}], root=tmp / "forecasts")
        v = append_resolutions("smoke", season, [{"forecast_id": f"smoke-{season}-1", "status": "void",
                                                  "void_reason": "smoke test"}], root=tmp / "forecasts")
        added = check("HEAD", tmp)
        print(f"smoke: wrote {n} forecast and {v} void; check vs base: {added}")
        _git("add", "-A", cwd=tmp)
        _git("commit", "-q", "-m", "write", cwd=tmp)
        f = tmp / "forecasts" / "smoke" / f"{season}.jsonl"
        f.write_bytes(f.read_bytes().replace(b'"point":3.0', b'"point":4.0'))
        try:
            check("HEAD", tmp)
        except LedgerError as e:
            print(f"smoke: an edited line fails as it should: {e}")
        else:
            raise SystemExit("smoke: the check missed an edited line")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sparvi.ledger", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="append-only check against a base commit")
    c.add_argument("--base", required=True)
    c.add_argument("--pr", action="store_true", help="also refuse any new ledger line (pull requests)")
    sub.add_parser("smoke", help="exercise the writer and the check in a scratch copy")
    a = ap.parse_args(argv)
    if a.cmd == "smoke":
        smoke()
        return 0
    try:
        added = check(a.base, pr=a.pr)
    except LedgerError as e:
        print(f"ledger check FAILED: {e}", file=sys.stderr)
        return 1
    print("ledger check passed: " + ("; ".join(added) if added else "no changes"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
