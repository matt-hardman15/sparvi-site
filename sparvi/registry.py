"""Model registry, versions, and champion/challenger (PLAN-scout A2).

Every model behind a published number has a card, models/<sport>/<name>/card.yml: what it predicts,
its code, its training data, its validation metrics, the frozen evaluation set it is judged on, and
its status. Each version has a record, models/<sport>/<name>/versions/<version>.json, written when the
version is registered: code hash, training-data hash, artifact hashes, the eval-set identity, and the
validation metrics. Old versions stay reproducible from their record plus the git history.

Versioning (semantic): MAJOR = a new specification (target, features, method); MINOR = new training
data under the same specification (e.g. season rollover); PATCH = a bug fix.

    python -m sparvi.registry list
    python -m sparvi.registry check                     # CI: code changed without a new version fails
    python -m sparvi.registry register <name> <version> [--challenger] [--note "..."]
    python -m sparvi.registry promote <name> <version>  # challenger -> champion, if it earns it
    python -m sparvi.registry export                    # site/_generated/models.yml + site/data/models/registry.json

Promotion requires both (PLAN-scout A2): the challenger beats the champion on the same frozen eval
set (primary metric), and it doesn't lose on live ledger forecasts over the minimum window. Until the
ledger exists (L1/L2) there are no live forecasts, and the record says so. Every promotion, demotion
and retirement is appended to review/model-changes.md with before/after metrics.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from sparvi.config import ROOT

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
STATUSES = {"champion", "challenger", "retired"}


def _sha(path: Path) -> str:
    """Content hash with line endings normalized (a Windows checkout and CI must agree)."""
    b = path.read_bytes()
    if path.suffix in {".py", ".sql", ".yml", ".yaml", ".toml", ".md", ".txt", ".json", ".csv", ".sh"}:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def _get(d: dict, dotted: str):
    for k in dotted.split("."):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


@dataclass
class Model:
    card_path: Path
    card: dict

    @property
    def name(self) -> str:
        return self.card["name"]

    @property
    def dir(self) -> Path:
        return self.card_path.parent

    @property
    def root(self) -> Path:
        return self.card_path.parents[3]

    @property
    def version(self) -> str:
        return self.card["version"]

    def files(self, key: str) -> list[Path]:
        out = []
        for pat in self.card.get(key, []):
            hits = sorted(glob.glob(str(self.root / pat)))
            out += [Path(h) for h in hits] if hits else []
        return out

    def code_hash(self) -> tuple[str, dict[str, str]]:
        files = {p.relative_to(self.root).as_posix(): _sha(p) for p in self.files("code")}
        missing = [c for c in self.card.get("code", []) if not glob.glob(str(self.root / c))]
        if missing:
            raise FileNotFoundError(f"{self.name}: code paths not found: {missing}")
        h = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        return h, files

    def record_path(self, version: str) -> Path:
        return self.dir / "versions" / f"{version}.json"

    def record(self, version: str) -> dict:
        return json.loads(self.record_path(version).read_text(encoding="utf-8"))

    def metrics(self) -> tuple[dict, dict]:
        v = self.card["validation"]
        data = json.loads((self.root / v["file"]).read_text(encoding="utf-8"))
        metrics = {k: _get(data, path) for k, path in v["metrics"].items()}
        ev = self.card["eval"]
        eval_set = {k: _get(data, path) for k, path in ev.get("set_keys", {}).items()}
        return metrics, eval_set


def models(root: Path = ROOT) -> list[Model]:
    return [Model(p, yaml.safe_load(p.read_text(encoding="utf-8"))) for p in sorted((root / "models").glob("*/*/card.yml"))]


def get(name: str, root: Path = ROOT) -> Model:
    for m in models(root):
        if m.name == name:
            return m
    raise KeyError(f"no model card named {name!r}")


def stamp(name: str, root: Path = ROOT) -> dict:
    """What an export publishes next to a model's numbers: name, version, and a label for prose."""
    m = get(name, root)
    return {"model": m.name, "version": m.version, "label": f"{m.card['title']} v{m.version}"}


# ---- registration and promotion --------------------------------------------------------------------
def register(name: str, version: str, challenger: bool = False, note: str = "", today: date | None = None,
             root: Path = ROOT, hash_training_data: bool = True) -> dict:
    """Write the version record for the files as they are now. A new champion version updates the card."""
    if not SEMVER.match(version):
        raise ValueError(f"version must be MAJOR.MINOR.PATCH, got {version!r}")
    m = get(name, root)
    path = m.record_path(version)
    if path.exists():
        raise FileExistsError(f"{path.relative_to(root)} exists; versions are never rewritten (use a new version)")
    code_hash, code_files = m.code_hash()
    metrics, eval_set = m.metrics()
    data = {p.relative_to(root).as_posix(): _sha(p) for p in m.files("training_data")} if hash_training_data else {}
    rec = {
        "model": name, "version": version, "status": "challenger" if challenger else "champion",
        "registered": (today or date.today()).isoformat(), "note": note,
        "code_hash": code_hash, "code_files": code_files,
        "training_data": data or "not hashed (raw data not present)",
        "artifacts": {p.relative_to(root).as_posix(): _sha(p) for p in m.files("artifacts")},
        "eval": {"set": m.card["eval"]["set"], "set_keys": eval_set, "primary": m.card["eval"]["primary"],
                 "higher_is_better": m.card["eval"]["higher_is_better"]},
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    if not challenger:
        _set_card(m, version=version, status="champion")
    else:
        ch = m.card.setdefault("challengers", [])
        ch.append({"version": version, "since": rec["registered"]})
        _write_card(m)
    return rec


CARD_HEADER = "# Model card (PLAN-scout A2). Change `code` files only with a new version: python -m sparvi.registry --help\n"


def _write_card(m: Model) -> None:
    m.card_path.write_text(CARD_HEADER + yaml.safe_dump(m.card, sort_keys=False, allow_unicode=True, width=110),
                           encoding="utf-8", newline="\n")


def _set_card(m: Model, **kw) -> None:
    m.card.update(kw)
    _write_card(m)


def live_comparison(m: Model, challenger: str) -> dict:
    """Champion vs. challenger on live ledger forecasts. None exist before L1/L2."""
    sport = m.card["sport"]
    ledger = list((m.root / "forecasts" / sport).glob("*.jsonl")) if (m.root / "forecasts" / sport).exists() else []
    if not ledger:
        return {"available": False, "reason": "no ledger forecasts yet (L1/L2)", "passes": True}
    raise NotImplementedError("live champion/challenger scoring arrives with the ledger (L2)")


def promote(name: str, version: str, today: date | None = None, root: Path = ROOT) -> dict:
    m = get(name, root)
    champ_v = m.version
    champ, chal = m.record(champ_v), m.record(version)
    if chal["status"] != "challenger":
        raise ValueError(f"{name} {version} is not a challenger")
    if champ["eval"]["set_keys"] != chal["eval"]["set_keys"] or champ["eval"]["primary"] != chal["eval"]["primary"]:
        raise ValueError(f"{name}: champion and challenger were evaluated on different frozen sets "
                         f"({champ['eval']['set_keys']} vs {chal['eval']['set_keys']}); re-evaluate on the same one")
    key, hib = champ["eval"]["primary"], champ["eval"]["higher_is_better"]
    a, b = champ["metrics"][key], chal["metrics"][key]
    better = b > a if hib else b < a
    live = live_comparison(m, version)
    decision = {"model": name, "champion": champ_v, "challenger": version, "primary": key,
                "champion_value": a, "challenger_value": b, "frozen_set_passes": better, "live": live,
                "promoted": bool(better and live["passes"])}
    if decision["promoted"]:
        m.card.setdefault("history", []).append({"version": champ_v, "status": "retired", "until": (today or date.today()).isoformat()})
        m.card["challengers"] = [c for c in m.card.get("challengers", []) if c["version"] != version]
        _set_card(m, version=version, status="champion")
        log_change(root, today or date.today(), f"{m.card['title']} {champ_v} → {version} (promoted)",
                   f"Frozen eval set: {champ['eval']['set']}. {key}: {a:.4g} → {b:.4g}. Live: "
                   + ("not yet available (no ledger forecasts)" if not live["available"] else "passed") + ".")
    return decision


def log_change(root: Path, day: date, title: str, body: str) -> None:
    p = root / "review" / "model-changes.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    text = p.read_text(encoding="utf-8") if p.exists() else "# Model changes\n\nNewest first.\n"
    head, sep, rest = text.partition("\n## ")
    entry = f"\n## {day.isoformat()} · {title}\n{body}\n"
    p.write_text(head.rstrip("\n") + "\n" + entry + ((sep + rest) if sep else ""), encoding="utf-8", newline="\n")


# ---- what the site shows --------------------------------------------------------------------------------
def site_numbers(root: Path = ROOT) -> dict[str, str]:
    """Prose numbers for models.yml: '<name>' -> 'Stuff+ v1.0.0', '<name>_version' -> '1.0.0'."""
    out = {}
    for m in models(root):
        if m.card["status"] == "champion":
            out[m.name] = f"{m.card['title']} v{m.version}"
            out[f"{m.name}_version"] = m.version
    return out


def site_registry(root: Path = ROOT) -> list[dict]:
    rows = []
    for m in models(root):
        rec = m.record(m.version)
        rows.append({"model": m.name, "title": m.card["title"], "sport": m.card["sport"], "status": m.card["status"],
                     "version": m.version, "registered": rec["registered"], "primary": rec["eval"]["primary"],
                     "primary_value": rec["metrics"][rec["eval"]["primary"]],
                     "challengers": [c["version"] for c in m.card.get("challengers", [])]})
    return rows


def export(root: Path = ROOT) -> None:
    from sparvi import io
    if root == ROOT:
        io.write_numbers("models", site_numbers(root), "sparvi/registry.py")
        io.write_json(root / "site" / "data" / "models" / "registry.json", site_registry(root), indent=1)


def _stale_site_files(root: Path = ROOT) -> list[str]:
    import tempfile

    from sparvi import io
    out = []
    with tempfile.TemporaryDirectory() as td:
        saved = io.GENERATED
        try:
            io.GENERATED = Path(td)
            io.write_numbers("models", site_numbers(root), "sparvi/registry.py")
            io.write_json(Path(td) / "registry.json", site_registry(root), indent=1)
        finally:
            io.GENERATED = saved
        pairs = [(Path(td) / "models.yml", root / "site" / "_generated" / "models.yml"),
                 (Path(td) / "registry.json", root / "site" / "data" / "models" / "registry.json")]
        for want, have in pairs:
            if not have.exists() or have.read_bytes().replace(b"\r\n", b"\n") != want.read_bytes():
                out.append(f"{have.relative_to(root)} is stale: python -m sparvi.registry export")
    return out


# ---- the CI check --------------------------------------------------------------------------------------
def check(root: Path = ROOT) -> list[str]:
    problems = []
    for m in models(root):
        c = m.card
        missing = [k for k in ("name", "title", "sport", "status", "version", "code", "validation", "eval") if k not in c]
        if missing:
            problems.append(f"{m.dir.relative_to(root)}: card missing {missing}")
            continue
        if c["status"] not in STATUSES or not SEMVER.match(c["version"]):
            problems.append(f"{m.name}: bad status/version ({c['status']}, {c['version']})")
            continue
        if not m.record_path(c["version"]).exists():
            problems.append(f"{m.name}: no record for version {c['version']} (python -m sparvi.registry register)")
            continue
        rec = m.record(c["version"])
        now, files = m.code_hash()
        if now != rec["code_hash"]:
            changed = sorted(f for f in set(files) | set(rec["code_files"]) if files.get(f) != rec["code_files"].get(f))
            problems.append(f"{m.name}: code changed since v{c['version']} ({', '.join(changed)}). Bump the version "
                            "(MAJOR new spec / MINOR new data / PATCH fix), register it, and log it in review/model-changes.md")
        for ch in c.get("challengers", []):
            if not m.record_path(ch["version"]).exists():
                problems.append(f"{m.name}: challenger {ch['version']} has no record")
    if root == ROOT and not problems:
        problems += _stale_site_files(root)
    return problems


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # titles have arrows; Windows consoles default to cp1252
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("check")
    r = sub.add_parser("register"); r.add_argument("name"); r.add_argument("version")
    r.add_argument("--challenger", action="store_true"); r.add_argument("--note", default="")
    r.add_argument("--no-data-hash", action="store_true", help="skip hashing raw training data (CI has none)")
    p = sub.add_parser("promote"); p.add_argument("name"); p.add_argument("version")
    sub.add_parser("export")
    a = ap.parse_args()
    if a.cmd == "list":
        for m in models():
            print(f"{m.card['sport']:<9} {m.name:<22} v{m.version:<8} {m.card['status']:<10} {m.card['title']}")
    elif a.cmd == "check":
        probs = check()
        for x in probs:
            print("FAIL", x)
        print(f"registry: {len(models())} models, {len(probs)} problems")
        return 1 if probs else 0
    elif a.cmd == "register":
        rec = register(a.name, a.version, a.challenger, a.note, hash_training_data=not a.no_data_hash)
        export()
        print(json.dumps({k: rec[k] for k in ("model", "version", "status", "metrics")}, indent=2, default=str))
    elif a.cmd == "promote":
        print(json.dumps(promote(a.name, a.version), indent=2, default=str))
        export()
    elif a.cmd == "export":
        export()
    return 0


if __name__ == "__main__":
    sys.exit(main())
