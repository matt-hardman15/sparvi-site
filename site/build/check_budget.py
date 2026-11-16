"""Hosting budget check (PLAN-seo S2). Last Quarto post-render step.

Measures docs/ (file count and bytes, by top-level section and by folder), writes
docs/_build-report.json, and compares against the current host's limits in budget.toml: warns at 60%
and fails the build at 75% of any limit. Check the report after adding a sport or a page type.

Standard library only: the deploy job runs Quarto without the project's virtualenv.
"""
from __future__ import annotations

import json
import os
import sys
import tomllib
from collections import Counter
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", SITE.parent / "docs"))
if not OUT.is_absolute():
    OUT = SITE / OUT
REPORT = "_build-report.json"


def measure(root: Path) -> dict:
    files = [p for p in root.rglob("*") if p.is_file() and p.name != REPORT]
    sizes = {p.relative_to(root).as_posix(): p.stat().st_size for p in files}
    sec_n, sec_b, dir_n, dir_b = Counter(), Counter(), Counter(), Counter()
    for rel, b in sizes.items():
        parts = rel.split("/")
        sec = parts[0] if len(parts) > 1 else "(root)"
        d = "/".join(parts[:3]) if len(parts) > 3 else "/".join(parts[:-1]) or "(root)"
        sec_n[sec] += 1; sec_b[sec] += b; dir_n[d] += 1; dir_b[d] += b  # noqa: E702
    return {"files": len(sizes), "bytes": sum(sizes.values()), "largest_file": max(sizes.items(), key=lambda kv: kv[1], default=("", 0)),
            "sections": {s: {"files": sec_n[s], "bytes": sec_b[s]} for s in sorted(sec_n)},
            "top_folders_by_files": dir_n.most_common(10), "top_folders_by_bytes": dir_b.most_common(10)}


def check(m: dict, cfg: dict) -> tuple[list[str], list[str], dict]:
    lim = cfg["hosts"][cfg["host"]]
    use = {}
    if "max_bytes" in lim:
        use["bytes"] = m["bytes"] / lim["max_bytes"]
    if "max_files" in lim:
        use["files"] = m["files"] / lim["max_files"]
    if "max_file_bytes" in lim:
        use["largest_file"] = m["largest_file"][1] / lim["max_file_bytes"]
    warn = [f"{k} at {v:.0%} of the {cfg['host']} limit" for k, v in use.items() if cfg["warn_at"] <= v < cfg["fail_at"]]
    fail = [f"{k} at {v:.0%} of the {cfg['host']} limit (fails at {cfg['fail_at']:.0%})" for k, v in use.items() if v >= cfg["fail_at"]]
    return warn, fail, use


def main() -> int:
    if os.environ.get("QUARTO_PROJECT_RENDER_ALL") not in (None, "1"):
        return 0
    cfg = tomllib.loads((SITE / "build" / "budget.toml").read_text(encoding="utf-8"))
    m = measure(OUT)
    warn, fail, use = check(m, cfg)
    report = {"host": cfg["host"], "usage": {k: round(v, 4) for k, v in use.items()}, **m}
    (OUT / REPORT).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"budget: {m['files']:,} files, {m['bytes'] / 1e6:.0f} MB ("
          + ", ".join(f"{k} {v:.0%}" for k, v in use.items()) + f" of {cfg['host']} limits); {REPORT} written")
    if warn or fail:
        print("budget: " + "; ".join(warn + fail), file=sys.stderr)
        for label, rows in (("files", m["top_folders_by_files"]), ("bytes", m["top_folders_by_bytes"])):
            print(f"  largest folders by {label}: " + ", ".join(f"{d} ({n:,})" for d, n in rows), file=sys.stderr)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
