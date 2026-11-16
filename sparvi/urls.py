"""Stable URLs for generated entity pages (PLAN-seo S2).

A player's page lives at /baseball/players/<slug>-<mlbam_id>/. The id keeps it unique (two players
can share a name) and stable through name changes; the slug is ASCII.

`slug()` has a JavaScript twin, `Sparvi.slug` in site/assets/js/sparvi.js, so tools can link to a page
from a name without downloading a lookup table. tests/test_site.py checks the two agree on every name.

`update_registry()` keeps site/data/baseball/player_urls.json: every page path a player has ever had,
oldest first, the current one last. It only grows, so a URL that was ever published keeps resolving
(the page generator writes a redirect for each old path).
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path


def slug(name: str) -> str:
    """'José Ramírez' -> 'jose-ramirez'. Accents dropped; anything else outside a-z0-9 becomes a hyphen."""
    s = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-") or "player"


def player_path(pid: int, name: str) -> str:
    return f"{slug(name)}-{pid}"


def update_registry(path: Path, players: dict[int, str]) -> dict[str, list[str]]:
    """Add each current player's path (id -> name) to the registry at `path`; never remove one."""
    reg: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for pid, name in players.items():
        cur = player_path(pid, name)
        paths = [p for p in reg.get(str(pid), []) if p != cur]
        reg[str(pid)] = paths + [cur]
    return dict(sorted(reg.items(), key=lambda kv: int(kv[0])))
