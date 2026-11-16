"""Each season's final player numbers, kept so player pages survive the season rollover (PLAN-seo S2:
"if a player has no current data, show his last season"; BACKLOG SEO-2b).

The export writes site/data/baseball/archive/players_<season>.json on every run (so during the season it
is the latest snapshot, and after the rollover it stays frozen). Each record is the player's page file cut
down to what the static page shows: bio rows, flags, and each timeline series reduced to its final
season-to-date value. site/build/player_pages.py renders a last-season page from it for any player in the
URL registry with no data this season. Not published on its own (removed from docs/ like players/).
"""
from __future__ import annotations

from pathlib import Path


def _last(series: dict) -> dict:
    return {f: v[-1:] for f, v in series.items()}


def slim(player: dict) -> dict:
    tl = {}
    for lvl, t in (player.get("timeline") or {}).items():
        out = {"dates": t["dates"][-1:]}
        for role in ("pitcher", "hitter"):
            if t.get(role) and t[role].get("std"):
                out[role] = {"std": _last(t[role]["std"])}
        if t.get("pitch_types"):
            out["pitch_types"] = [{"pt": p["pt"], "group": p.get("group"), "series": _last(p["series"])} for p in t["pitch_types"]]
        tl[lvl] = out
    return {"id": player["id"], "name": player["name"], "season": player["season"],
            "hitter_aaa": player.get("hitter_aaa"), "pitcher_aaa": player.get("pitcher_aaa"),
            "stuff": player.get("stuff") or [], "flags": player.get("flags") or [], "timeline": tl}


def path_for(root: Path, season: int) -> Path:
    return root / "archive" / f"players_{season}.json"


def archive(players: list[dict]) -> dict:
    return {str(p["id"]): slim(p) for p in sorted(players, key=lambda p: p["id"])}
