"""The play-by-play integrity table for /status/: every game whose play-by-play disagrees with the official result
in a way the ratings act on or the status page lists, with its class and decision.

    python -m cfb.qa.pbp_integrity                 # after dbt: reads int_cfb_game_integrity and stg_cfb_plays
    python -m cfb.qa.pbp_integrity --from-status   # without a build: classes from the listed score gaps, repairs pending

Writes site/data/status/cfb_pbp_integrity.json (aggregates and game results only; no plays). Listed: every swapped
game, every game the rule repairs or excludes, and every over/under game off by `cfb_pbp_big_gap`+ points. The rule
is football/dbt/models/intermediate/int_cfb_game_integrity.sql.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import duckdb
import yaml

from ingest.cfb.integrity import EXPLAIN, WHAT, classify
from sparvi.config import ROOT
from sparvi.io import write_json

DB = Path(os.environ.get("FOOTBALL_DB_PATH", ROOT / "football" / "dbt" / "football.duckdb"))
STATUS = Path(os.environ.get("SPARVI_STATUS_DIR", ROOT / "site" / "data" / "status"))
VARS = yaml.safe_load((ROOT / "football" / "dbt" / "dbt_project.yml").read_text(encoding="utf-8"))["vars"]
RULE = [
    "CFBD's games table is the authority for who played and the final score.",
    "Swapped (the play-by-play reaches the official score with home and away exchanged): repaired. The game's scoring "
    "plays (offensive touchdowns and made field goals) show whether the team labels or the two score fields are "
    "exchanged, and that one is put right; if they can't tell, the teams stay as labeled and the running score is ignored.",
    "Over (a running score above a team's official final): kept. Teams, yards and EPA don't depend on the score "
    "fields; plays whose running score is impossible are never counted as garbage time.",
    "Under (the play-by-play stops short of the official score): kept as is.",
    "A play naming a team that isn't one of the game's two teams is dropped.",
]


def _row(season, gid, home, away, hp, ap, ph, pa, cls, decision) -> dict:
    return {"season": int(season), "id": int(gid), "home": home, "away": away, "result": f"{int(hp)}-{int(ap)}",
            "pbp": f"{int(ph)}-{int(pa)}", "class": cls, "gap": int(max(abs(ph - hp), abs(pa - ap))),
            "decision": decision, "what": WHAT[decision]}


def from_db(db: Path = DB) -> dict:
    con = duckdb.connect(str(db), read_only=True)
    big = VARS["cfb_pbp_big_gap"]
    rows = con.sql(f"""select season, game_id, home_team, away_team, home_points, away_points, pbp_home, pbp_away, pbp_class, decision
                       from int_cfb_game_integrity
                       where pbp_class = 'swapped' or decision <> 'keep' or (pbp_class <> 'match' and gap >= {big})
                       order by season, game_id""").fetchall()
    by_class = dict(con.sql("select pbp_class, count(*) from int_cfb_game_integrity group by 1 order by 1").fetchall())
    by_decision = dict(con.sql("select decision, count(*) from int_cfb_game_integrity where pbp_class <> 'match' group by 1 order by 1").fetchall())
    n_plays = con.sql("select count(*) from stg_cfb_plays").fetchone()[0]
    repaired = con.sql("""select count(*) from stg_cfb_plays p join int_cfb_game_integrity i using (game_id)
                          where i.decision in ('repair_teams', 'repair_scores')""").fetchone()[0]
    con.close()
    return {"built": True, "games_checked": int(sum(by_class.values())), "by_class": by_class, "by_decision": by_decision,
            "plays": int(n_plays), "plays_in_repaired_games": int(repaired), "games": [_row(*r) for r in rows]}


def from_status(status_dir: Path = STATUS) -> dict:
    """Without a build: the 9+ point games the ingest checks listed, classified; swapped games await their evidence."""
    gaps = json.loads((status_dir / "cfb.json").read_text(encoding="utf-8")).get("score_gaps", [])
    games = []
    for g in gaps:
        hp, ap = map(float, g["result"].split("-"))
        ph, pa = map(float, g["pbp"].split("-"))
        cls = classify(hp, ap, ph, pa)
        games.append(_row(g["season"], g["id"], g["home"], g["away"], hp, ap, ph, pa, cls, "pending" if cls == "swapped" else "keep"))
    return {"built": False, "games": games}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-status", action="store_true")
    a = ap.parse_args()
    res = from_status() if a.from_status else from_db()
    out = {"rule": RULE, "classes": {k: v for k, v in EXPLAIN.items() if k != "match"},
           "thresholds": {k: VARS[k] for k in ("cfb_pbp_big_gap", "cfb_pbp_swap_evidence_points", "cfb_pbp_exclude_gap")}, **res}
    write_json(STATUS / "cfb_pbp_integrity.json", out, indent=1)
    n = {}
    for g in out["games"]:
        n[(g["class"], g["decision"])] = n.get((g["class"], g["decision"]), 0) + 1
    print(f"cfb play-by-play integrity: {len(out['games'])} games listed", dict(sorted(n.items())))


if __name__ == "__main__":
    main()
