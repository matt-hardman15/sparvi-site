"""Write the portal board's site data (SC-3a): site/data/cfb/portal/{index,board_<t>,programs_<t>}.json.

The only writer of these files. Per player: last season's aggregates (share, adjusted EPA per play and its percentile,
value over replacement, yards), size, experience, market tier, top components, similar players, destination. No
247Sports stars or ratings (Matt, Nov 15: used only inside the tier). Per program: its experimental fit top 25 by
position (ranks only, no probabilities), role availability, style in words, whether it kept its head coach.

Windows: the last completed window (as of May 1, after the spring window) and the current one once CFBD has entries.

    python cfb/portal/export.py                       # every window the board shows
    python cfb/portal/export.py --window 2027 --as-of 2027-01-05
"""
from __future__ import annotations

import argparse

import pandas as pd

from backtest.gate import require_passing
from cfb.fit import labels as LB
from cfb.fit import model as M
from cfb.portal import board as B
from ingest.cfb.players import seasons
from sparvi import registry
from sparvi.config import ROOT, cfb
from sparvi.io import write_json

OUT = ROOT / "site" / "data" / "cfb" / "portal"
MODEL = "cfb_system_fit"
PROJ_MODEL = "cfb_player_projection"


def windows(today: pd.Timestamp) -> list[tuple[int, pd.Timestamp]]:
    """The completed window (season = cfb().season: its board as of May 1) and, if it has entries, the next one as of
    today."""
    cur = cfb().season
    out = [(cur, pd.Timestamp(f"{cur}-05-01"))]
    nxt = cur + 1
    if nxt in seasons()["portal"]:
        from backtest.asof import Loader
        p = M.portal(Loader(), nxt, today)
        if len(p):
            out.append((nxt, today.normalize()))
    return out


def _r(v, d):
    return None if v is None or pd.isna(v) else round(float(v), d)


def player_rows(pl: pd.DataFrame) -> list[dict]:
    rows = []
    for r in pl.sort_values(["group", "vor", "player_id"], ascending=[True, False, True]).itertuples():
        rows.append({"id": r.player_id, "name": r.name, "pos": r.group, "from": r.team_from, "lvl": r.level,
                     "ht": _r(r.height, 0), "wt": _r(r.weight, 0), "exp": int(r.exp), "share": _r(r.share, 3),
                     "epa": _r(r.epa_adj, 3), "epa_pct": _r(r.epa_pct, 0), "vor": _r(r.vor, 1), "yds": _r(r.yards, 0),
                     "tier": int(r.tier), "floor": bool(r.floor), "str": list(r.strengths), "sim": list(r.similar),
                     "dest": None if pd.isna(r.destination) else r.destination, "best": list(r.best),
                     "entered": str(pd.Timestamp(r.transferDate).date()) if pd.notna(r.transferDate) else None})
    return rows


def export(t: int, as_of: pd.Timestamp) -> dict:
    st = require_passing(MODEL)
    pst = require_passing(PROJ_MODEL)
    res = B.build(t, as_of)
    stamp = registry.stamp(MODEL)
    head = {"window": t, "as_of": str(as_of.date()), "model": stamp["label"], "model_state": st["state"],
            "model_reason": st.get("reason"), "proj_model": registry.stamp(PROJ_MODEL)["label"], "proj_state": pst["state"],
            "proj_reason": pst.get("reason")}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / f"board_{t}.json", {**head, "tier_rules": {str(k): v for k, v in B.TIER_RULES.items()},
                                          "players": player_rows(res["players"])})
    numbers = {"horizon_years": B.HORIZON_YEARS, "strengths_n": B.N_STRENGTHS, "similar_seasons": B.SIMILAR_SEASONS, "last_cohort": B.LAST_COHORT,
               "backup_min": dict(LB.BACKUP_MIN)}
    write_json(OUT / f"programs_{t}.json", {**head, "top": B.TOP_FIT, "numbers": numbers, "programs": res["programs"]})
    return {"window": t, "as_of": str(as_of.date()), "players": int(len(res["players"])), "result": res}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int)
    ap.add_argument("--as-of")
    a = ap.parse_args()
    today = pd.Timestamp(a.as_of) if a.as_of else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    wins = [(a.window, today)] if a.window else windows(today)
    done = [{k: v for k, v in export(t, d).items() if k != "result"} for t, d in wins]
    write_index(done)
    print(done)


def write_index(done: list[dict]) -> None:
    """index.json: every window on the board, newest first (windows not rebuilt this run keep their entry)."""
    import json
    idx = OUT / "index.json"
    old = json.loads(idx.read_text(encoding="utf-8"))["windows"] if idx.exists() else []
    keep = {w["window"]: w for w in old}
    keep.update({w["window"]: {k: v for k, v in w.items() if k != "result"} for w in done})
    write_json(idx, {"windows": [keep[k] for k in sorted(keep, reverse=True)]})


if __name__ == "__main__":
    main()
