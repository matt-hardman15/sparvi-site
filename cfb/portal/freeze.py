"""Freeze the live portal window's lists into the forecast ledger (SC-3a; run daily in a window by .github/workflows/
portal.yml, on main only: the ledger writer refuses anywhere else).

For each FBS program and position: one ranking record with the program's experimental fit top 25, and in `baselines`
the same board's top 25 by last season's production (yards), profile (adjusted efficiency composite), and transfer
rating (player ids only; no ratings are written). A record is written only when the program's list or a baseline list
changed since its last record, so a quiet day adds nothing.

Target: the listed player succeeds at that program within two seasons (wins the role and beats the starters' average);
resolves after season t+1. Scoring later counts, per list, the listed players who signed there and succeeded, against
the baselines' lists.

    python cfb/portal/freeze.py              # the live window (cfb.season + 1) as of today, if it has entries
"""
from __future__ import annotations

import re
import sys

import pandas as pd

from cfb.portal import board as B
from cfb.portal import export as X
from sparvi import ledger
from sparvi.config import cfb

SPORT = "cfb"
TARGET = "transfer_success_2y"
PROJ_MODEL = "cfb_player_projection"
PROJ_TARGET = "season_yards"         # QB passing on dropbacks, RB rushing, WR/TE receiving, at that program


def _slug(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def baselines(b: pd.DataFrame, top: int = B.TOP_FIT) -> dict[str, dict[str, list[str]]]:
    """Per position: the team-independent top `top` by production, profile, and transfer rating (ties by player id)."""
    one = b.drop_duplicates("player").copy()
    one["rating_key"] = one["stars"].fillna(0) + one["rating"].fillna(0) / 10
    one["eff_key"] = one["eff"].fillna(-1e9)
    out = {}
    for g, d in one.groupby("group"):
        out[g] = {name: d.sort_values([col, "player"], ascending=[False, True], kind="stable")["player"].head(top).tolist()
                  for name, col in (("production", "yards"), ("profile", "eff_key"), ("transfer_rating", "rating_key"))}
    return out


def records(res: dict, version: str, existing: list[dict], day: pd.Timestamp, proj_version: str | None = None) -> list[dict]:
    """Ranking records for each program and position: the fit list (cfb_system_fit) and, with proj_version, the
    projected-yards list (cfb_player_projection, with each player's projection and 80% interval). Each list is
    written only when it or its baselines changed since that model's last record for it."""
    t = res["window"]
    base = baselines(res["board"])
    last = {}
    for r in existing:
        d = r.get("detail") or {}
        if d.get("kind") == "ranking" and d.get("window") == t:
            last[(r.get("model", X.MODEL), d["program"], d["position"])] = (d["list"], r.get("baselines"))
    out = []
    for prog in res["programs"]:
        lists = [(X.MODEL, version, "fit", g, lst) for g, lst in prog["fit"].items()]
        if proj_version:
            lists += [(PROJ_MODEL, proj_version, "proj", g, lst) for g, lst in prog.get("proj", {}).items()]
        for model, ver, kind, g, lst in lists:
            ids = [x[0] for x in lst]
            if not ids:
                continue
            bl = base.get(g, {})
            if last.get((model, prog["team"], g)) == (ids, bl):
                continue
            detail = {"kind": "ranking", "window": t, "program": prog["team"], "position": g, "as_of": str(day.date()),
                      "top": B.TOP_FIT, "list": ids}
            if kind == "proj":
                detail["projections"] = [[x[0], x[1], x[2], x[3]] for x in lst]   # id, yards, lo80, hi80
            out.append({
                "forecast_id": f"cfb-{'portal' if kind == 'fit' else 'proj'}-{t}-{_slug(prog['team'])}-{g.lower()}-{day:%Y%m%d}",
                "model": model, "model_version": ver, "entity_type": "program_board",
                "entity_id": f"{t}:{prog['team']}:{g}" + ("" if kind == "fit" else ":proj"),
                "entity_name": f"{prog['team']} {g} {'board' if kind == 'fit' else 'projected yards'}, {t} window",
                "target": TARGET if kind == "fit" else PROJ_TARGET,
                "horizon": f"seasons {t}-{t + 1}" if kind == "fit" else f"season {t}",
                "resolves_after": f"{t + 2}-03-01" if kind == "fit" else f"{t + 1}-03-01", "min_sample": None,
                "prob": None, "point": None, "lo80": None, "hi80": None, "baselines": bl, "deserved_target": None,
                "detail": detail})
    return out


def main() -> int:
    from sparvi import registry
    t = cfb().season + 1
    day = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    wins = dict(X.windows(day))
    if t not in wins:
        print(f"no live window: the {t} portal has no entries yet")
        return 0
    done = X.export(t, wins[t])
    X.write_index([done])
    res = done["result"]
    existing, _ = ledger.read(SPORT, t)
    recs = records(res, registry.get(X.MODEL).version, existing, day, registry.get(PROJ_MODEL).version)
    n = ledger.append_forecasts(SPORT, t, recs) if recs else 0
    print(f"{t} window, {day.date()}: {n} ranking records written ({len(res['programs'])} programs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
