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


def records(res: dict, version: str, existing: list[dict], day: pd.Timestamp) -> list[dict]:
    t = res["window"]
    base = baselines(res["board"])
    last = {}
    for r in existing:
        d = r.get("detail") or {}
        if d.get("kind") == "ranking" and d.get("window") == t:
            last[(d["program"], d["position"])] = (d["list"], r.get("baselines"))
    out = []
    for prog in res["programs"]:
        for g, lst in prog["fit"].items():
            ids = [p for p, _ in lst]
            if not ids:
                continue
            bl = base.get(g, {})
            if last.get((prog["team"], g)) == (ids, bl):
                continue
            out.append({
                "forecast_id": f"cfb-portal-{t}-{_slug(prog['team'])}-{g.lower()}-{day:%Y%m%d}",
                "model": X.MODEL, "model_version": version, "entity_type": "program_board",
                "entity_id": f"{t}:{prog['team']}:{g}", "entity_name": f"{prog['team']} {g} board, {t} window",
                "target": TARGET, "horizon": f"seasons {t}-{t + 1}", "resolves_after": f"{t + 2}-03-01", "min_sample": None,
                "prob": None, "point": None, "lo80": None, "hi80": None, "baselines": bl, "deserved_target": None,
                "detail": {"kind": "ranking", "window": t, "program": prog["team"], "position": g, "as_of": str(day.date()),
                           "top": B.TOP_FIT, "list": ids}})
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
    recs = records(res, registry.get(X.MODEL).version, existing, day)
    n = ledger.append_forecasts(SPORT, t, recs) if recs else 0
    print(f"{t} window, {day.date()}: {n} ranking records written ({len(res['programs'])} programs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
