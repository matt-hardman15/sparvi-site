"""CFB-PROJ in the backtest harness (card models/cfb/player_projection/card.yml, pre-registered before its first run).

Fold t (season t), decision May 1 of season t:
  predict   cfb/proj/pit.project: every QB, RB, WR, TE on each FBS team's projected roster as of May 1, trained on
            outcome seasons 2016 .. t-1; point = projected season-t yards in his category (QB passing on dropbacks,
            RB rushing, WR/TE receiving). entity_id "t|team|category|key" (key: his id as of May 1, a transfer's
            previous id); `player` is the bootstrap block, `position` the gate's segment (QB, RB, WR, TE).
  outcome   his season-t yards in that category at any team (0 if he recorded none). A transfer whose id changed at
            his new school is followed by the roster-based transfer link (labels.transfers).
  baselines carry: his own last-season yards in the category wherever he played (0 for freshmen);
            share_only: his last share x the destination's last-season yards in the category (0 for freshmen).
RB receiving is projected too but is not one of the pre-registered segments, so it isn't scored here.
"""
from __future__ import annotations

import pandas as pd

from cfb.fit import labels as LB
from cfb.fit import model as M
from cfb.proj import pit

SEGMENT = {"QB": "QB", "RB": "RB", "WR_rec": "WR", "TE_rec": "TE"}
GROUP = {"QB": "QB", "RB": "RB", "WR_rec": "REC", "TE_rec": "REC"}


def _eid(t: int, team, cat, key) -> str:
    return f"{t}|{team}|{cat}|{key}"


def fit(loader, fold, permute_seed: int | None = None) -> dict:
    """Training happens inside predict (every component is fit on seasons before t as of the decision date)."""
    return {"permute_seed": permute_seed}


def predict(fitted, loader, fold) -> pd.DataFrame:
    t = fold.test_season
    c = pit.project(loader, t, fold.decision_date, permute_seed=fitted["permute_seed"])
    c = c[c["category"].isin(SEGMENT)].reset_index(drop=True)
    return pd.DataFrame({"entity_id": [_eid(t, tm, cat, k) for tm, cat, k in zip(c["team"], c["category"], c["key"], strict=True)],
                         "point": c["proj"].to_numpy(), "carry": c["carry"].to_numpy(), "share_only": c["share_only"].to_numpy(),
                         "lo80": c["lo80"].to_numpy(), "hi80": c["hi80"].to_numpy(), "player": c["key"].astype(str).to_numpy(),
                         "position": c["category"].map(SEGMENT).to_numpy(), "status": c["status"].to_numpy()})


def outcome(loader, fold) -> pd.DataFrame:
    t = fold.test_season
    c = pit.candidates(loader, t, fold.decision_date, live=True)
    c["category"] = [g if g != "REC" else f"{p}_rec" for g, p in zip(c["group"], c["pos"], strict=True)]
    c = c[c["category"].isin(SEGMENT)]
    cur, prev = pit.season(loader, t), pit.season(loader, t - 1)
    played = {t - 1: prev["played"], t: cur["played"]}
    moved = LB.transfers({t - 1: prev["roster"], t: cur["roster"]}, t, M.portal(loader, t, M.done(t)).assign(season=t), played)
    new_id = dict(zip(moved["prev_id"].dropna().astype(str), moved.loc[moved["prev_id"].notna(), "player_id"].astype(str), strict=True))
    yards = cur["p"].set_index(["player_id", "group"])["yards"]
    y = [yards.get((new_id.get(str(k), str(k)), GROUP[cat]), 0.0) for k, cat in zip(c["key"], c["category"], strict=True)]
    return pd.DataFrame({"entity_id": [_eid(t, tm, cat, k) for tm, cat, k in zip(c["team"], c["category"], c["key"], strict=True)],
                         "y": y}).drop_duplicates("entity_id")
