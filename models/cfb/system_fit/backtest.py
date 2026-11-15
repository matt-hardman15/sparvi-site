"""SYSTEM-FIT in the backtest harness (card models/cfb/system_fit/card.yml, pre-registered 2026-11-15).

Fold t (the portal window into season t), decision May 1 of season t:
  fit       transfer cohorts 2018 .. t-2 (two-year outcome complete), type and system dimensions refit on 2017 .. t-1;
            one logistic regression per fit group (cfb/fit/model.py)
  predict   every portal entrant filed by May 1 x every FBS program: entity_id "t|player|program"; point = his chance
            of ranking in that program's top 10 at his position; the baselines' credits on the same boards; `player`
            is the bootstrap block
  outcome   the cohort's successes within two years at FBS programs (y = 1), read after season t+1 completes
"""
from __future__ import annotations

import pandas as pd

from cfb.fit import model as M


def fit(loader, fold, permute_seed: int | None = None) -> dict:
    return M.fit_fold(loader, fold.test_season, fold.train_until, permute_seed)


def predict(fitted, loader, fold) -> pd.DataFrame:
    b = M.board(fitted, loader, fold.test_season, fold.decision_date)
    return pd.DataFrame({"entity_id": b["season"].astype(str) + "|" + b["player"] + "|" + b["team"],
                         "point": b["fit_credit"], "transfer_rating": b["transfer_rating_credit"],
                         "raw_production": b["raw_production_credit"], "profile_only": b["profile_only_credit"],
                         "random": b["random_credit"], "player": b["player"], "group": b["group"], "p": b["p"]})


def outcome(loader, fold) -> pd.DataFrame:
    t = fold.test_season
    lab = M.cohort_outcomes(loader, t, t + 1)
    s = lab[lab["fbs"] & lab["success2"]].drop_duplicates("entity_id")
    return pd.DataFrame({"entity_id": s["entity_id"], "y": 1.0})
