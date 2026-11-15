"""SYSTEM-FIT in the backtest harness (card models/cfb/system_fit/card.yml, pre-registered 2026-11-15).

Fold t (the portal window into season t), decision May 1 of season t:
  fit       transfer cohorts 2018 .. t-2 (two-year outcome complete), type and system dimensions refit on 2017 .. t-1;
            one logistic regression per fit group (cfb/fit/model.py)
  predict   every portal entrant filed by May 1 x every FBS program: entity_id "t|player|program"; point = his chance
            of ranking in that program's top 10 at his position; the baselines' credits on the same boards; `player`
            is the bootstrap block
  outcome   the cohort's successes within two years at FBS programs (y = 1), read after season t+1 completes

random_direction (the label-permutation null, Matt 2026-11-15): the mean board credit of N_DIRECTIONS rankers that score
each pair by a random direction over the same standardized features. A model trained on shuffled labels is one such
ranker; any of them puts players with real production at the board's extremes, so beating uniform random isn't a leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfb.fit import model as M

N_DIRECTIONS, DIRECTION_SEED = 20, 2026


def fit(loader, fold, permute_seed: int | None = None) -> dict:
    return M.fit_fold(loader, fold.test_season, fold.train_until, permute_seed)


def random_direction(fitted, loader, t: int, as_of, b: pd.DataFrame) -> np.ndarray:
    """Mean top-10 credit over random-direction rankers, on the same pairs and in the same row order as M.board."""
    e = M.entrants(loader, t, as_of)
    teams_t = loader.load(f"teams_{t}", as_of, columns=["school", "conference"])
    conf_t = dict(zip(teams_t["school"], teams_t["conference"], strict=True))
    fbs = sorted(conf_t)
    pid = pd.Series(np.repeat(e["player_id"].to_numpy(), len(fbs)))
    team = pd.Series(np.tile(fbs, len(e)))
    x = M.pairs(M.player_frame(loader, t, as_of, fitted["dims"]), M.team_frame(loader, t, as_of, fitted["dims"], conf_t), pid, team)
    grp = np.repeat(e["group"].to_numpy(), len(fbs))
    x["is_te"] = (grp == "TE").astype(float)
    fg = pd.Series(grp).map(M.FIT_GROUP).to_numpy()
    assert (pid.to_numpy() == b["player"].to_numpy()).all() and (team.to_numpy() == b["team"].to_numpy()).all()
    rng = np.random.default_rng(DIRECTION_SEED + t)
    total = np.zeros(len(x))
    for _ in range(N_DIRECTIONS):
        s = np.zeros(len(x))
        for g, m in fitted["models"].items():
            sel = fg == g
            if sel.any():
                s[sel] = M._standardize(x.loc[sel, m["cols"]], m["ref"]) @ rng.normal(size=len(m["cols"]))
        total += M.credit(pd.Series(s), [b["team"], b["group"]])
    return total / N_DIRECTIONS


def predict(fitted, loader, fold) -> pd.DataFrame:
    b = M.board(fitted, loader, fold.test_season, fold.decision_date)
    return pd.DataFrame({"entity_id": b["season"].astype(str) + "|" + b["player"] + "|" + b["team"],
                         "point": b["fit_credit"], "transfer_rating": b["transfer_rating_credit"],
                         "raw_production": b["raw_production_credit"], "profile_only": b["profile_only_credit"],
                         "random": b["random_credit"],
                         "random_direction": random_direction(fitted, loader, fold.test_season, fold.decision_date, b),
                         "player": b["player"], "group": b["group"], "p": b["p"]})


def outcome(loader, fold) -> pd.DataFrame:
    t = fold.test_season
    lab = M.cohort_outcomes(loader, t, t + 1)
    s = lab[lab["fbs"] & lab["success2"]].drop_duplicates("entity_id")
    return pd.DataFrame({"entity_id": s["entity_id"], "y": 1.0})
