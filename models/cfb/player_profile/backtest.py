"""PLAYER-PROFILE in the backtest harness: when a player changes teams, do his competition-normalized components
carry over to the new school better than his raw ones?

Fold (outcome season t): no fitted weights (each season's profile comes from that season's plays alone), so fit only
records the label-permutation seed.
  predict   as of February 1 of season t: the season t-1 profile. For every opponent-adjusted feature and player with
            the feature's minimum: point = shrunk (adjusted, reliability-weighted) value, last_season = raw value, both
            standardized within feature. entity_id = "<athlete id>:<feature>"; `player` is the bootstrap block.
  outcome   the season t profile: the opponent-adjusted (unshrunk) value, standardized within feature, for movers
            (main team in season t differs from season t-1) with the minimum in both seasons.
FBS games only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cfb.profile import engine as G
from cfb.profile import events as E

OUTCOME_COLS = ["epa", "yards", "success", "first_down", "td", "complete", "sack", "interception", "explosive",
                "stuffed", "run10", "attempt"]


def _events(loader, s: int, as_of, permute_seed: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    games = loader.load(f"games_{s}", as_of, columns=["id"])
    plays = loader.load(f"plays_{s}", as_of)
    ev = E.plays(plays[plays["gameId"].isin(set(games["id"]))], loader.load(f"play_players_{s}", as_of))
    if permute_seed is not None:     # label permutation: every outcome shuffled across plays of the same opportunity
        rng = np.random.default_rng(permute_seed + s)
        for o in ev["opp"].unique():
            idx = np.flatnonzero(ev["opp"].to_numpy() == o)
            perm = rng.permutation(idx)
            cols = [c for c in OUTCOME_COLS if c in ev]
            ev.loc[idx, cols] = ev.loc[perm, cols].to_numpy()
    ros = loader.load(f"players_roster_{s}", as_of, columns=["id", "position"])
    pos = ros.assign(id=ros["id"].astype(str)).drop_duplicates("id").set_index("id")["position"]
    return ev, pos


def _profile(loader, s: int, as_of, permute_seed: int | None = None) -> pd.DataFrame:
    ev, pos = _events(loader, s, as_of, permute_seed)
    prof = G.season_profile(ev, pos, s)
    adjusted = {f["id"]: f["min_n"] for f in G.load_features() if f["adjust"] == "opponent"}
    prof = prof[prof["feature"].isin(adjusted)]
    return prof[prof["n"] >= prof["feature"].map(adjusted)]


def _z(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out.groupby("feature")[c].transform(lambda v: (v - v.mean()) / v.std())
    return out


def fit(loader, fold, permute_seed: int | None = None) -> dict:
    return {"permute_seed": permute_seed}


def predict(fitted, loader, fold) -> pd.DataFrame:
    p = _z(_profile(loader, fold.test_season - 1, fold.decision_date, fitted["permute_seed"]), ["shrunk", "raw"])
    return pd.DataFrame({"entity_id": p["player_id"] + ":" + p["feature"], "point": p["shrunk"], "last_season": p["raw"],
                         "player": p["player_id"], "feature": p["feature"], "team_prev": p["team"]})


def pairs(loader, fold) -> pd.DataFrame:
    """Season t outcome (adjusted, standardized within feature) for every player-feature with the minimum, with the
    season t-1 team, so callers can split movers and stayers."""
    t = fold.test_season
    cur = _z(_profile(loader, t, fold.resolve_after), ["adj", "raw"])
    prev = _profile(loader, t - 1, fold.resolve_after)
    team_prev = prev.drop_duplicates("player_id").set_index("player_id")["team"]
    cur["team_prev"] = cur["player_id"].map(team_prev)
    cur = cur[cur["team_prev"].notna()]
    return cur.assign(entity_id=cur["player_id"] + ":" + cur["feature"], mover=cur["team"] != cur["team_prev"])


def outcome(loader, fold) -> pd.DataFrame:
    d = pairs(loader, fold)
    d = d[d["mover"]]
    return pd.DataFrame({"entity_id": d["entity_id"], "y": d["adj"]})
