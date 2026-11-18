"""Profile engine (cfb/profile/engine.py): the per_team_play kind (RECV-MEASURES) and the biased-season flag."""
import numpy as np
import pandas as pd

from cfb.profile import engine as G


def _events():
    """Two games for team A (10 pass plays each), one receiver who played both, one who played only game 1."""
    rows = []
    for gid in (1, 2):
        for k in range(10):
            pid = f"{gid}-{k}"
            rows.append({"play_id": pid, "game_id": gid, "offense": "A", "defense": "B", "kind": "pass", "opp": "dropback",
                         "player_id": "qb", "yards": 0.0, "complete": 0.0, "epa": 0.0})
            who = "r1" if k < 4 or gid == 2 and k < 6 else ("r2" if gid == 1 and k < 8 else None)
            if who:
                rows.append({"play_id": pid, "game_id": gid, "offense": "A", "defense": "B", "kind": "pass", "opp": "target",
                             "player_id": who, "yards": 10.0, "complete": 1.0, "epa": 0.5})
    return pd.DataFrame(rows)


FEAT = {"id": "x_yptp", "label": "x", "group": ["WR"], "opp": "target", "filter": "complete == 1", "value": "yards",
        "kind": "per_team_play", "adjust": "none", "better": "higher", "min_n": 5}


def test_per_team_play_divides_by_team_pass_plays_in_his_games():
    ev = _events()
    pos = pd.Series({"qb": "QB", "r1": "WR", "r2": "WR"})
    prof = G.season_profile(ev, pos, 2025, [FEAT]).set_index("player_id")
    # r1: 4 + 6 catches x 10 yards over 20 team pass plays; r2: 4 catches x 10 over game 1's 10 plays.
    assert prof.loc["r1", "n"] == 20 and np.isclose(prof.loc["r1", "raw"], 5.0)
    assert prof.loc["r2", "n"] == 10 and np.isclose(prof.loc["r2", "raw"], 4.0)


def test_biased_seasons_keep_values_but_not_percentiles():
    ev = _events()
    pos = pd.Series({"qb": "QB", "r1": "WR", "r2": "WR"})
    f = {**FEAT, "biased_seasons": [2023]}
    biased = G.season_profile(ev, pos, 2023, [f])
    clean = G.season_profile(ev, pos, 2025, [f])
    assert biased["pct"].isna().all() and biased["raw"].notna().all()
    assert clean["pct"].notna().any()
