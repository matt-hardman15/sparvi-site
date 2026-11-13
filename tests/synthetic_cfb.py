"""Synthetic CollegeFootballData files for testing the CFB dbt models and data checks in CI (real data is
never committed). Shaped by contracts/cfb_ingest.yml: plays, games (with Elo and a bowl), teams, polls,
lines, plus kicks and penalties the efficiency plays must leave out, and running scores that end at
each game's result (the score check).

    write_cfb_fixture(out_dir, season) -> cfb/{plays,games,teams,rankings,lines}_{season-1,season}.parquet
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from fixture_contract import write_conformed

TEAMS = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel"]
CONF = {t: ("SEC" if i < 4 else "Sun Belt") for i, t in enumerate(TEAMS)}
WEEKS = 7
PLAYS_PER_SIDE = 60


def _season(season: int, rng: np.random.Generator, gid0: int, pid0: int):
    pairs = list(combinations(TEAMS, 2))
    rng.shuffle(pairs)
    games, used = [], set()
    for week in range(1, WEEKS + 1):
        busy = set()
        for a, b in pairs:
            if (a, b) in used or a in busy or b in busy:
                continue
            used.add((a, b)); busy |= {a, b}
            games.append({"week": week, "seasonType": "regular", "home": a, "away": b})
    games.append({"week": 1, "seasonType": "postseason", "home": TEAMS[0], "away": TEAMS[5]})   # a bowl
    strength = {t: rng.normal(0, 0.1) for t in TEAMS}
    elo = {t: 1500 + 400 * strength[t] for t in TEAMS}
    plays, gout = [], []
    pid = pid0
    for i, g in enumerate(games):
        gid = gid0 + i
        score = {g["home"]: 0, g["away"]: 0}
        for period in (1, 2, 3, 4):
            for off, dff in ((g["home"], g["away"]), (g["away"], g["home"])):
                for j in range(PLAYS_PER_SIDE // 4):
                    kind = rng.choice(["Rush", "Pass Reception", "Pass Incompletion", "Sack", "Punt", "Penalty"],
                                      p=[0.45, 0.3, 0.15, 0.04, 0.04, 0.02])
                    scrimmage = kind not in ("Punt", "Penalty")
                    gained = int(rng.normal(5, 7)) if scrimmage else 0
                    down = int(rng.integers(1, 5))
                    dist = int(rng.integers(1, 15))
                    pid += 1
                    plays.append({"season": season, "week": g["week"], "seasonType": g["seasonType"], "id": pid, "gameId": gid,
                                  "driveId": gid * 100 + period, "driveNumber": period, "playNumber": j + 1,
                                  "offense": off, "offenseConference": CONF[off], "defense": dff, "defenseConference": CONF[dff],
                                  "home": g["home"], "away": g["away"], "offenseScore": float(score[off]), "defenseScore": float(score[dff]),
                                  "period": period, "down": float(down), "distance": float(dist), "yardline": float(rng.integers(1, 99)),
                                  "yardsToGoal": float(rng.integers(1, 99)), "yardsGained": float(gained), "playType": kind,
                                  "ppa": float(rng.normal(strength[off] - strength[dff], 1.2)) if scrimmage else None,
                                  "scoring": False})
                    if scrimmage and rng.random() < 0.05:
                        score[off] += 7
        gout.append({"id": gid, "season": season, "week": g["week"], "seasonType": g["seasonType"],
                     "startDate": f"{season}-09-{g['week'] + 5:02d}T19:00:00.000Z" if g["seasonType"] == "regular" else f"{season}-12-28T19:00:00.000Z",
                     "completed": True, "neutralSite": g["seasonType"] == "postseason", "conferenceGame": CONF[g["home"]] == CONF[g["away"]],
                     "homeId": TEAMS.index(g["home"]), "homeTeam": g["home"], "homeConference": CONF[g["home"]], "homeClassification": "fbs",
                     "homePoints": float(score[g["home"]]), "homePregameElo": elo[g["home"]], "homePostgameElo": elo[g["home"]] + 5,
                     "awayId": TEAMS.index(g["away"]), "awayTeam": g["away"], "awayConference": CONF[g["away"]], "awayClassification": "fbs",
                     "awayPoints": float(score[g["away"]]), "awayPregameElo": elo[g["away"]], "awayPostgameElo": elo[g["away"]] - 5})
        # An end-of-game marker carrying the final score (as CFBD's feed does).
        pid += 1
        plays.append({**plays[-1], "id": pid, "playType": "End of Game", "ppa": None, "offense": g["home"], "defense": g["away"],
                      "offenseScore": float(score[g["home"]]), "defenseScore": float(score[g["away"]])})
    return pd.DataFrame(plays), pd.DataFrame(gout), pid


def write_cfb_fixture(out: Path, season: int, seed: int = 0) -> None:
    d = out / "cfb"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pid = 0
    for k, s in enumerate((season - 1, season)):
        plays, games, pid = _season(s, rng, gid0=10_000 * (k + 1), pid0=pid)
        write_conformed(plays, d / f"plays_{s}.parquet", "cfb", "plays")
        write_conformed(games, d / f"games_{s}.parquet", "cfb", "games")
        write_conformed(pd.DataFrame({"id": range(len(TEAMS)), "school": TEAMS, "conference": [CONF[t] for t in TEAMS]}),
                        d / f"teams_{s}.parquet", "cfb", "teams")
        write_conformed(pd.DataFrame({"season": s, "seasonType": "regular", "week": 3, "poll": "AP Top 25",
                                      "rank": range(1, len(TEAMS) + 1), "school": TEAMS}), d / f"rankings_{s}.parquet", "cfb", "rankings")
        write_conformed(pd.DataFrame({"game_id": games["id"], "season": s, "week": games["week"], "seasonType": games["seasonType"],
                                      "homeTeam": games["homeTeam"], "awayTeam": games["awayTeam"], "provider": "Book",
                                      "spread": -3.0, "spreadOpen": -2.5}), d / f"lines_{s}.parquet", "cfb", "lines")
