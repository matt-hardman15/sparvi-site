"""Second-stage features for the college basketball forecasts (CBB-0).

The ratings' margin error is regressed (least squares) on game features. Each feature is the home team's value less
the away team's, and each team value comes only from games before the forecast date. Groups:

    luck          points per game above or below a 34% league three-point rate, for and against, season to date
    availability  share of the team's minutes so far held by players who did not play in its last game
    continuity    share of last season's minutes played by players seen so far this season, and last-season minutes of
                  players arriving from other teams; both fade as the season goes on (FADE / (FADE + games played))
    form          the ratings' mean error on the team's last five games
    rest          days since the team's last game, capped at ten
    four_factors  season-to-date net turnover rate, offensive-rebound rate and free-throw rate (team less opponents)
    transfers     share of last season's minutes that left through the portal, and last-season minutes of players
                  arriving through it; both fade the same way

Which groups are used is decided on the tuning seasons (develop.py) and fixed in the model card.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUPS = {"luck": ["luck"], "availability": ["missing"], "continuity": ["returning", "incoming"], "form": ["form"],
          "rest": ["rest"], "four_factors": ["net_to", "net_orb", "net_ftr"], "transfers": ["portal_out", "portal_in"]}
# The groups the published model uses (cbb_team_rating 1.0.0): those whose removal raised the error on the tuning seasons
# (develop.py). Rest, four factors and transfers were tested and dropped. A change here is a new model version.
KEPT = ["luck", "availability", "continuity", "form"]
LEAGUE_3P = 0.34          # the three-point rate a shot is credited at
FADE = 5.0                # games after which a preseason input counts half
REST_CAP, FORM_GAMES = 10, 5
FOUR = {"to": "turnoverRatio", "orb": "offensiveReboundPct", "ftr": "freeThrowRate"}


def team_rows(f: pd.DataFrame) -> pd.DataFrame:
    """Two rows a game (one per team) in date order, with the ratings' signed error from that team's side and `k`,
    the number of games the team had played before it."""
    err = f["margin"] - f["pred_margin"]
    t = pd.concat([pd.DataFrame({"game_id": f["game_id"], "season": f["season"], "date": f["date"], "team": f[side], "err": sign * err})
                   for side, sign in (("home", 1), ("away", -1))])
    t = t.sort_values(["season", "team", "date", "game_id"]).reset_index(drop=True)
    t["k"] = t.groupby(["season", "team"]).cumcount()
    return t


def _to_date(t: pd.DataFrame, col: str) -> pd.Series:
    """The team's mean of `col` over its earlier games this season (0 before its first)."""
    before = t.groupby(["season", "team"])[col].cumsum() - t[col]
    return (before / t["k"].where(t["k"] > 0)).fillna(0.0)


def rest_form(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby(["season", "team"])
    t["rest"] = g["date"].diff().dt.days.clip(upper=REST_CAP).fillna(REST_CAP)
    t["form"] = g["err"].transform(lambda s: s.shift(1).rolling(FORM_GAMES, min_periods=1).mean()).fillna(0.0)
    return t


def box(t: pd.DataFrame, loader, seasons: list[int], as_of) -> pd.DataFrame:
    """Three-point luck and the four factors, from team box scores."""
    stats = {"m": "teamStats_threePointFieldGoals_made", "a": "teamStats_threePointFieldGoals_attempted",
             "om": "opponentStats_threePointFieldGoals_made", "oa": "opponentStats_threePointFieldGoals_attempted"}
    ff = {f"{side}_{k}": f"{side}Stats_fourFactors_{v}" for k, v in FOUR.items() for side in ("team", "opponent")}
    cols = ["gameId", "teamId", *stats.values(), *ff.values()]
    b = pd.concat([loader.load(f"cbb_team_games_{s}", as_of, columns=cols) for s in seasons]).drop_duplicates(["gameId", "teamId"])
    b = b.rename(columns={"gameId": "game_id", "teamId": "team", **{v: k for k, v in {**stats, **ff}.items()}})
    t = t.merge(b, on=["game_id", "team"], how="left")
    t["_x"] = 3 * ((t["m"] - LEAGUE_3P * t["a"]) - (t["om"] - LEAGUE_3P * t["oa"])).astype(float).fillna(0.0)
    t["luck"] = _to_date(t, "_x")
    for k in FOUR:
        own, opp = t[f"team_{k}"].astype(float), t[f"opponent_{k}"].astype(float)
        t["_x"] = ((opp - own) if k == "to" else (own - opp)).fillna(0.0)      # fewer turnovers than the opponent is good
        t[f"net_{k}"] = _to_date(t, "_x")
    return t.drop(columns=["_x", *stats, *ff])


def players(t: pd.DataFrame, loader, seasons: list[int], as_of) -> pd.DataFrame:
    """Availability and continuity, from player box scores. Reads the season before the first one for last season's minutes."""
    cols = ["gameId", "season", "teamId", "player_athleteId", "player_minutes"]
    p = pd.concat([loader.load(f"cbb_player_games_{s}", as_of, columns=cols) for s in [min(seasons) - 1, *seasons]])
    p.columns = ["game_id", "season", "team", "player", "minutes"]
    p = p.dropna(subset=["player"]).assign(minutes=lambda d: d["minutes"].fillna(0.0).astype(float))
    p = p.groupby(["season", "team", "game_id", "player"], as_index=False)["minutes"].sum()
    last_same = p.groupby(["season", "team", "player"])["minutes"].sum().to_dict()
    last_any = p.groupby(["season", "player"])["minutes"].sum().to_dict()
    team_total = p.groupby(["season", "team"])["minutes"].sum().to_dict()
    by_team = {k: v for k, v in p.groupby(["season", "team"])}
    out = []
    for (season, team), og in t[["season", "team", "game_id"]].groupby(["season", "team"]):
        games = og["game_id"].to_numpy()
        n = len(games)
        missing, returning, incoming = np.zeros(n), np.full(n, np.nan), np.full(n, np.nan)
        pg = by_team.get((season, team))
        if pg is not None:
            mat = pg.pivot_table(index="player", columns="game_id", values="minutes", aggfunc="sum").reindex(columns=games).fillna(0.0)
            M, ids = mat.to_numpy(), mat.index.to_numpy()
            cum = np.cumsum(M, axis=1)
            prev_total = team_total.get((season - 1, team), 0.0)
            same = np.array([last_same.get((season - 1, team, i), 0.0) for i in ids])
            anyw = np.array([last_any.get((season - 1, i), 0.0) for i in ids])
            for k in range(n):               # game k: only columns before k are read
                if k >= 2:
                    share = cum[:, k - 2] / max(cum[:, k - 2].sum(), 1.0)
                    missing[k] = share[M[:, k - 1] <= 0].sum()
                if k >= 1 and prev_total > 0:
                    seen = cum[:, k - 1] > 0
                    returning[k] = same[seen].sum() / prev_total
                    incoming[k] = (anyw[seen] - same[seen]).sum() / prev_total
        out.append(pd.DataFrame({"season": season, "team": team, "game_id": games, "missing": missing, "returning": returning, "incoming": incoming}))
    return t.merge(pd.concat(out), on=["season", "team", "game_id"], how="left")


def transfers(t: pd.DataFrame, loader, seasons: list[int], as_of) -> pd.DataFrame:
    """Portal moves before each season, weighted by last season's minutes. The portal table of year Y holds the moves
    between seasons Y and Y+1; the loader serves it from the start of season Y+1. Players are matched by name within
    the team they left."""
    rows = []
    for s in seasons:
        name = f"cbb_portal_{s - 1}"
        if name not in loader.tables:
            continue
        portal = loader.load(name, as_of)
        if portal.empty:
            continue
        pg = loader.load(f"cbb_player_games_{s - 1}", as_of, columns=["teamId", "player_name", "player_minutes"])
        mins = pg.groupby(["teamId", "player_name"], as_index=False)["player_minutes"].sum()
        total = mins.groupby("teamId")["player_minutes"].sum()
        portal = portal.assign(player_name=portal["firstName"].astype(str) + " " + portal["lastName"].astype(str))
        moved = portal.merge(mins, left_on=["origin_id", "player_name"], right_on=["teamId", "player_name"])
        d = pd.DataFrame({"portal_out": (moved.groupby("origin_id")["player_minutes"].sum() / total).dropna(),
                          "portal_in": (moved.dropna(subset=["destination_id"]).groupby("destination_id")["player_minutes"].sum() / total).dropna()})
        rows.append(d.fillna(0.0).rename_axis("team").reset_index().assign(season=s))
    if not rows:
        return t.assign(portal_out=0.0, portal_in=0.0)
    return t.merge(pd.concat(rows).astype({"team": "int64"}), on=["season", "team"], how="left")


def build(f: pd.DataFrame, loader, seasons: list[int], as_of, ret_mean: float) -> pd.DataFrame:
    """`f` (walk-forward forecasts with outcomes for the games already played) with one column per feature: the home
    team's value less the away team's. `ret_mean` is the typical returning share, so the continuity input is a
    difference from typical (fixed from the tuning seasons, in the card)."""
    t = transfers(players(box(rest_form(team_rows(f)), loader, seasons, as_of), loader, seasons, as_of), loader, seasons, as_of)
    fade = FADE / (FADE + t["k"])
    t["missing"] = t["missing"].fillna(0.0)
    t["returning"] = (t["returning"].fillna(ret_mean) - ret_mean) * fade
    for c in ("incoming", "portal_out", "portal_in"):
        t[c] = t[c].fillna(0.0) * fade
    names = [c for cols in GROUPS.values() for c in cols]
    side = t.set_index(["game_id", "team"])[names]
    out = f.copy()
    h = side.reindex(pd.MultiIndex.from_frame(out[["game_id", "home"]])).to_numpy()
    a = side.reindex(pd.MultiIndex.from_frame(out[["game_id", "away"]])).to_numpy()
    out[names] = h - a
    return out


def raw_returning(f: pd.DataFrame, loader, seasons: list[int], as_of) -> float:
    """The mean returning share over team-games of `seasons` (used once, on the tuning seasons, to set `ret_mean`)."""
    t = players(team_rows(f), loader, seasons, as_of)
    return float(t["returning"].mean())


def fit(d: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Least squares of the ratings' margin error on `cols`, with an intercept."""
    X = np.c_[np.ones(len(d)), d[cols].to_numpy(float)]
    return np.linalg.lstsq(X, (d["margin"] - d["pred_margin"]).to_numpy(float), rcond=None)[0]


def apply(d: pd.DataFrame, cols: list[str], beta: np.ndarray) -> np.ndarray:
    return d["pred_margin"].to_numpy(float) + np.c_[np.ones(len(d)), d[cols].to_numpy(float)] @ beta
