"""Pull FBS college football from CFBD: teams, games (with pre-game Elo), betting lines, poll
rankings, and play-by-play, for every season in sparvi.toml [cfb].

    python ingest/cfb/pull.py --estimate        # how many uncached calls a run needs (no calls)
    python ingest/cfb/pull.py                   # pull (asks for --yes above the approval line)
    python ingest/cfb/pull.py --current         # only the current season (the weekly refresh)

Output (data/raw/cfb/, never committed; raw CFBD data is not redistributed):
    teams_{season}.parquet      FBS teams and conferences
    games_{season}.parquet      every game involving an FBS team, regular season + postseason
    lines_{season}.parquet      one row per game x sportsbook: spread/total, opening and closing
    rankings_{season}.parquet   one row per poll x week x ranked team (AP, Coaches, CFP)
    wepa_{season}.parquet, sp_{season}.parquet   CFBD's season-level adjusted EPA (WEPA) and SP+ (benchmarks)
    plays_{season}.parquet      play-by-play with CFBD's EPA (`ppa`)
    cache/, _calls.csv          raw responses and the call log (see client.py)

Calls: about five per season plus one per week of play-by-play. A finished week or season is
never requested twice.
"""
from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta

import pandas as pd

from ingest.cfb.client import BudgetExhausted, Client, cache_path
from sparvi.config import RAW, cfb

OUT = RAW / "cfb"
TYPES = ("regular", "postseason")
FINISHED_DAYS = 3        # a week counts as finished (cacheable) this long after its last game


def _now() -> datetime:
    return datetime.now(UTC)


def season_done(season: int) -> bool:
    return season < cfb().season or _now() > datetime(season + 1, 2, 15, tzinfo=UTC)


def requests_for(season: int, games: pd.DataFrame | None) -> list[tuple[str, dict, bool]]:
    """(endpoint, params, completed) for one season. Play-by-play weeks come from the games."""
    done = season_done(season)
    r = [("/teams/fbs", {"year": season}, done),
         # Season-level benchmarks (Tier 1): CFBD's opponent-adjusted EPA (WEPA) and SP+. Season totals
         # only, so they're compared on next-season games, never inside a season (no look-ahead).
         ("/wepa/team/season", {"year": season}, done),
         ("/ratings/sp", {"year": season}, done)]
    for t in TYPES:
        r += [("/games", {"year": season, "seasonType": t, "classification": "fbs"}, done),
              ("/lines", {"year": season, "seasonType": t}, done),
              ("/rankings", {"year": season, "seasonType": t}, done)]
    if games is not None and len(games):
        g = games.assign(start=pd.to_datetime(games["startDate"], utc=True))
        for (t, w), wk in g.groupby(["seasonType", "week"]):
            if wk["start"].min() > _now():
                continue                                    # nothing played yet
            finished = bool(wk["completed"].all()) and wk["start"].max() < _now() - timedelta(days=FINISHED_DAYS)
            r.append(("/plays", {"year": season, "week": int(w), "seasonType": t}, done or finished))
    return r


# Fixed numeric types: CFBD returns whole numbers as ints, so a season without missing values would
# otherwise come out BIGINT and one with gaps DOUBLE (the contract checks types).
FLOATS = {"games": ["homePoints", "awayPoints", "homePregameElo", "awayPregameElo", "homePostgameElo", "awayPostgameElo"],
          "rankings": ["points"],
          "lines": ["spread", "spreadOpen", "overUnder", "overUnderOpen", "homeMoneyline", "awayMoneyline"],
          "plays": ["ppa", "offenseScore", "defenseScore", "down", "distance", "yardline", "yardsToGoal", "yardsGained"]}


def _floats_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def _floats(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    for c in FLOATS[kind]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def _games_frame(raw: list[dict]) -> pd.DataFrame:
    keep = ["id", "season", "week", "seasonType", "startDate", "completed", "neutralSite", "conferenceGame",
            "homeId", "homeTeam", "homeConference", "homeClassification", "homePoints", "homePregameElo", "homePostgameElo",
            "awayId", "awayTeam", "awayConference", "awayClassification", "awayPoints", "awayPregameElo", "awayPostgameElo"]
    df = pd.DataFrame(raw)
    return df[[c for c in keep if c in df.columns]]


def _lines_frame(raw: list[dict]) -> pd.DataFrame:
    rows = [{"game_id": g["id"], "season": g["season"], "week": g["week"], "seasonType": g["seasonType"],
             "homeTeam": g["homeTeam"], "awayTeam": g["awayTeam"], **{k: ln.get(k) for k in
             ("provider", "spread", "spreadOpen", "overUnder", "overUnderOpen", "homeMoneyline", "awayMoneyline")}}
            for g in raw for ln in g.get("lines") or []]
    return pd.DataFrame(rows)


# The FBS polls (CFBD also returns FCS and Division II polls, which aren't used).
FBS_POLLS = ("AP Top 25", "Coaches Poll", "Playoff Committee Rankings")


def _rankings_frame(raw: list[dict]) -> pd.DataFrame:
    rows = [{"season": w["season"], "seasonType": w["seasonType"], "week": w["week"], "poll": p["poll"],
             "rank": r["rank"], "school": r["school"], "teamId": r.get("teamId"), "points": r.get("points")}
            for w in raw for p in w.get("polls", []) for r in p.get("ranks", [])]
    df = pd.DataFrame(rows)
    return df[df["poll"].isin(FBS_POLLS)].drop_duplicates(["season", "seasonType", "week", "poll", "school"]) if len(df) else df


PLAY_COLS = ["id", "gameId", "driveId", "driveNumber", "playNumber", "offense", "offenseConference", "defense",
             "defenseConference", "home", "away", "offenseScore", "defenseScore", "period", "down", "distance",
             "yardline", "yardsToGoal", "yardsGained", "playType", "ppa", "scoring"]


def pull_season(c: Client, season: int) -> None:
    get = lambda ep, prm, done: c.get(ep, prm, done)   # noqa: E731
    reqs = requests_for(season, None)
    teams = pd.DataFrame(get(*reqs[0]))
    wepa = pd.DataFrame([{"season": w["year"], "team": w["team"], "conference": w.get("conference"),
                          "epa": w["epa"]["total"], "epa_pass": w["epa"]["passing"], "epa_rush": w["epa"]["rushing"],
                          "epa_allowed": w["epaAllowed"]["total"], "epa_allowed_pass": w["epaAllowed"]["passing"],
                          "epa_allowed_rush": w["epaAllowed"]["rushing"]} for w in get(*reqs[1])])
    sp = pd.DataFrame([{"season": x["year"], "team": x["team"], "conference": x.get("conference"), "rating": x.get("rating"),
                        "offense": (x.get("offense") or {}).get("rating"), "defense": (x.get("defense") or {}).get("rating")}
                       for x in get(*reqs[2]) if x.get("team") != "nationalAverages"])
    games = pd.concat([_games_frame(get(ep, prm, done)) for ep, prm, done in reqs if ep == "/games"], ignore_index=True)
    lines = pd.concat([_lines_frame(get(ep, prm, done)) for ep, prm, done in reqs if ep == "/lines"], ignore_index=True)
    ranks = pd.concat([_rankings_frame(get(ep, prm, done)) for ep, prm, done in reqs if ep == "/rankings"], ignore_index=True)
    plays = []
    for ep, prm, done in requests_for(season, games):
        if ep == "/plays":
            df = pd.DataFrame(get(ep, prm, done))
            if len(df):
                df = df.reindex(columns=PLAY_COLS)
                df.insert(0, "season", season)
                df.insert(1, "week", prm["week"])
                df.insert(2, "seasonType", prm["seasonType"])
                plays.append(df)
    OUT.mkdir(parents=True, exist_ok=True)
    keep_teams = [c for c in ("id", "school", "mascot", "abbreviation", "conference", "division", "classification",
                              "color", "alternateColor") if c in teams.columns]
    teams[keep_teams].to_parquet(OUT / f"teams_{season}.parquet", index=False)
    games, lines, ranks = _floats(games, "games"), _floats(lines, "lines"), _floats(ranks, "rankings")
    games.sort_values(["seasonType", "week", "id"]).to_parquet(OUT / f"games_{season}.parquet", index=False)
    if len(wepa):
        wepa.sort_values("team").to_parquet(OUT / f"wepa_{season}.parquet", index=False)
    if len(sp):
        _floats_cols(sp, ["rating", "offense", "defense"]).sort_values("team").to_parquet(OUT / f"sp_{season}.parquet", index=False)
    lines.sort_values(["game_id", "provider"]).to_parquet(OUT / f"lines_{season}.parquet", index=False)
    ranks.sort_values(["seasonType", "week", "poll", "rank", "school"]).to_parquet(OUT / f"rankings_{season}.parquet", index=False)
    if plays:
        p = _floats(pd.concat(plays, ignore_index=True).drop_duplicates("id"), "plays").sort_values(["gameId", "driveNumber", "playNumber", "id"])
        p.to_parquet(OUT / f"plays_{season}.parquet", index=False)
    print(f"{season}: {len(teams)} FBS teams, {len(games):,} games, {len(lines):,} lines, {len(ranks):,} ranks, "
          f"{sum(map(len, plays)):,} plays (calls so far {c.calls}, left {c.remaining})")


def estimate(seasons: list[int]) -> int:
    """Uncached calls a pull needs. Weeks of play-by-play are counted from cached games, or ~16 a season."""
    n = 0
    for s in seasons:
        base = requests_for(s, None)
        n += sum(1 for ep, prm, done in base if not (done and cache_path(ep, prm).exists()))
        gpaths = [cache_path(ep, prm) for ep, prm, _ in base if ep == "/games"]
        if all(p.exists() for p in gpaths):
            import gzip
            import json
            g = pd.concat([_games_frame(json.loads(gzip.decompress(p.read_bytes()))) for p in gpaths], ignore_index=True)
            n += sum(1 for ep, prm, done in requests_for(s, g) if ep == "/plays" and not (done and cache_path(ep, prm).exists()))
        else:
            n += 16
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", action="store_true", help="only the current season")
    ap.add_argument("--estimate", action="store_true", help="count the calls needed and stop")
    ap.add_argument("--yes", action="store_true", help="go ahead above the approval line")
    a = ap.parse_args()
    cfg = cfb()
    seasons = [cfg.season] if a.current else cfg.seasons
    need = estimate(seasons)
    print(f"estimated uncached CFBD calls: {need} (approval line {cfg.approve_over}, reserve {cfg.call_reserve})")
    if a.estimate:
        return 0
    if need > cfg.approve_over and not a.yes:
        print("more than the approval line: get Matt's OK, then rerun with --yes")
        return 2
    c = Client()
    left = c.info().get("remainingCalls")
    c.remaining = left
    print(f"CFBD calls left this month: {left}")
    if left is not None and left - need < cfg.call_reserve:
        print(f"not enough quota for {need} calls above the {cfg.call_reserve}-call reserve; stopping")
        return 3
    try:
        for s in seasons:
            pull_season(c, s)
    except BudgetExhausted as e:
        print(e)
        return 3
    print(f"done: {c.calls} calls, {c.remaining} left this month")
    return 0


if __name__ == "__main__":
    sys.exit(main())
