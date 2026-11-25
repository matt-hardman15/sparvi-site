"""Today's college basketball forecasts into the forecast ledger, and yesterday's results resolved (CBB-DAILY; run
daily by .github/workflows/cbb.yml after the pull, on main only: the ledger writer refuses anywhere else).

One line per game, written once, before its tip-off: the forecast home margin with an 80% interval. The home win
probability and the total are in `detail` (a ledger line is a point with an interval or a probability, not both), and
the baselines' margins are in `baselines`: the ratings alone, last season's ratings and home court. The source's Elo
baseline is not written: the ledger is public, and a per-game number that is a fixed line through the source's own
rating would hand that rating out game by game. It is scored later, in aggregate, from the private tables. A game
already in the ledger is never forecast again, whatever changed since.

Resolution: a forecast whose game is final resolves with the home margin (the score is kept as `sample`); a cancelled
game is voided. A postponed game stays open until it is played or cancelled.

Games already played when the job first ran are not in the ledger: forecasts start the day it shipped.

    python -m cbb.ratings.freeze               # in Actions: write to the ledger
    python -m cbb.ratings.freeze --dry-run     # anywhere: print what would be written, write nothing
"""
from __future__ import annotations

import sys

import pandas as pd

from cbb.ratings import daily as D
from cbb.ratings import model as M

SPORT, MODEL, TARGET = "cbb", "cbb_team_rating", "home_margin"


def forecast_id(season: int, game_id: int) -> str:
    return f"cbb-game-{season}-{int(game_id)}"


def records(f: pd.DataFrame, season: int, version: str, existing: set[str], now: pd.Timestamp) -> list[dict]:
    """Ledger records for the games of `f` (daily.day's forecasts) that have not tipped off at `now` (UTC) and are not
    in the ledger yet."""
    out = []
    for r in f.itertuples():
        fid = forecast_id(season, r.game_id)
        if fid in existing or r.start <= now or r.status != "scheduled":
            continue
        base = {"ratings": round(float(r.ratings_margin), 2), "prior": round(float(r.prior_margin), 2), "home": round(float(r.home_margin), 2)}
        vs = "vs." if r.neutral else "at"
        out.append({
            "forecast_id": fid, "model": MODEL, "model_version": version, "entity_type": "game", "entity_id": int(r.game_id),
            "entity_name": f"{r.away_name} {vs} {r.home_name}", "target": TARGET, "horizon": "game",
            # A scored resolution needs this date to have come: the UTC day after tip-off, when the result is in the pull.
            "resolves_after": str((r.start.tz_convert("UTC").normalize() + pd.Timedelta(days=1)).date()), "min_sample": None,
            "prob": None, "point": round(float(r.margin), 2), "lo80": round(float(r.margin_lo80), 2), "hi80": round(float(r.margin_hi80), 2),
            "baselines": base, "deserved_target": None,
            "detail": {"kind": "game", "game_day": str(r.date.date()), "tipoff": r.start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "home_id": int(r.home), "away_id": int(r.away), "home": r.home_name, "away": r.away_name, "neutral": bool(r.neutral),
                       "game_type": r.game_type, "p_home": round(float(r.p_home), 4), "total": round(float(r.total), 1)}})
    return out


def resolutions(forecasts: list[dict], resolved: set[str], games: pd.DataFrame, today) -> list[dict]:
    """Resolutions for open forecasts: scored once the game is final and its `resolves_after` has come; void if cancelled."""
    g = games.drop_duplicates("id").set_index("id")
    out = []
    for fc in forecasts:
        fid = fc["forecast_id"]
        if fid in resolved or (fc.get("detail") or {}).get("kind") != "game" or fc["entity_id"] not in g.index:
            continue
        row = g.loc[fc["entity_id"]]
        if row["status"] == "cancelled":
            out.append({"forecast_id": fid, "status": "void", "void_reason": "the game was cancelled"})
        elif row["status"] == "final" and pd.notna(row["homePoints"]) and pd.notna(row["awayPoints"]) and str(today) >= fc["resolves_after"]:
            h, a = int(row["homePoints"]), int(row["awayPoints"])
            out.append({"forecast_id": fid, "status": "scored", "actual": h - a, "sample": f"{h}-{a}"})
    return out


def main(argv: list[str] | None = None) -> int:
    from backtest.asof import Loader
    from sparvi import ledger, registry
    from sparvi.config import cbb
    dry = "--dry-run" in (sys.argv[1:] if argv is None else argv)
    cfg = cbb()
    season, earlier = cfg.season, cfg.history_seasons
    now = pd.Timestamp.now(tz="UTC")
    on = (now.tz_localize(None) - M.UTC_TO_GAME_DAY).normalize()
    loader = Loader()
    f, ratings = D.day(loader, on, season, earlier)
    fc, res = ledger.read(SPORT, season)
    recs = records(f, season, registry.get(MODEL).version, {r["forecast_id"] for r in fc}, now) if len(f) else []
    games = loader.load(f"cbb_games_{season}", on + pd.Timedelta(days=2), columns=["id", "status", "homePoints", "awayPoints"])
    done = resolutions(fc, {r["forecast_id"] for r in res}, games, now.date())
    print(f"{on.date()}: {len(f)} games scheduled between Division I teams, {len(ratings)} teams rated; "
          f"{len(recs)} forecasts to write, {len(done)} resolutions ({sum(r['status'] == 'void' for r in done)} void)")
    if dry:
        return 0
    n = ledger.append_forecasts(SPORT, season, recs) if recs else 0
    m = ledger.append_resolutions(SPORT, season, done) if done else 0
    print(f"ledger: {n} forecasts and {m} resolutions written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
