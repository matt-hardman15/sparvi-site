"""Every 4th-down decision of the backtest seasons and the current one, valued by the decision engine (F4).

Season t is decided by an engine fit on seasons FIRST..t-1 as of Sept 1 of t (walk-forward, the backtest's folds), so
no decision is judged by a model that saw its season. For each decision: every choice's win probability, the
go-minus-best-kick margin with a 90% interval from engines refit on resampled games, the call, and what the actual
choice cost.

    call        "go" / "kick" when the interval of the margin is entirely on one side of zero; otherwise "toss-up"
    cost        win probability of the best choice minus that of the actual one, counted only on clear calls
                (a toss-up costs nothing by definition)

The newest engine also fills a chart of standard situations (GRID: a tie game with 10:00 left in the 2nd quarter,
outdoors, even teams, all timeouts, averaged over home and away) for every field position and distance, with the
same intervals: the "when should you go" map.

Writes output/decisions_{season}.parquet (play-level, private: gitignored), output/grid_{season}.json, and
output/decisions.json (the engine's tables and run settings per season). nfl/export_site_data.py turns the rows into the site's aggregates.

    python -m nfl.fourth.decisions                 # every season; reuses a season's file when its inputs are unchanged
    python -m nfl.fourth.decisions --boot 50 --seasons 2025 2026
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.asof import Loader
from nfl.fourth import decide as D
from nfl.fourth.data import FIRST_SEASON
from sparvi.config import RAW, nfl

OUT = Path(__file__).resolve().parent / "output"
FIRST_DECIDED = 2020           # the backtest's first test season
BOOT = 50
KEEP = ["game_id", "play_id", "season", "week", "season_type", "game_date", "posteam", "defteam", "home_team", "away_team",
        "ydstogo", "yardline_100", "score_differential", "game_seconds_remaining", "half_seconds_remaining", "game_half",
        "choice"]


def _fingerprint(season: int, boot: int) -> str:
    """Code and raw inputs behind one season's file: the engine and model code, and the raw files it reads."""
    h = hashlib.sha256(f"{season}|{boot}".encode())
    here = Path(__file__).resolve().parent
    for f in [here / "decide.py", here / "decisions.py", here / "models.py", here / "data.py", here / "fg.py"]:
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    for s in range(FIRST_SEASON, season + 1):
        for f in (RAW / "nfl" / f"pbp_{s}.parquet", RAW / "nfl" / "state" / f"pbp_state_{s}.parquet"):
            h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


GRID_YARDLINES = list(range(1, 100))
GRID_YTG = list(range(1, 16))


def grid(season: int) -> pd.DataFrame:
    """Standard situations: every yards-from-goal x yards-to-go pair, as the home and as the away team."""
    g = pd.DataFrame([(y, t, h) for y in GRID_YARDLINES for t in GRID_YTG if t <= y for h in (1.0, 0.0)],
                     columns=["yardline_100", "ydstogo", "home"]).astype(float)
    return g.assign(season=season, score_differential=0.0, game_seconds_remaining=2700.0, half_seconds_remaining=900.0,
                    half2=0.0, ot=0.0, down=4.0, posteam_timeouts_remaining=3.0, defteam_timeouts_remaining=3.0,
                    recv_2h=0.5, strength=0.0, indoors=0.0, wind=5.0)


def _grid_table(g: pd.DataFrame, r: pd.DataFrame, b: np.ndarray) -> list[dict]:
    """Home and away rows averaged (within each refit, so the interval is of the average)."""
    key = [g["yardline_100"].astype(int), g["ydstogo"].astype(int)]
    cols = ["wp_go", "wp_fg", "wp_punt", "p_convert", "p_fg", "punt_net", "margin"]
    t = r[cols].groupby(key).mean()
    bm = pd.DataFrame(b.T).groupby([k.to_numpy() for k in key]).mean()
    t["margin_lo"], t["margin_hi"] = np.percentile(bm, 5, axis=1), np.percentile(bm, 95, axis=1)
    t["call"] = np.where(t["margin_lo"] > 0, "go", np.where(t["margin_hi"] < 0, "kick", "toss-up"))
    t["kick"] = np.where(t["wp_fg"].fillna(-1) >= t["wp_punt"].fillna(-1), "fg", "punt")
    t = t.rename_axis(["yardline_100", "ydstogo"]).reset_index()
    num = [c for c in t.columns if t[c].dtype.kind == "f"]
    t[num] = t[num].round(4)
    return json.loads(t.to_json(orient="records"))


def season_decisions(loader: Loader, season: int, boot: int, today: pd.Timestamp) -> tuple[pd.DataFrame, dict, list[dict]]:
    eng, train = D.engine_for(loader, season, FIRST_SEASON, f"{season}-09-01")
    s = D.situations(D.plays(loader, [season], today))
    g = grid(season)
    r_all = eng.decide(pd.concat([s[D.STATE], g[D.STATE]], ignore_index=True))      # one pass, one set of refits
    b_all = D.bootstrap(train, pd.concat([s[D.STATE], g[D.STATE]], ignore_index=True), boot)
    r, b = r_all.iloc[:len(s)].reset_index(drop=True), b_all[:, :len(s)]
    grid_rows = _grid_table(g, r_all.iloc[len(s):].reset_index(drop=True), b_all[:, len(s):])
    out = pd.concat([s[KEEP], r], axis=1)
    out["margin_lo"], out["margin_hi"] = np.percentile(b, 5, axis=0), np.percentile(b, 95, axis=0)
    out["call"] = np.where(out["margin_lo"] > 0, "go", np.where(out["margin_hi"] < 0, "kick", "toss-up"))
    out["wp_actual"] = eng.chosen(s, r)
    wrong = ((out["call"] == "go") & (out["choice"] != "go")) | ((out["call"] == "kick") & (out["choice"] == "go"))
    out["cost"] = np.where(wrong, (np.maximum(out["wp_go"], out["wp_kick"]) - out["wp_actual"]).clip(lower=0), 0.0)
    info = {"train_seasons": list(eng.seasons), "decisions": int(len(out)), "bootstrap": boot,
            "p_block": eng.p_block, "fg_snap": eng.fg_snap, "fg_max_yardline": eng.fg_max, "runoff": eng.runoff, "kick_start": eng.kick_start,
            "calls": {k: int(v) for k, v in out["call"].value_counts().sort_index().items()}}
    return out, info, grid_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--seasons", nargs="+", type=int, default=list(range(FIRST_DECIDED, nfl().season + 1)))
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    meta_path = OUT / "decisions.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"seasons": {}}
    loader, today = Loader(), pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    for t in a.seasons:
        fp, f = _fingerprint(t, a.boot), OUT / f"decisions_{t}.parquet"
        if f.exists() and meta["seasons"].get(str(t), {}).get("inputs") == fp:
            print(f"{t}: unchanged")
            continue
        out, info, grid_rows = season_decisions(loader, t, a.boot, today)
        out.to_parquet(f, index=False)
        (OUT / f"grid_{t}.json").write_text(json.dumps(grid_rows), encoding="utf-8", newline="\n")
        meta["seasons"][str(t)] = info | {"inputs": fp}
        # Written after every season, so a stopped run keeps what it finished.
        meta["seasons"] = dict(sorted(meta["seasons"].items()))
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")
        print(f"{t}: {info['decisions']:,} decisions, calls {info['calls']}", flush=True)


if __name__ == "__main__":
    main()
