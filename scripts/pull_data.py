"""
Pull pitch-level Statcast data (MLB + Triple-A) from Baseball Savant and save as parquet.

Usage:
    pip install -r requirements.txt
    python scripts/pull_data.py                 # MLB 2024-2026 + AAA 2025-2026
    python scripts/pull_data.py --seasons 2026 --refresh   # re-pull the current season
    python scripts/pull_data.py --seasons 2026  # just one season

Output: data/raw/statcast_{level}_{season}.parquet
Re-running skips any season/level file that already exists.
"""
import argparse
import io
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

MLB_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
MILB_URL = "https://baseballsavant.mlb.com/statcast-search-minors/csv"

# Regular season windows (generous; empty days just return nothing)
SEASON_WINDOWS = {
    2024: ("2024-03-20", "2024-09-30"),
    2025: ("2025-03-18", "2025-09-28"),
    2026: ("2026-03-25", "2026-09-27"),
    # Add the next season's window here at rollover (opening day to end of regular season).
}

KEEP_COLS = [
    "game_pk", "game_date", "game_year", "game_type", "home_team", "away_team",
    "pitcher", "batter", "player_name", "p_throws", "stand",
    "at_bat_number", "pitch_number", "inning", "inning_topbot",
    "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b",
    "pitch_type", "pitch_name", "release_speed", "release_spin_rate", "spin_axis",
    "release_pos_x", "release_pos_y", "release_pos_z", "release_extension", "arm_angle",
    "pfx_x", "pfx_z", "plate_x", "plate_z", "sz_top", "sz_bot", "zone",
    "vx0", "vy0", "vz0", "ax", "ay", "az",
    "description", "type", "events", "bb_type",
    "launch_speed", "launch_angle", "hit_distance_sc", "bat_speed", "swing_length",
    "estimated_woba_using_speedangle", "estimated_ba_using_speedangle",
    "woba_value", "woba_denom", "delta_run_exp", "delta_pitcher_run_exp",
]


def fetch_window(url: str, start: str, end: str, level: str | None) -> pd.DataFrame:
    params = {
        "all": "true",
        "type": "details",
        "player_type": "pitcher",
        "game_date_gt": start,
        "game_date_lt": end,
        "hfGT": "R|",
    }
    if level:
        params["hfLevel"] = f"{level}|"
        params["minors"] = "true"
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            if not r.text.strip() or r.text.lstrip().startswith("<"):
                return pd.DataFrame()
            return pd.read_csv(io.StringIO(r.text), low_memory=False)
        except Exception as e:  # noqa: BLE001
            wait = 5 * (attempt + 1)
            print(f"    retry {attempt + 1} for {start}..{end}: {e} (sleeping {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"failed after retries: {start}..{end}")


def pull_season(season: int, level: str) -> None:
    out = RAW / f"statcast_{level.lower()}_{season}.parquet"
    if out.exists():
        print(f"skip {out.name} (exists)")
        return
    url, lvl = (MLB_URL, None) if level == "MLB" else (MILB_URL, level)
    start, end = (date.fromisoformat(d) for d in SEASON_WINDOWS[season])
    frames = []
    day = start
    while day <= end:
        # Savant caps rows per request, so pull 3-day chunks for MLB, 5 for AAA
        step = 3 if level == "MLB" else 5
        chunk_end = min(day + timedelta(days=step - 1), end)
        df = fetch_window(url, day.isoformat(), chunk_end.isoformat(), lvl)
        if len(df):
            frames.append(df[[c for c in KEEP_COLS if c in df.columns]])
        print(f"  {level} {day}..{chunk_end}: {len(df):,} pitches")
        day = chunk_end + timedelta(days=1)
        time.sleep(1.0)
    if not frames:
        print(f"WARNING: no rows for {level} {season}")
        return
    full = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["game_pk", "at_bat_number", "pitch_number"]
    )
    full.to_parquet(out, index=False)
    print(f"wrote {out.name}: {len(full):,} pitches, {full['pitcher'].nunique():,} pitchers")


def pull_milb_rosters(season: int) -> None:
    """Player ids -> name, org, level, age from the MLB Stats API (for the PD report cards)."""
    out = RAW / f"milb_players_{season}.parquet"
    if out.exists():
        print(f"skip {out.name} (exists)")
        return
    rows = []
    for sport_id, lvl in [(1, "MLB"), (11, "AAA"), (12, "AA"), (13, "A+"), (14, "A")]:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/sports/{}/players".format(sport_id),
            params={"season": season, "hydrate": "currentTeam"},
            timeout=60,
        )
        r.raise_for_status()
        for p in r.json().get("people", []):
            team = p.get("currentTeam", {})
            rows.append({
                "player_id": p["id"],
                "name": p.get("fullName"),
                "birth_date": p.get("birthDate"),
                "position": p.get("primaryPosition", {}).get("abbreviation"),
                "bats": p.get("batSide", {}).get("code"),
                "throws": p.get("pitchHand", {}).get("code"),
                "level": lvl,
                "team_id": team.get("id"),
                "parent_org_id": team.get("parentOrgId", team.get("id")),
            })
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {out.name}: {len(rows):,} player-level rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--levels", nargs="+", default=["MLB", "AAA"])
    ap.add_argument("--refresh", action="store_true",
                    help="re-pull the given seasons even if files exist (use for the current season)")
    args = ap.parse_args()
    if args.refresh:
        for s in args.seasons:
            for f in RAW.glob(f"*_{s}.parquet"):
                f.unlink()
    for s in args.seasons:
        pull_milb_rosters(s)
        for lvl in args.levels:
            if lvl == "AAA" and s < 2025:
                continue
            pull_season(s, lvl)
