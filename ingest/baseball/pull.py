"""
Pull pitch-level Statcast data (MLB + Triple-A) from Baseball Savant and save as parquet.

Seasons and date windows come from sparvi.toml (see sparvi.config).

Usage:
    uv sync
    python ingest/baseball/pull.py                    # every season in the config window
    python ingest/baseball/pull.py --current --refresh  # re-pull the current season
    python ingest/baseball/pull.py --current --incremental  # just the new games (+ a 3-day overlap)
    python ingest/baseball/pull.py --seasons 2025     # just one season

Output: data/raw/statcast_{level}_{season}.parquet
Re-running skips any season/level file that already exists.
"""
import argparse
import io
import time
from datetime import date, timedelta

import pandas as pd
import requests

from sparvi.config import RAW, baseball

CFG = baseball()
RAW.mkdir(parents=True, exist_ok=True)

MLB_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
MILB_URL = "https://baseballsavant.mlb.com/statcast-search-minors/csv"

# Regular-season windows live in sparvi.toml [baseball.season_windows].
SEASON_WINDOWS = CFG.season_windows

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


KEY = ["game_pk", "at_bat_number", "pitch_number"]
ORDER = ["game_date", "game_pk", "at_bat_number", "pitch_number"]
# Incremental pulls re-fetch the last few days already on file: Savant revises recent games (pitch
# types, expected stats) for a day or two after they're played. A weekly full re-pull (in the
# workflow) catches anything older.
OVERLAP_DAYS = 3


def fetch_range(level: str, start: date, end: date) -> pd.DataFrame:
    """Every regular-season pitch at `level` from start to end (inclusive), in Savant-sized chunks."""
    url, lvl = (MLB_URL, None) if level == "MLB" else (MILB_URL, level)
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
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """One row per pitch (the latest version wins) in a fixed order, so full and incremental pulls
    of the same games produce the same file."""
    return df.drop_duplicates(subset=KEY, keep="last").sort_values(ORDER, kind="stable").reset_index(drop=True)


def pull_season(season: int, level: str) -> None:
    out = RAW / f"statcast_{level.lower()}_{season}.parquet"
    if out.exists():
        print(f"skip {out.name} (exists)")
        return
    start, end = (date.fromisoformat(d) for d in SEASON_WINDOWS[season])
    full = fetch_range(level, start, end)
    if full.empty:
        print(f"WARNING: no rows for {level} {season}")
        return
    full = tidy(full)
    full.to_parquet(out, index=False)
    print(f"wrote {out.name}: {len(full):,} pitches, {full['pitcher'].nunique():,} pitchers")


def update_season(season: int, level: str, today: date | None = None) -> None:
    """Incremental: re-pull from OVERLAP_DAYS before the last game on file to today (or the season's
    end), replace those days, and append anything newer. Falls back to a full pull with no file."""
    out = RAW / f"statcast_{level.lower()}_{season}.parquet"
    if not out.exists():
        print(f"no {out.name} yet: full pull")
        return pull_season(season, level)
    old = pd.read_parquet(out)
    win_start, win_end = (date.fromisoformat(d) for d in SEASON_WINDOWS[season])
    last = date.fromisoformat(str(old["game_date"].max())[:10])
    start = max(win_start, last - timedelta(days=OVERLAP_DAYS))
    end = min(win_end, today or date.today())
    if start > end:
        print(f"{out.name}: up to date (last game {last})")
        return
    new = fetch_range(level, start, end)
    if new.empty:
        # Nothing at all, even for days already on file: keep what we have rather than delete it.
        print(f"{out.name}: Savant returned no rows for {start}..{end}; kept the file as is")
        return
    kept = old[old["game_date"].astype(str).str[:10] < start.isoformat()]
    merged = tidy(pd.concat([kept, new], ignore_index=True))
    merged.to_parquet(out, index=False)
    print(f"updated {out.name}: re-pulled {start}..{end} ({len(new):,} pitches); "
          f"{len(old):,} -> {len(merged):,} pitches, last game {merged['game_date'].max()}")


def pull_milb_rosters(season: int) -> None:
    """Player ids -> name, org, level, age from the MLB Stats API (for the PD report cards)."""
    out = RAW / f"milb_players_{season}.parquet"
    if out.exists():
        print(f"skip {out.name} (exists)")
        return
    rows = []
    for sport_id, lvl in [(1, "MLB"), (11, "AAA"), (12, "AA"), (13, "A+"), (14, "A")]:
        r = requests.get(
            f"https://statsapi.mlb.com/api/v1/sports/{sport_id}/players",
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


def pull_affiliates(season: int) -> None:
    """Triple-A club -> parent MLB org for one season (data/raw/aaa_affiliates_SEASON.parquet).

    Farm-system rankings attribute each Triple-A player-season to an org through this. Same schema as
    ingest/baseball/pull_rosters.py writes for the current season; skipped if the file exists.
    """
    out = RAW / f"aaa_affiliates_{season}.parquet"
    if out.exists():
        print(f"skip {out.name} (exists)")
        return
    api = "https://statsapi.mlb.com/api/v1/teams"
    orgs = {t["id"]: t["abbreviation"]
            for t in requests.get(api, params={"sportId": 1, "season": season}, timeout=60).json()["teams"]}
    rows = [{"season": season, "team_id": t["id"], "team": t["abbreviation"], "team_name": t["name"],
             "org_id": t.get("parentOrgId"), "org": orgs.get(t.get("parentOrgId"))}
            for t in requests.get(api, params={"sportId": 11, "season": season}, timeout=60).json()["teams"]]
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {out.name}: {len(rows)} clubs")


def pull_heights() -> None:
    """Listed height for every batter in the raw files (data/raw/batter_heights.parquet).

    Location+ measures pitch height against the rulebook zone (27% to 53.5% of the batter's height,
    the definition Statcast uses from 2026). Earlier seasons recorded a stance-based zone per pitch,
    so the height lets every season share one definition. Only batters not already on file are
    fetched, so this takes seconds.
    """
    import re
    out = RAW / "batter_heights.parquet"
    have = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["batter", "height_in"])
    files = sorted(RAW.glob("statcast_*_*.parquet"))
    if not files:
        return
    ids = pd.concat([pd.read_parquet(f, columns=["batter"]) for f in files])["batter"].dropna().astype(int).unique()
    need = sorted(set(ids) - set(have["batter"].astype(int)))
    rows = []
    for i in range(0, len(need), 100):
        r = requests.get("https://statsapi.mlb.com/api/v1/people",
                         params={"personIds": ",".join(map(str, need[i:i + 100]))}, timeout=60)
        r.raise_for_status()
        for person in r.json().get("people", []):
            m = re.match(r"(\d+)' (\d+)\"", person.get("height") or "")
            rows.append({"batter": person["id"], "height_in": int(m[1]) * 12 + int(m[2]) if m else None})
        time.sleep(0.3)
    allh = pd.concat([have, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("batter", keep="last")
    allh["batter"] = allh["batter"].astype("int64")
    allh.sort_values("batter").to_parquet(out, index=False)
    print(f"batter heights: {len(allh):,} on file ({len(rows):,} new), {int(allh['height_in'].isna().sum())} unknown")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=CFG.mlb_seasons)
    ap.add_argument("--current", action="store_true", help=f"just the current season ({CFG.season})")
    ap.add_argument("--levels", nargs="+", default=["MLB", "AAA"])
    ap.add_argument("--refresh", action="store_true",
                    help="re-pull the given seasons even if files exist (use for the current season)")
    ap.add_argument("--incremental", action="store_true",
                    help=f"only fetch new games (plus the last {OVERLAP_DAYS} days on file) for the given seasons")
    args = ap.parse_args()
    if args.current:
        args.seasons = [CFG.season]
    if args.refresh:
        for s in args.seasons:
            for f in RAW.glob(f"*_{s}.parquet"):
                f.unlink()
    for s in args.seasons:
        if args.incremental:  # rosters are small and change daily in season: always re-pull them
            (RAW / f"milb_players_{s}.parquet").unlink(missing_ok=True)
        pull_milb_rosters(s)
        for lvl in args.levels:
            if lvl == "AAA" and s not in CFG.aaa_seasons:
                continue
            if lvl == "AAA":
                pull_affiliates(s)
            (update_season if args.incremental else pull_season)(s, lvl)
    pull_heights()
