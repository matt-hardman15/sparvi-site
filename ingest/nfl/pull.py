"""
Pull NFL play-by-play, schedules, teams, and team stats from nflverse (via nflreadpy).

Seasons come from sparvi.toml [nfl] (see sparvi.config). nflverse data is CC-BY 4.0; credit
"nflverse" on every page that uses it.

Usage:
    python ingest/nfl/pull.py              # every season in the config window (skips past seasons on file)
    python ingest/nfl/pull.py --current    # re-pull only the current season (+ schedules, teams)

Output (data/raw/nfl/, never committed):
    pbp_{season}.parquet          regular-season and postseason plays, the columns the pipeline reads
    team_stats_{season}.parquet   nflverse regular-season team stats (for the reconciliation check)
    schedules.parquet             every game in the window, with scores (team codes normalized)
    source/schedules.parquet, source/teams.parquet   as nflverse sent them: every column, original codes (snapshots)
    teams.parquet                 abbreviations, names, divisions, colors
"""
import argparse

import nflreadpy as nfl

from football.common.teams import NFL_ALIASES
from sparvi.config import RAW
from sparvi.config import nfl as nfl_config

CFG = nfl_config()
OUT = RAW / "nfl"

# Only what the pipeline reads; the full file has ~370 columns.
PBP_COLS = [
    "play_id", "game_id", "season", "week", "season_type", "game_date",
    "home_team", "away_team", "posteam", "defteam", "posteam_type",
    "qtr", "down", "ydstogo", "yardline_100", "play_type",
    "pass", "rush", "qb_dropback", "sack", "qb_scramble", "qb_kneel", "qb_spike",
    "pass_attempt", "rush_attempt", "complete_pass", "interception", "fumble_lost",
    "two_point_attempt", "penalty", "yards_gained", "air_yards",
    "epa", "qb_epa", "success", "wp", "vegas_wp", "cpoe", "xpass", "pass_oe",
    "posteam_score", "defteam_score", "home_score", "away_score", "result",
]


def save(df, path) -> None:
    pdf = df.to_pandas() if hasattr(df, "to_pandas") else df
    pdf.to_parquet(path, index=False)
    print(f"wrote {path.name}: {len(pdf):,} rows")


def pull_season(season: int) -> None:
    pbp = nfl.load_pbp([season]).select(PBP_COLS)
    save(pbp, OUT / f"pbp_{season}.parquet")
    save(nfl.load_team_stats([season], summary_level="reg"), OUT / f"team_stats_{season}.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", action="store_true", help="re-pull only the current season")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    for s in CFG.seasons:
        current = s == CFG.season
        if args.current and not current:
            continue
        if not current and (OUT / f"pbp_{s}.parquet").exists() and (OUT / f"team_stats_{s}.parquet").exists():
            print(f"skip {s} (on file)")
            continue
        pull_season(s)

    sched = nfl.load_schedules(CFG.seasons).to_pandas()
    teams = nfl.load_teams().to_pandas()
    # As the source sent them (every column, original team codes): what the snapshots record (A5a).
    # The schedules' betting lines are kept there for the future market benchmark (L3), never published.
    (OUT / "source").mkdir(exist_ok=True)
    save(sched, OUT / "source" / "schedules.parquet")
    save(teams, OUT / "source" / "teams.parquet")

    keep = ["game_id", "season", "game_type", "week", "gameday", "home_team", "away_team",
            "home_score", "away_score", "result", "overtime", "location"]
    sched = sched[keep].replace({"home_team": NFL_ALIASES, "away_team": NFL_ALIASES})
    save(sched, OUT / "schedules.parquet")
    save(teams[~teams["team_abbr"].isin(list(NFL_ALIASES))], OUT / "teams.parquet")  # the 32 current codes


if __name__ == "__main__":
    main()
