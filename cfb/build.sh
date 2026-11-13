#!/usr/bin/env bash
# College football rebuild: dbt (football project, cfb_* models) -> ratings, backtest, power rating
# -> AP-poll test -> exported JSON -> thumbnail. Assumes data/raw/cfb/ is populated
# (python ingest/cfb/pull.py) and the venv is active (or `uv run cfb/build.sh`). Seasons come from
# sparvi.toml [cfb]. Rendering is separate (root build.sh or the workflow). ~10 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."

SEASON=$(python -m sparvi.config cfb.season)
FIRST=$(python -m sparvi.config cfb.first_season)
NFL_SEASON=$(python -m sparvi.config nfl.season)
NFL_FIRST=$(python -m sparvi.config nfl.first_season)
echo "cfb seasons $FIRST-$SEASON"
# One dbt project for both football sports: only the CFB models (and the score check, which reads raw sources).
(cd football/dbt && DBT_PROFILES_DIR=. dbt build --select stg_cfb_plays+ cfb_reconcile_scores \
  --vars "{nfl_season: $NFL_SEASON, nfl_first_season: $NFL_FIRST, cfb_season: $SEASON, cfb_first_season: $FIRST}")
python cfb/adjusted_ratings/adjusted.py
python cfb/adjusted_ratings/polls.py
python cfb/export_site_data.py
python cfb/thumb.py
