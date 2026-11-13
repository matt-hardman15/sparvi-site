#!/usr/bin/env bash
# NFL rebuild: dbt (football project) -> opponent adjustment -> exported JSON -> static team pages.
# Assumes data/raw/nfl/ is populated (python scripts/pull_nfl.py) and the venv is active (or this
# runs under `uv run nfl/build.sh`). Seasons come from sparvi.toml [nfl]. Rendering is separate:
# the root build.sh (or the workflow) runs `quarto render` once for every sport.
set -euo pipefail
cd "$(dirname "$0")/.."

SEASON=$(python -m sparvi.config nfl.season)
FIRST=$(python -m sparvi.config nfl.first_season)
echo "nfl seasons $FIRST-$SEASON"
(cd football/dbt && DBT_PROFILES_DIR=. dbt build --vars "{nfl_season: $SEASON, nfl_first_season: $FIRST}")
python nfl/team_efficiency/adjusted.py
python nfl/export_site_data.py
python nfl/team_pages.py
