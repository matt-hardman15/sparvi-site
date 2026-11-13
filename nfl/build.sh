#!/usr/bin/env bash
# NFL rebuild: dbt (football project) -> opponent adjustment -> exported JSON -> static team pages.
# Assumes data/raw/nfl/ is populated (python ingest/nfl/pull.py) and the venv is active (or this
# runs under `uv run nfl/build.sh`). Seasons come from sparvi.toml [nfl]. Rendering is separate:
# the root build.sh (or the workflow) runs `quarto render` once for every sport.
set -euo pipefail
cd "$(dirname "$0")/.."

SEASON=$(python -m sparvi.config nfl.season)
FIRST=$(python -m sparvi.config nfl.first_season)
echo "nfl seasons $FIRST-$SEASON"
CFB_SEASON=$(python -m sparvi.config cfb.season)
CFB_FIRST=$(python -m sparvi.config cfb.first_season)
# One dbt project for both football sports: build only the NFL models (everything downstream of the
# NFL plays). Both sports' season vars are passed because dbt parses every model.
(cd football/dbt && DBT_PROFILES_DIR=. dbt build --select stg_nfl_plays+ nfl_reconcile_team_stats   --vars "{nfl_season: $SEASON, nfl_first_season: $FIRST, cfb_season: $CFB_SEASON, cfb_first_season: $CFB_FIRST}")
python nfl/team_efficiency/adjusted.py
python nfl/export_site_data.py
python nfl/team_pages.py
