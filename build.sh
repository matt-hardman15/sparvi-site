#!/usr/bin/env bash
# Full rebuild: models -> pipeline -> exported JSON -> site in docs/.
# Assumes data/raw/ is populated (python scripts/pull_data.py), the venv is active or this
# runs under `uv run ./build.sh`, and Quarto is installed. Seasons come from sparvi.toml.
set -euo pipefail
cd "$(dirname "$0")"

SEASON=$(python -m sparvi.config baseball.season)
# The pipeline's "today" is the last game in the data, so off-season builds still show
# a meaningful trailing window. During the season this is yesterday's games.
last_game() { python -c "import duckdb; print(duckdb.sql(\"select max(game_date)::date from 'data/raw/statcast_$1_${SEASON}.parquet'\").fetchone()[0])"; }
# Each level has its own as-of: Triple-A and MLB regular seasons end on different days.
AS_OF=$(last_game aaa)
AS_OF_MLB=$(last_game mlb)
echo "season = $SEASON, as_of_date = $AS_OF (Triple-A), $AS_OF_MLB (MLB)"

python baseball/pitch_quality/train.py
python baseball/pitch_quality/robustness.py
python baseball/shape_stability/analysis.py
(cd baseball/pd_pipeline && DBT_PROFILES_DIR=. dbt build --vars "{season: $SEASON, as_of_date: '$AS_OF', as_of_date_mlb: '$AS_OF_MLB'}")
python baseball/two_week_signal/analysis.py
python baseball/export_site_data.py
(cd site && quarto render)
# Quarto renders a missing prose number as "?meta:..." instead of failing. Fail here instead.
if grep -rl "?meta:" docs --include=*.html; then echo "ERROR: unresolved prose numbers (see files above)"; exit 1; fi
echo "site built in docs/"
