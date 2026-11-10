#!/usr/bin/env bash
# Full rebuild: models -> pipeline -> exported JSON -> site in docs/.
# Assumes data/raw/ is populated (python scripts/pull_data.py) and Quarto is installed.
set -euo pipefail
cd "$(dirname "$0")"

# The pipeline's "today" is the last game in the data, so off-season builds still show
# a meaningful trailing window. During the season this is yesterday's games.
AS_OF=$(python -c "import duckdb; print(duckdb.sql(\"select max(game_date) from 'data/raw/statcast_aaa_2026.parquet'\").fetchone()[0])")
echo "as_of_date = $AS_OF"

python baseball/pitch_quality/train.py
python baseball/pitch_quality/robustness.py
python baseball/shape_stability/analysis.py
(cd baseball/pd_pipeline && DBT_PROFILES_DIR=. dbt build --vars "{as_of_date: '$AS_OF'}")
python baseball/two_week_signal/analysis.py
python baseball/export_site_data.py
(cd site && quarto render)
echo "site built in docs/"
