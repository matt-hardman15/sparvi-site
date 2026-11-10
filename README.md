# Sparvi

Free, independent sports analytics: findings meant to change a decision, with the data
behind them in explorers readers can filter and check. Live at **https://sparvi.io**.

## What's here

| Path | |
|---|---|
| `scripts/pull_data.py` | Pulls pitch-level Statcast (MLB, Triple-A) and MLB Stats API rosters into `data/raw/` (not committed) |
| `baseball/pitch_quality/` | Stuff+ model (LightGBM): `train.py`, `robustness.py` |
| `baseball/shape_stability/` | Do midseason pitch shape changes persist into next season? |
| `baseball/pd_pipeline/` | dbt + DuckDB player development pipeline: windows, percentiles, change flags |
| `baseball/two_week_signal/` | Which two-week Triple-A numbers carry into the next two weeks |
| `baseball/export_site_data.py` | Writes aggregated JSON for the site (no pitch-level data is published) |
| `site/` | Quarto site. Interactive charts use Observable Plot and d3, self-hosted in `site/assets/js/` |

## Build locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/pull_data.py          # first run ~30-60 min
./build.sh                           # models -> dbt -> JSON -> docs/
cd docs && python -m http.server 8000   # http://localhost:8000
```

Needs the [Quarto CLI](https://quarto.org/docs/get-started/). On Windows, run `build.sh` from Git Bash.

## Publishing

`.github/workflows/site.yml` deploys to GitHub Pages:

- **Every push to `main`** renders the site from the committed JSON in `site/data/` and deploys (a few minutes).
- **Daily, April through October**, it re-pulls the current season, rebuilds every model and the dbt
  pipeline, commits the refreshed JSON, and deploys. Run it by hand from the Actions tab
  ("Run workflow") at any time.

## Adding a sport

Each sport gets its own top-level folder (`basketball/`, `football/`, …) with its own pull,
models, and an `export_site_data.py` that writes to `site/data/<sport>/`. Pages go in
`site/<sport>/` and reuse `site/assets/js/sparvi.js` (tables, charts, percentile bars, controls)
and `site/assets/css/sparvi.css`, so every sport looks like one site.

## Season rollover (each spring)

1. Add the new season's date window to `SEASON_WINDOWS` in `scripts/pull_data.py`.
2. Bump the season in `baseball/pitch_quality/train.py` (train/holdout years),
   `baseball/pd_pipeline/dbt_project.yml` (`season`), `baseball/export_site_data.py` (`SEASON`),
   `build.sh`, and the `--seasons` value in the workflow.
3. Run the workflow manually once.
