# Sparvi

Free, independent sports analytics: findings meant to change a decision, with the data
behind them in explorers readers can filter and check. Live at **https://sparvi.io**.

## What's here

| Path | |
|---|---|
| `sparvi.toml` | The current season and season windows: the only place a season is set |
| `sparvi/` | Shared Python package: config, site-data writers with schema checks, provenance, plot style |
| `scripts/pull_data.py` | Pulls pitch-level Statcast (MLB, Triple-A) and MLB Stats API rosters into `data/raw/` (not committed) |
| `baseball/pitch_quality/` | Stuff+ model (LightGBM): `train.py`, `robustness.py` |
| `baseball/shape_stability/` | Do midseason pitch shape changes persist into next season? |
| `baseball/pd_pipeline/` | dbt + DuckDB player development pipeline: windows, percentiles, change flags |
| `baseball/two_week_signal/` | Which two-week Triple-A numbers carry into the next two weeks |
| `baseball/export_site_data.py` | Writes aggregated JSON for the site and the numbers quoted in article prose (no pitch-level data is published) |
| `site/` | Quarto site. Interactive charts use Observable Plot and d3, self-hosted in `site/assets/js/`. Article numbers come from `site/_generated/<sport>.yml` |
| `tests/` | Config, the dbt pipeline on synthetic data, and a browser test of every page |

## Build locally

```bash
uv sync                                    # Python 3.12 + exact locked versions into .venv
uv run python scripts/pull_data.py         # first run ~30-60 min
uv run ./build.sh                          # models -> dbt -> JSON -> docs/
uv run pytest && uv run ruff check .       # browser tests need `uv run playwright install chromium` once
cd docs && python -m http.server 8000      # http://localhost:8000
```

Needs [uv](https://docs.astral.sh/uv/) and the [Quarto CLI](https://quarto.org/docs/get-started/).
On Windows, run `build.sh` from Git Bash.

## Publishing

GitHub Actions deploys to GitHub Pages:

- **`deploy`**: every push to `main` renders the site from the committed JSON in `site/data/` and deploys (a few minutes).
- **`refresh-baseball`**: daily, April through October, re-pulls the current season, rebuilds every model
  and the dbt pipeline, commits the refreshed data, and deploys. Weekly in the off-season it keeps the
  past-season data cache alive. Run it by hand from the Actions tab ("Run workflow") at any time.
- **`check`**: every pull request runs lint, the dbt pipeline on synthetic data, and a browser test of
  every page. `main` only accepts pull requests that pass. Dependabot opens weekly upgrade PRs.

## Adding a sport

Each sport gets its own top-level folder (`basketball/`, `football/`, …) with its own pull, models,
and an `export_site_data.py` that writes `site/data/<sport>/` and `site/_generated/<sport>.yml`
through the shared `sparvi` package, plus a `[<sport>]` section in `sparvi.toml` and its own
`refresh-<sport>.yml` on its own calendar. Pages go in `site/<sport>/` and reuse
`site/assets/js/sparvi.js` (tables, charts, percentile bars, controls) and `site/assets/css/sparvi.css`,
so every sport looks like one site. CLAUDE.md has the full checklist.

## Season rollover (each spring)

1. In `sparvi.toml`, add the new season's date window and bump `season`. Everything else derives from it.
2. Merge that as a pull request, then run `refresh-baseball` manually once.
