# Sparvi

Free, independent sports analytics: findings meant to change a decision, with the data
behind them in explorers readers can filter and check. Live at **https://sparvi.io**.

## What's here

| Path | |
|---|---|
| `sparvi.toml` | Each sport's current season and windows: the only place a season is set |
| `sparvi/` | Shared Python package: config, site-data writers with schema checks, provenance, plot style |
| `ingest/` | Stage 1 (PLAN-scout A0): each sport's pulls and the **data checks** that run before any build (`ingest/<sport>/checks.py`, shared code in `ingest/common/`). Results go to `site/data/status/<sport>.json` and the `/status/` page |
| `contracts/` | What each sport's raw files must look like for the stages after ingest: columns and types, keys, allowed missing values. The checks enforce them, and the synthetic test fixtures are shaped by them |
| `ingest/baseball/pull.py` | Pulls pitch-level Statcast (MLB, Triple-A), MLB Stats API rosters, Triple-A affiliates per season (for farm-system rankings), and batter heights (for the strike zone) into `data/raw/` (not committed) |
| `baseball/pitch_quality/` | Stuff+, Location+, and Pitching+ models (LightGBM): `train.py`, `robustness.py` (out-of-sample validation), `thumb.py` |
| `baseball/shape_stability/` | Do midseason pitch shape changes persist into next season? |
| `baseball/pd_pipeline/` | dbt + DuckDB pipeline over MLB and Triple-A pitches: pitcher appearances and roles (SP/RP/Swing), Triple-A windows, percentiles, change flags, weekly snapshots (season to date and last 30 days) behind the date slider, and the Triple-A and MLB season lines behind the translation |
| `baseball/rule5/` | Rule 5 eligibility estimate, its validation against published lists, and the board's thumbnail |
| `ingest/baseball/pull_rosters.py` | 40-man rosters, Triple-A affiliates, and player signing history from the MLB Stats API (run by hand after the protection deadline) |
| `scripts/check_roles.py` | Spot-checks pitcher games and starts against MLB's official stats |
| `baseball/two_week_signal/` | Which two-week Triple-A numbers carry into the next two weeks |
| `baseball/farm_systems/` | Farm-system rankings: each org's in-season Triple-A improvement, shrunk with intervals, and whether it repeats year to year (`analysis.py`, `thumb.py`) |
| `baseball/aaa_translation/` | Triple-A -> MLB translation: which numbers carry, how much they shift, what predicts MLB production, and per-metric translation lines (`analysis.py`, `thumb.py`) |
| `baseball/export_site_data.py` | Writes aggregated JSON for the site and the numbers quoted in article prose (no pitch-level data is published) |
| `ingest/nfl/pull.py` | NFL play-by-play, schedules, teams, and team stats from nflverse (CC-BY 4.0) into `data/raw/nfl/` (not committed; ~30 seconds) |
| `football/common/` | Football definitions shared by the NFL and (later) college sections: garbage-time filter, opponent adjustment (ridge), team codes |
| `football/dbt/` | dbt + DuckDB project for football (`nfl_*` models): efficiency plays, team-game sums, weekly windows (season to date, last 4 weeks), season totals, game logs, and a reconciliation against nflverse team stats |
| `nfl/team_efficiency/` | Opponent adjustment as of every week, and the choice of its ridge penalty by next-week margin prediction |
| `nfl/export_site_data.py`, `nfl/team_pages.py` | NFL site data and prose numbers; the 32 static team pages and their preview images |
| `site/` | Quarto site. Player pages load one `data/baseball/players/{id}.json` each; the tools load columnar weekly snapshot files. Interactive charts use Observable Plot and d3, self-hosted in `site/assets/js/`. Article numbers come from `site/_generated/<sport>.yml` |
| `site/build/` | `seo.py` runs after every render: canonical URLs, sitemap `lastmod` from each page's last commit, `sitemap-index.xml`, `robots.txt`, and a check that fails the build on missing or duplicate titles/descriptions; `og_default.py` draws the default social image |
| `models/`, `forecasts/`, `scoring/`, `review/` | Later stages (model registry, forecast ledger, scoring, review log); placeholders with a README each until A2, L1, L2. `review/model-changes.md` logs model changes |
| `tests/` | Config, the dbt pipeline on synthetic data, and a browser test of every page |

## Build locally

```bash
uv sync                                    # Python 3.12 + exact locked versions into .venv
uv run python ingest/baseball/pull.py         # first run ~30-60 min
uv run python ingest/nfl/pull.py          # NFL, ~30 seconds (optional: without it the committed NFL data is used)
uv run python -m ingest.baseball.checks    # data checks (and -m ingest.nfl.checks); writes site/data/status/
uv run ./build.sh                          # models -> dbt -> JSON -> docs/ (runs nfl/build.sh too when NFL data is present)
uv run pytest && uv run ruff check .       # browser tests need `uv run playwright install chromium` once
SPARVI_PERF=1 uv run pytest tests/test_perf.py -s   # opt-in: pages interactive in < 2 s on throttled "Fast 4G"
cd docs && python -m http.server 8000      # http://localhost:8000
```

Needs [uv](https://docs.astral.sh/uv/) and the [Quarto CLI](https://quarto.org/docs/get-started/).
On Windows, run `build.sh` from Git Bash.

## Publishing

GitHub Actions deploys to GitHub Pages:

- **`deploy`**: every push to `main` renders the site from the committed JSON in `site/data/` and deploys (a few minutes).
- **`refresh-baseball`**: daily, April through October, pulls the new games (only games since the last
  one on file, plus a 3-day overlap for Savant's revisions; a full re-pull on Mondays or on request),
  rebuilds every model and the dbt pipeline, commits the refreshed data, and deploys. Weekly in the off-season it keeps the
  past-season data cache alive. Run it by hand from the Actions tab ("Run workflow") at any time.
  If the raw data and code match what the last refresh built from (`scripts/build_fingerprint.py`,
  stored in `baseball/build_inputs.sha256`), it skips the rebuild: model training is repeatable on one
  machine but not bit-for-bit across CPU models. The manual run's "rebuild" option overrides this.
- **`nfl`**: Tuesdays, September through February, pulls nflverse, rebuilds the football pipeline,
  adjustment, export, and team pages, commits the NFL data, and deploys. Independent of baseball.
- **Data checks**: both refreshes run their sport's checks right after the pull. If one fails, nothing
  is rebuilt, the site keeps the last data that passed, the failure shows on `/status/`, and the run
  ends red. A source changing a column or type fails the checks until `contracts/` and the code
  that reads it are updated together.
- **`check`**: every pull request runs lint, both dbt projects on synthetic data, and a browser test of
  every page. `main` only accepts pull requests that pass. Dependabot opens weekly upgrade PRs.

## Adding a sport

Each sport gets its own top-level folder (`basketball/`, `football/`, …) with its own pull, models,
and an `export_site_data.py` that writes `site/data/<sport>/` and `site/_generated/<sport>.yml`
through the shared `sparvi` package, plus a `[<sport>]` section in `sparvi.toml` and its own
`refresh-<sport>.yml` on its own calendar. Pages go in `site/<sport>/` and reuse
`site/assets/js/sparvi.js` (tables, charts, percentile bars, controls) and `site/assets/css/sparvi.css`,
so every sport looks like one site. Also add `_generated/<sport>.yml` to `metadata-files` in
`site/_quarto.yml`, the sport to the navbar and home-page sport strip, and its pages to `tests/test_site.py`.

Football specifics: the NFL and college football share `football/common/` (one garbage-time rule per
level, one opponent adjustment) and one dbt project, `football/dbt/`, with `nfl_*` and `cfb_*` models.
Each has its own top-level folder for analyses (`nfl/`, `cfb/`), its own workflow, and its own
`sparvi.toml` section. NFL team pages are static (`site/nfl/teams/<code>/`, generated by
`nfl/team_pages.py`) so each has its own title and preview image. The site's week slider is
`Sparvi.dateSlider` with a week label. College football (CFBD API) needs `CFBD_API_KEY` in a local
`.env` and as an Actions secret; never in code, logs, or `site/`.

## Season rollover

NFL (each August): bump `season` under `[nfl]` in `sparvi.toml`.

Baseball (each spring):

1. In `sparvi.toml`, add the new season's date window and bump `season`. Everything else derives from it.
2. Merge that as a pull request, then run `refresh-baseball` manually once. Stuff+ is scaled to the
   prior season automatically; if a league changes its schedule length, update `full_season_games`
   in `baseball/pd_pipeline/dbt_project.yml` (it prorates the qualified pools).

Each fall, update `[baseball.rule5]` in `sparvi.toml` (protection deadline, draft date, CBA expiry), run
`uv run python ingest/baseball/pull_rosters.py` the day after the deadline, rebuild, and refresh
`baseball/rule5/validation_list.csv` with that year's published eligibility lists.
