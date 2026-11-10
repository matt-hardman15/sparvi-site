# Sparvi: working guide

Sparvi (sparvi.io) is a free, independent sports analytics site. It publishes analysis meant to
change a decision, with the data behind every finding in an interactive explorer so readers can
check it or reach their own conclusion. Not a portfolio, not tied to any job application.
Matt Hardman runs it; his name appears only on the About page.

## Stack
- Python 3.12 (`.python-version`) managed with uv: ranges in `pyproject.toml`, exact versions in
  `uv.lock`. `uv sync` to set up; `uv add`/`uv lock` to change deps (never pip-install ad hoc).
  pandas, LightGBM, DuckDB 1.5.x (keep `duckdb>=1.5,<1.6`), dbt-duckdb, Quarto (pinned in workflows).
  Dependabot opens weekly upgrade PRs; the `check` workflow tests them.
- Interactive charts: Observable Plot + d3, self-hosted in `site/assets/js/`. No CDNs, no Google Fonts,
  no trackers. Fonts are self-hosted in `site/assets/fonts/`.
- Shared UI components live in `site/assets/js/sparvi.js` (`Sparvi.table`, `.plot`, `.pbars`, `.seg`,
  `.select`, `.search`, `.number`, `.corr`, `.fmt`, `.pctColor`, `.plusColor`, `.playerLink`) and
  `site/assets/css/sparvi.css`. Reuse them. Don't add a framework or a second charting library.
- Hosting: GitHub Pages. Repo: github.com/matt-hardman15/sparvi-site (public).
  - `deploy.yml`: push to main = render committed data + deploy.
  - `refresh-baseball.yml`: Apr–Oct daily = re-pull current season, `./build.sh`, commit data,
    call deploy. Nov–Mar weekly = keep the past-season raw-data cache from expiring.
  - `check.yml`: every PR = ruff, dbt on synthetic data, render + browser test of every page.
  main is protected: changes go through a PR with passing checks (the refresh bot is exempt).

## Layout
- `sparvi.toml`: the current season and season windows. The only place a season is set.
- `sparvi/`: shared Python package for every sport: `config` (reads sparvi.toml; also
  `python -m sparvi.config baseball.season` for shell), `io` (UTF-8 JSON writers with schema checks,
  prose numbers), `provenance` (data ranges, library versions, seeds), `plotstyle`.
- `scripts/pull_data.py`: Statcast + MLB Stats API into `data/raw/` (gitignored, never committed).
- `baseball/<project>/`: one folder per analysis; outputs in `<project>/output/`.
- `baseball/pd_pipeline/`: dbt project (staging → intermediate → marts). Metric definitions live in
  `macros/rates.sql`; flag thresholds in `models/marts/rpt_flags.sql`. Run with `DBT_PROFILES_DIR=.`.
- `baseball/export_site_data.py`: the ONLY thing that writes `site/data/baseball/*.json` and
  `site/_generated/baseball.yml` (every number quoted in baseball prose). Each output declares the
  columns the pages read; a missing column fails the build. It also asserts the claims articles
  make about the data (`StaleProse`): if one fails, rewrite the sentence, don't loosen the check.
- `site/`: Quarto source. Articles at `site/<sport>/<slug>/index.qmd` (+ `thumb.png`),
  tools at `site/<sport>/tools/`, player page at `site/baseball/player.qmd?id=`.
- `build.sh`: full rebuild (`uv run ./build.sh`). `docs/` is build output (gitignored; Actions deploys it).
- `tests/`: `test_config.py`, `test_pipeline.py` (dbt on `synthetic_statcast.py`, never real data),
  `test_site.py` (Playwright: every page, every control, console errors, data budget).

## Rules
1. **Aggregates only.** Never publish pitch-level or raw play-level data. The site stays free and
   ad-free (data-terms reasons). Credit sources and keep the "not affiliated" footer.
2. **Every article follows the same shape:** a `.finding` box (one or two sentences, with the key
   numbers) → a `.so-what` box (2–3 actionable bullets) → interactive explorer(s) in `.explorer`
   with controls, a chart, and a caption → plain-language sections → `<details class="method">`
   covering data, definitions, model, checks, and what the analysis can't tell you.
3. **Never hand-type a number into prose.** Add it to the `N` dict in the sport's
   `export_site_data.py`, formatted the way the sentence prints it, and write
   `{{< meta baseball.<key> >}}` in the article (in JS, read `meta.json`). Years too: use
   `{{< meta baseball.season >}}` / `season_prev`. If a sentence states a direction or ranking
   ("held", "more than"), add a `StaleProse` check for it. Never round toward a nicer story.
4. **State uncertainty plainly.** Include sample sizes and intervals where they exist. Say when an
   effect is small or the low end of an interval is near zero. Correlation is not causation.
5. **Voice:** plain, declarative, concise. No hype words ("game-changing", "unlock", "dive in",
   "leverage"), no em-dash-heavy rhetoric, no first person singular in articles.
6. **Charts:** color follows meaning. Diverging blue (#2a78d6) = worse, gray = average, orange
   (#eb6834) = better. Brand teal (#0f5257) for neutral emphasis. Percentiles are always oriented so
   higher = better. One axis per chart, hover tooltips on every interactive chart, `clip: true` on
   scatters with fixed domains, captions that tell the reader what to look for.
7. **Verify before you're done:** `uv run ./build.sh` (or the relevant steps), then
   `uv run pytest` (locally, `SPARVI_BROWSER_CHANNEL=chrome` uses installed Chrome) and
   `uv run ruff check .`. Add page interactions to `tests/test_site.py` for new controls. Serve
   `docs/` and look at the changed pages yourself too; tests don't judge whether a chart reads well.
   Pipeline outputs must be deterministic: a rerun on unchanged data must not change any file.
8. **Git:** small, descriptive commits. Do not add AI attribution or co-author trailers to
   commits or files. Never commit `data/raw`, `.duckdb` files, or secrets.

## Adding a sport
New top-level folder (`basketball/`, `football/`, `soccer/`) with its own pull script, analysis
folders, and `export_site_data.py` that writes `site/data/<sport>/` and `site/_generated/<sport>.yml`
through `sparvi.io`. Add a `[<sport>]` section to `sparvi.toml` and a dataclass in `sparvi/config.py`.
Add `_generated/<sport>.yml` to `metadata-files` in `site/_quarto.yml`. Copy `refresh-baseball.yml`
to `refresh-<sport>.yml` with that sport's calendar (it commits only its own data and calls
`deploy.yml`). Pages in `site/<sport>/`, reusing sparvi.js/css. Add it to the navbar, the home-page
sport strip, the README, and `tests/test_site.py`.
Check each data source's terms before using it (nflverse, StatsBomb open data, and NBA stats each have their own).

## Season rollover (each spring)
In `sparvi.toml`, add the new season's window under `[baseball.season_windows]` and bump `season`.
Open a PR, merge it, then run `refresh-baseball` by hand once. Every pull, model, dbt run, cache key,
and prose year follows. If the export raises `StaleProse` (e.g. a worked example's season pair aged
out of the window), rewrite that passage with current examples.