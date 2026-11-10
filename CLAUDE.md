# Sparvi: working guide

Sparvi (sparvi.io) is a free, independent sports analytics site. It publishes analysis meant to
change a decision, with the data behind every finding in an interactive explorer so readers can
check it or reach their own conclusion. Not a portfolio, not tied to any job application.
Matt Hardman runs it; his name appears only on the About page.

## Stack
- Python (pandas, LightGBM, DuckDB 1.5.x; keep `duckdb>=1.5,<1.6`), dbt-duckdb, Quarto website.
- Interactive charts: Observable Plot + d3, self-hosted in `site/assets/js/`. No CDNs, no Google Fonts,
  no trackers. Fonts are self-hosted in `site/assets/fonts/`.
- Shared UI components live in `site/assets/js/sparvi.js` (`Sparvi.table`, `.plot`, `.pbars`, `.seg`,
  `.select`, `.search`, `.number`, `.corr`, `.fmt`, `.pctColor`, `.plusColor`, `.playerLink`) and
  `site/assets/css/sparvi.css`. Reuse them. Don't add a framework or a second charting library.
- Hosting: GitHub Pages via `.github/workflows/site.yml`. Push to main = render + deploy.
  Schedule (Apr–Oct daily) = re-pull current season, `./build.sh`, commit `site/data`, deploy.

## Layout
- `scripts/pull_data.py`: Statcast + MLB Stats API into `data/raw/` (gitignored, never committed).
- `baseball/<project>/`: one folder per analysis; outputs in `<project>/output/`.
- `baseball/pd_pipeline/`: dbt project (staging → intermediate → marts). Metric definitions live in
  `macros/rates.sql`; flag thresholds in `models/marts/rpt_flags.sql`. Run with `DBT_PROFILES_DIR=.`.
- `baseball/export_site_data.py`: the ONLY thing that writes `site/data/baseball/*.json`.
- `site/`: Quarto source. Articles at `site/<sport>/<slug>/index.qmd` (+ `thumb.png`),
  tools at `site/<sport>/tools/`, player page at `site/baseball/player.qmd?id=`.
- `build.sh`: full rebuild. `docs/` is build output (gitignored; Actions deploys it).

## Rules
1. **Aggregates only.** Never publish pitch-level or raw play-level data. The site stays free and
   ad-free (data-terms reasons). Credit sources and keep the "not affiliated" footer.
2. **Every article follows the same shape:** a `.finding` box (one or two sentences, with the key
   numbers) → a `.so-what` box (2–3 actionable bullets) → interactive explorer(s) in `.explorer`
   with controls, a chart, and a caption → plain-language sections → `<details class="method">`
   covering data, definitions, model, checks, and what the analysis can't tell you.
3. **Every number in prose must come from a file the code wrote** (`output/*.json`, the exported
   JSON, or a query you ran). After any change, re-read the outputs and check every number in the
   affected write-up. If a number changed, update the prose; never round toward a nicer story.
4. **State uncertainty plainly.** Include sample sizes and intervals where they exist. Say when an
   effect is small or the low end of an interval is near zero. Correlation is not causation.
5. **Voice:** plain, declarative, concise. No hype words ("game-changing", "unlock", "dive in",
   "leverage"), no em-dash-heavy rhetoric, no first person singular in articles.
6. **Charts:** color follows meaning. Diverging blue (#2a78d6) = worse, gray = average, orange
   (#eb6834) = better. Brand teal (#0f5257) for neutral emphasis. Percentiles are always oriented so
   higher = better. One axis per chart, hover tooltips on every interactive chart, `clip: true` on
   scatters with fixed domains, captions that tell the reader what to look for.
7. **Verify before you're done:** run `./build.sh` (or the relevant steps), serve `docs/` with
   `python -m http.server`, load every changed page, check the browser console for errors,
   exercise the controls, and look at the result.
8. **Git:** small, descriptive commits. Do not add AI attribution or co-author trailers to
   commits or files. Never commit `data/raw`, `.duckdb` files, or secrets.

## Adding a sport
New top-level folder (`basketball/`, `football/`, `soccer/`) with its own pull script, analysis
folders, and `export_site_data.py` writing to `site/data/<sport>/`. Pages in `site/<sport>/`, reusing
sparvi.js/css. Add it to the navbar in `site/_quarto.yml`, the home-page sport strip, and the README.
Check each data source's terms before using it (nflverse, StatsBomb open data, and NBA stats each have their own).

## Season rollover (each spring)
Add the season to `SEASON_WINDOWS` in `scripts/pull_data.py`, then bump the season in
`baseball/pitch_quality/train.py`, `baseball/pd_pipeline/dbt_project.yml`,
`baseball/export_site_data.py`, `build.sh`, and the workflow's `--seasons`. Run the workflow manually once.