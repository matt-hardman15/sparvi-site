# Sparvi baseball v2: implementation plan

Scope: four new pieces (Rule 5 board, Location+/Pitching+, Triple-A → MLB translation, farm
system rankings) plus two cross-cutting features: **starter/reliever filters** and a **date slider**
that shows how ratings and metrics evolve over the season, for the league views and for individual
pitchers.

Follow `CLAUDE.md` for everything not specified here (article shape, voice, chart rules, aggregates-only,
verify-before-done). Work milestone by milestone. Each one ends with its "Done when" checks passing
and a commit. Don't start the next milestone with the previous one broken.

---

## Milestone order and why

| # | Milestone | Why this order |
|---|---|---|
| M1 | Pitcher roles (SP/RP) + role filters on existing views | Small, needed by everything after it |
| M2 | Rule 5 board | Time-sensitive: has to ship before the Rule 5 draft at the Winter Meetings |
| M3 | Time-series infrastructure + date slider (retrofit existing views) | Every later piece should launch with the slider built in |
| M4 | Location+ and Pitching+ | Completes the pitch model; upgrades leaderboard, player pages, Rule 5 board |
| M5 | Triple-A → MLB translation | Needs MLB hitter/pitcher marts from M3 |
| M6 | Farm system rankings | Needs org mapping, roles, and Stuff+ by half-season |

**First task before M2:** look up the 2026 Rule 5 dates (40-man protection deadline, and the draft
date at the Winter Meetings) on MLB.com. Tell Matt the dates and plan M2 to publish shortly after
the protection deadline, when the eligible pool is final.

---

## M1 · Pitcher roles (starter / reliever)

### Data model (dbt, `baseball/pd_pipeline`)
1. **Extend staging to MLB.** Add `statcast_mlb_{2024,2025,2026}` as sources and a `level` column
   (`'MLB'` / `'AAA'`) to `stg_pitches`. Every downstream model groups by `level`. Keep the
   existing Triple-A marts working: filter `level = 'AAA'` where they need it.
2. **`int_pitcher_appearances`** (one row per pitcher × game):
   `level, season, game_pk, game_date, pitcher_id, team, started, pitches, batters_faced, outs_recorded (from events), first_inning`.
   - `started` = this pitcher threw the first pitch for his team in the game (minimum `at_bat_number`
     on his team's defensive side). Don't rely on `inning = 1` alone, because openers break it.
3. **`dim_pitcher_role`** (pitcher × level × season, and as-of a date; see M3):
   - `starts`, `relief_apps`, `start_share = starts / games`
   - `role`: **SP** if `start_share >= 0.5 and starts >= 3`, **RP** if `start_share <= 0.2`,
     otherwise **Swing**.
   - Document thresholds in the glossary. Openers usually land in RP or Swing, which is fine.
   - Also store `role_by_pitches` (share of pitches thrown in starts) as a tiebreak column for later
     use; don't surface it yet.

### Export + UI
- Add `role` (SP / RP / Swing) to `stuff_pitchers.json`, `aaa_pitchers.json`, and player page data.
- New shared control `Sparvi.roleFilter(value, onchange)` in `sparvi.js`: segmented **All · Starters · Relievers**
  (Swing shows under All only, and as its own option if count > 0 after filtering).
- Add the filter to: Stuff+ leaderboard (both views), Triple-A explorer (pitcher side), and later
  Rule 5 board, Location+/Pitching+ views, farm rankings drill-down.
- Make the default minimum-pitch qualifier role-aware: SP 500 / RP 250 for MLB;
  SP 300 / RP 150 for Triple-A. Show the active qualifier in the caption.
- Show the role as a small tag on player pages ("SP · 27 G, 26 GS").

### Done when
- Spot-check 10 known pitchers (5 starters, 5 relievers) against Baseball Reference game logs:
  game and start counts match exactly.
- Role filter works on every view above, survives switching level and team, and is reflected in
  the URL (see URL state in M3; implement `?role=` now).

---

## M2 · Rule 5 board

### What it is
A tool page (`/baseball/tools/rule-5.html`) plus a short article. It covers Triple-A players who are
**likely Rule 5 eligible and not on a 40-man roster**, ranked on the things that predict MLB value
(pitchers: Stuff+ and, once M4 lands, Pitching+; hitters: swing decisions and contact quality).

### Data (new pull script `scripts/pull_rosters.py`, MLB Stats API, run on Matt's machine/Actions)
- **40-man rosters** for all 30 clubs: `GET /api/v1/teams/{teamId}/roster?rosterType=40Man`
  (MLB team ids from `/api/v1/teams?sportId=1`). Snapshot it **after the protection deadline** and
  store the snapshot date.
- **Player bio + acquisition:** `GET /api/v1/people/{id}?hydrate=draft,transactions`
  → birthDate, draftYear, and transactions (to find first pro signing date for international/undrafted).
- **Affiliate → parent org:** `GET /api/v1/teams?sportId=11&season=2026` (`parentOrgId`). Reused in M6.

### Eligibility estimate (be explicit that it's an estimate)
Rule 5 basics: a player signed at **18 or younger** becomes eligible after **5** seasons; signed at
**19 or older**, after **4**. "Season" counts the signing year if signed before the end of that
season's minor league regular season. Implement as:
- `signing_date` = draft signing / first signing transaction; else draft year (assume summer); else unknown.
- `age_at_signing` = age on the signing date (MLB uses age as of the following June 5 for draft picks;
  use that rule for drafted players).
- `eligible_from_year` computed from the rule; `likely_eligible` = not on a 40-man AND
  `eligible_from_year <= 2026`. Status values: **Eligible**, **Likely eligible** (estimated signing
  date), **Unknown**. Show the status and the reason in a tooltip.
- Exclude players with significant 2026 MLB time who were outrighted only if the data makes that
  clear; otherwise keep them and note it.
- Validate against at least 10 players named in published Rule 5 eligibility lists or protection
  coverage (MLB Pipeline, team beat writers). Report the hit rate in the method section.

### Page
- Tabs: **Pitchers · Hitters**. Filters: role (pitchers), org, age max, minimum PA/BF, search.
- Pitchers table: name, org, age, role, BF, Stuff+, (Pitching+ after M4), K%, BB%, K−BB%, CSW%,
  FB velo, best pitch (type + Stuff+), eligibility status.
- Hitters table: name, org, age, position, PA, chase pctl, zone contact pctl, EV90 pctl, hard-hit pctl,
  xwOBA, eligibility status.
- A scatter (pitchers: Stuff+ vs K−BB%; hitters: chase pctl vs EV90 pctl), with role/org highlight.
- Article (short): finding = the handful of unprotected arms/bats whose underlying numbers stand out
  and why; so-what = what to watch for if they're picked; method box = eligibility estimate, its hit
  rate, data limits (no lower levels, no injuries, no makeup/scouting info).

### Done when
- Eligibility validation hit rate is reported (target ≥ 80% agreement on the check list).
- No player on any 40-man roster appears on the board.
- Page passes the standard verification (console clean, filters work, numbers in prose match exports).

---

## M3 · Time-series infrastructure + date slider

### Design decisions (do these exactly; they keep the slider honest)
1. **Snapshot cadence: weekly (each Sunday) plus the final day of the season.** Daily is too much
   data for the browser and too noisy. Snapshot dates come from the actual schedule: the last game
   date on or before each Sunday.
2. **Two windows per snapshot:** *season-to-date* (cumulative) and *trailing 30 days*. The slider
   shows season-to-date by default, with a toggle for "last 30 days".
3. **Fixed Stuff+ / Location+ / Pitching+ scale.** The scale's mean and SD come from the **full
   prior season's** qualified MLB pitchers and are frozen for the whole current season. Otherwise
   100 moves as the slider moves and early-season numbers mean nothing. Store the reference values
   in `output/scale_reference.json` and print them in the method section.
4. **Prorated qualifiers for percentiles.** The qualified pool as of date *d* uses
   `min = full_season_min × (team games played by d / full-season team games)`, floored at
   25 PA/BF. Percentiles at each snapshot are computed against that snapshot's pool.
5. **Role as of each snapshot** (from `int_pitcher_appearances` up to that date), so a swingman who
   moves to the rotation in June changes groups on the slider.

### Pipeline changes
- `pitch_quality/train.py`: also write **pitch-level predictions** (`game_pk, at_bat_number,
  pitch_number, xrv_stuff`, and later `xrv_loc`, `xrv_pitching`) to
  `pitch_quality/output/pitch_predictions_{level}_{season}.parquet`. Internal only: gitignored
  and never exported.
- dbt: source those predictions; new models
  - `int_snapshot_dates` (level, season, snapshot_date, team_games_played)
  - `fct_pitcher_snapshots`: pitcher × snapshot × window: role, pitches, BF, K%, BB%, CSW%, whiff%,
    chase%, xwOBA allowed, FB velo, Stuff+ (and later Location+, Pitching+), percentiles
  - `fct_pitch_type_snapshots`: pitcher × pitch type × snapshot (cumulative): n, usage, velo, IVB, HB,
    spin, whiff%, Stuff+
  - `fct_hitter_snapshots`: hitter × snapshot × window: the existing hitter metrics + percentiles
  - Implement the windows with the existing `windowed()` macro pattern, parameterized by snapshot
    date (join pitches to snapshot dates with `game_date <= snapshot_date`). Watch performance:
    DuckDB handles this fine with a range join, but don't materialize per-day.
- Tests: snapshot row counts monotonic for cumulative pitches; final snapshot equals the existing
  season-level marts **exactly** (this is the key correctness test); percentiles in 0–100.

### Export (`export_site_data.py`)
- **League-level files** (one per level): `snapshots_pitchers_{mlb,aaa}.json`,
  `snapshots_hitters_aaa.json`, in columnar form to keep them small:
  `{dates: [...], players: [{id, name, team, series: {stuff_plus: [...], k_pct: [...], ...}}]}`.
  Round aggressively (1 decimal for plus metrics, 3 for rates). Target < 2 MB gzipped per file.
  If larger, export only qualified-at-season-end players to the league files.
- **Per-player files** `site/data/baseball/players/{id}.json` holding everything a player page needs,
  including pitch-type snapshots and per-game velocity. This also sets up static player pages later.
  Player pages switch from loading the big shared JSON files to loading one per-player file.

### UI components (`sparvi.js`)
- `Sparvi.dateSlider(dates, value, onchange)`: range input over snapshot dates, label "Through
  Jun 29", ◀ ▶ step buttons, and a ▶ play button that animates through the season (≈600 ms per step,
  stop at end). Keyboard accessible.
- `Sparvi.urlState`: read/write `?asof=YYYY-MM-DD&role=SP&window=std` etc. with `history.replaceState`
  so every view is shareable at a point in time.
- Charts get a consistent "as-of" marker: a vertical rule at the slider date on all time-axis charts.

### Where the slider goes
- **Stuff+ leaderboard:** slider + window toggle. Table and scatter reflect the selected date. Add a
  "movers" sort: change in Stuff+ since the previous snapshot or since a chosen start date.
- **Triple-A explorer:** slider for the pitcher and hitter tables, percentiles, and scatter.
- **Player pages:** a "Season timeline" section. Small multiples (Stuff+ overall, FB velo, K%, BB%,
  whiff%, plus per-pitch Stuff+ lines), each with the as-of rule. The slider on the page updates the
  stat tiles, percentile bars, and arsenal table to that date. Hitters get chase, zone contact,
  EV90, hard-hit, xwOBA.
- League-level "season view" chart on the leaderboard: distribution of Stuff+ by role over time
  (median and 25–75% band, SP vs RP), so readers can see league trends (e.g., velo ramping up in spring).

### Done when
- Final-snapshot values equal season-level values for 20 random players (automated test in
  `tests/`).
- Moving the slider on each view changes tables, charts and stat tiles together. URL updates and
  reloading restores the view.
- File sizes within targets. The page with the largest file loads interactively in < 2 s on a
  throttled "Fast 4G" profile in Chrome devtools.
- Existing articles still render unchanged.

---

## M4 · Location+ and Pitching+

### Models (`baseball/pitch_quality/`)
- **Location model:** features `plate_x` (mirrored for batter handedness so "inside/outside" is
  consistent), `plate_z` normalized to the batter's zone (`(plate_z - sz_bot) / (sz_top - sz_bot)`),
  `balls`, `strikes`, `same_hand` (pitcher vs batter), and `pitch_type` group. No velocity or
  movement. Same target (`rv_x`), same per-group LightGBM setup, same train/holdout years.
- **Pitching model:** stuff features + location features + count. One model per group.
- Scale both on the frozen prior-season reference (M3 decision 3).
- Pitch-level predictions go into the same internal predictions parquet.

### Validation (same tests as Stuff+, reported side by side)
- Split-half reliability within season: Stuff+, Location+, Pitching+, actual RV.
- Out-of-sample year-over-year: refit on 2024 only, grade 2025, correlate with 2026 RV/100,
  bootstrap CIs. Expect Location+ to be less stable than Stuff+. Report whatever comes out.
- Check Pitching+ beats Stuff+ alone at predicting next-season RV. If it doesn't, say so plainly
  and don't promote it as the headline metric.
- Triple-A → MLB check for all three (as in the Stuff+ article).

### Site
- Leaderboard gets Location+ and Pitching+ columns, a metric picker for the scatter
  (Stuff+ vs Location+ quadrant view with labeled quadrants), role filter and date slider.
- Player pages: per-pitch Location+ in the arsenal table; Location+ line in the timeline.
- Article: "Stuff or command? Splitting pitching into what the ball does and where it goes."
  Finding = how stable/predictive command is vs stuff; so-what = which profile to bet on for
  improvement; explorer = quadrant chart with role filter + slider; method box = features,
  "location ≠ intended target" caveat, validation table.
- Update the Rule 5 board to include Pitching+.

### Done when
- Validation numbers in `output/metrics_location.json` and in the article match.
- All three grades available on leaderboard, player pages, and Rule 5 board, with filters + slider working.

---

## M5 · Triple-A → MLB translation

### Cohorts
- **Across seasons:** 2025 Triple-A (≥ 150 PA / ≥ 150 BF) → 2026 MLB (≥ 100 PA / ≥ 100 BF).
- **Within season (adds sample):** for 2025 and 2026, a player's Triple-A line *before* his first
  MLB game that season → his MLB line *after* it (same minimums). Use first MLB appearance date
  from the MLB data.
- Pool both and report each separately too. Weight by the smaller of the two samples.
- Pitchers split by role (SP/RP) at the MLB level.

### Metrics (need MLB marts from M3)
- Hitters: chase%, zone swing%, zone contact%, whiff%, K%, BB%, hard-hit%, EV90, xwOBA, wOBA.
- Pitchers: Stuff+, Location+, Pitching+, K%, BB%, CSW%, whiff%, xwOBA allowed, RV/100.

### Analyses
1. **Same-metric carryover:** correlation of each Triple-A metric with the same metric in MLB.
   This shows which skills travel.
2. **Level adjustment:** median change Triple-A → MLB per metric (e.g., chase +x pts, EV90 −y mph),
   with bootstrap CIs. This is the "translation factor".
3. **What predicts MLB production:** each Triple-A metric's correlation with MLB xwOBA (hitters)
   or MLB RV/100 (pitchers), and a small cross-validated ridge regression combining them.
   Report CV R² honestly. Compare against "Triple-A wOBA/ERA-equivalent alone".
4. **Projection:** apply the translation factors + regression to every current Triple-A player →
   "MLB-translated profile" with an uncertainty band.

### Site
- Article: "Which Triple-A numbers travel to the majors". Finding = which metrics carry and which
  don't (expected: swing decisions and stuff travel, Triple-A results less so; report actuals).
  Explorer: pick a metric → Triple-A vs MLB scatter with y=x and fitted line, role filter, cohort
  toggle, search.
- Tool: "Translation" panel on Triple-A player pages (projected MLB K%, BB%, xwOBA with range)
  plus a translated column set in the Triple-A explorer.
- Method box: selection bias (only promoted players are observed; promotion decisions use
  information we don't have), small samples, park/league environment.

### Done when
- Analyses written to `baseball/aaa_translation/output/` with n per cohort.
- Projections exist for every Triple-A player over the minimum. The page explains the band.
- Every number in the article matches the outputs.

---

## M6 · Farm system rankings

### Question
Which organizations' Triple-A pitchers (and hitters) improve **during the season**, and is that
consistent from year to year? The year-to-year consistency is the real test of whether an org
effect exists.

### Method (`baseball/farm_systems/analysis.py`)
- Org mapping from M2 (`parentOrgId` for each Triple-A club, per season).
- **Pitchers:** same design as shape-stability, at Triple-A. For pitchers with ≥ 100 pitches in
  both halves of a season with the same org: change in Stuff+, FB velo, and Pitching+ (after M4),
  league-adjusted per season.
- **Hitters:** change in chase%, zone contact%, EV90 with ≥ 100 PA each half.
- **Exclusions:** MLB rehab assignments (players with meaningful MLB time that season before the
  Triple-A stint), players who changed orgs mid-season (or attribute by stint, documented).
- **Org score:** mean change per org with **empirical-Bayes shrinkage** toward the league mean
  (per-org n is small, ~10–30), plus bootstrap 90% intervals.
- **Validity check:** correlate org scores 2025 vs 2026. If near zero, the article's finding is
  that in-season Triple-A improvement mostly isn't an org trait. That's a legitimate and
  interesting result. Don't spin it.
- **Survivorship:** promoted pitchers leave the second-half sample. Report how many per org and
  run a sensitivity version that includes their MLB second-half data where available.

### Site
- Article: "Which farm systems make players better?" Finding first (whatever the consistency check
  says). Explorer: org table with shrunk score, interval, n, 2025 vs 2026 side by side; click an org
  → its players' changes, with role filter and date slider for that org's Triple-A staff.
- Org pages are optional now; the drill-down is enough.

### Done when
- Shrinkage and intervals implemented and explained. The 2025 vs 2026 consistency result is
  reported and drives the headline.
- No team is ranked without its n and interval shown next to it.

---

## Cross-cutting checklist (every milestone)
- [ ] `./build.sh` runs clean from raw data. dbt tests pass. New tests added for new models.
- [ ] Aggregates only in `site/data` (grep exports for pitch-level columns before committing).
- [ ] Every new view: role filter where pitchers appear; date slider where a time dimension exists;
      URL state; hover tooltips; captions that say what to look for; mobile layout checked at 390 px.
- [ ] Every number in prose traced to an output file; method box includes n, intervals, caveats.
- [ ] Glossary updated (roles, Location+, Pitching+, translation, snapshot/qualifier rules).
- [ ] Workflow still completes within its timeout. If the snapshot build adds a lot of time,
      cache intermediate outputs.
- [ ] README updated; season-rollover list includes any new hardcoded season values.
- [ ] Commit per milestone with a descriptive message.