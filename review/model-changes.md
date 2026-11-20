# Model changes

Newest first. Each entry: date, model and version, what changed, why, and before/after metrics.

## 2026-11-20 · NFL win probability 1.0.1 → 2.0.0 (promoted)
Frozen eval set: Development: validation seasons 2018 (trained on 2016-2017) and 2019 (trained on 2016-2018).. dev_brier: 0.1491 → 0.147. Live: not yet available (no ledger forecasts).

## 2026-11-20 · Fourth-down chart: standard situations fixed (no model change)
The chart on `/nfl/fourth-downs/` called 4th-and-3 through 4th-and-7 from inside the offense's own 10 a clear go.
Cause: `nfl/fourth/decisions.grid` set `recv_2h = 0.5` ("unknown"). The win-probability model is trees, so 0.5 fell on
the "receives the 2nd-half kickoff" side of every split, and since the flag flips with the ball (1 - 0.5 = 0.5) the
team with the ball next always "received". That added about 2 x P(convert) x (the kickoff's worth) to every
go-minus-kick margin: +1.2 points on average, 290 of 1,380 cells changed sign. The grid's clock was also 15:00 left in
the 2nd quarter, not the 10:00 the page states. Fix: four rows a cell (home/away x receives or not), averaged; clock
at 10:00. Chart, 945 published cells: clear go 199 → 85, clear kick 20 → 194, toss-up 726 → 666; inside the own 10,
clear go 40 → 13 (none on 4th-and-3 or longer). Real plays always had 0/1 flags: all seven seasons were recomputed and
`teams`, `league` and `costly` in `fourth_downs.json` are identical. The possession-change transforms and the punt
model were checked and are right (tests/test_fourth_transforms.py; punts from the own 10 and in, 2016-2025: net 44.0
actual vs. 43.8 modeled, punting team won 44.0% vs. 43.8% modeled, in sample).

Also (Matt, Nov 20): the chart makes no call from the offense's own 20 back (`summary.CHART_NO_CALL_FROM`, applied in
the export: 200 of the 945 cells publish call "none" and no numbers). Teams almost never go for it there (2016-2025:
167 of 7,046 tries, averaging 11 yards to go), so the conversion rate and the value of a first down are extrapolated;
after the fix above the chart still had 4th-and-1 inside the own 10 as a clear go. Of the 745 cells left: 67 clear go,
142 clear kick, 536 toss-up. Real plays from there are still valued and judged.

Open, to be handled in a separate pull request (Matt, Nov 20): on 2020-2025 seasons it hadn't seen, win probability 1.0.1 rates a team with
the ball inside its own 6 about 4 points too high (2,200 plays: won 46.6%, rated 51.0%; 90% interval of the gap 2.4 to
6.4 points, resampling games). Field position barely moves
it there. That leaves the chart a point or so more go-friendly than nfl4th inside the own 10 (toss-ups where nfl4th
says punt), and touches the 1% of clear-go cost that comes from real plays there. Options are in pull request #104.

## 2026-11-19 · Stuff+ 2.0.0: the backtest trains on two seasons (PQ-TEST-2; no model change)
The pre-registered test trained on one season (2024) while the published model trains on two. Development folds that
never read 2026 (outcomes 2024 and 2025, `baseball/pitch_quality/output/development_windows.json`) put a two-season
model +0.040 in r above a one-season one (90% interval +0.020 to +0.063, 635 pitcher-seasons), so the test understated
the published model. Matt, Nov 19: change the test to match. The card's `backtest.splits.train_window` goes 1 → 2
(train 2023 and 2024, grade 2025, score against 2026); target, baseline, metric, strict gate, seed and test season are
unchanged. This amends a pre-registered block after its one-season 2026 result was seen (run 063ae43be446: r 0.307
vs. 0.193 for last season's results, passed); the card's `amended` entry records that. 2023 is an MLB history season:
the backtest workflow restores it from the private snapshots (`ingest/baseball/history.py`). Local dry run on the new
block, before CI's official run (324 pitchers, leakage tests not run): r 0.319 vs. 0.193, edge +0.127, 90% interval
+0.021 to +0.228, gate passed.

## 2026-11-18 · NFL field-goal make probability 1.0.0 → 2.0.0 (major: new spec, F4-V2)
**Why.** 1.0.0 (logistic on distance, every season since 2016) failed its pre-registered calibration gate on 2020-2025
(6,609 kicks): pooled calibration error 0.026 against a noise bar of 0.014. Walk-forward, it predicted 65% from 50-54
yards where 73% were made, and 51% from 55+ where 59% were made; the error grew from 0.024 (2020) to 0.043 (2024).
Long kicks kept improving and the fit averaged over older seasons. Win probability, net punt and fourth-down
conversion passed in the same run.

**What changed.** `nfl/fourth/fg.py`: distance and (distance - 40)^2 / 100, fit on the last three seasons before the
one predicted (Matt, Nov 18). The card's code list swaps `models.py` for `fg.py`, so the other three F4 models are
untouched. The square term bends upward at long range and extrapolates badly (70 yards: 51%), so the decision engine
offers no kick beyond the 99th percentile of attempted distances (61 yards on 2023-2025).

**Results.** Development seasons (2018, 2019): Brier 0.1205 vs. 0.1207, no real difference, as expected: the era effect
isn't in those seasons. On 2020-2025, which were read before this version was specified and so are not a test of it:
calibration error 0.016 (1.0.0 0.026; noise bar 0.013), Brier 0.1128 (0.1133). Other variants tried on the same seen
seasons: last three seasons without the square 0.024, all seasons with a season trend 0.023, all seasons with the
square and a trend 0.016. Every variant still predicts fewer makes than happened (83.9% vs. 85.2%). The clean test is
the 2026 season (February 2027); until then the version is experimental. Registered as champion by Matt's decision,
not by `registry promote`: the frozen development set can't show what the change is for.

## 2026-11-18 · NFL opponent adjustment 1.0.0 → 1.0.1 (patch: lambda grid, NFL-LAMBDA)
The chosen lambda (3200) was the largest value in the search grid, so the search didn't show it was a minimum. The grid
now runs to 12800. Next-week margin MAE, 2,160 games: 1600 10.3797, 3200 10.3789, 6400 10.3814, 12800 10.3839 (an
exploratory sweep to 1e8 levels off at 10.3876). Lambda stays 3200; `adjusted_team_week.csv` is byte-identical and no
published number changes. The curve is nearly flat (0.009 points from 800 up), and at 3200 the adjustment moves a
team's offense by 0.0025 EPA per play on average: in the NFL the opponent adjustment is close to off, as the method
note already says.

## 2026-11-18 · NFL win probability and net punt 1.0.0 → 1.0.1 (leakage test only; the models are unchanged)
The first CI run (fast fold, 2025) failed both models' shuffled-label test, one seed, in the better-than-null direction
(Brier -0.0034, 90% -0.0062 to -0.0006; punt MAE -0.025 yd, -0.041 to -0.009). No leak behind it: (1) win
probability's monotone constraint is a prior that keeps a leading-teams-win tilt on shuffled outcomes (seed 1 AUC 0.574
with it, 0.493 without); (2) single seeds swing widely (punt shuffled-label correlation over five seeds +0.06, +0.02,
+0.17, -0.14, +0.06). Matt (Nov 18): the shuffled-label fit drops the monotone constraint, and the test averages 10
seeds. With 10 seeds both shuffled-label models were significantly *worse* than the null (Brier +0.0055, 90% +0.004 to
+0.007; punts +0.011 yd, +0.001 to +0.022), as noise-fit models should be, so (Matt) the loss-metric test is one-sided:
it fails only if the shuffled-label model is significantly better than the null. Both pass on 2025. Changed before the
full 2020-2025 run; the models, gates and folds are as pre-registered.

## 2026-11-18 · New: NFL win probability, fourth-down conversion, field-goal make, net punt 1.0.0 (F4, pre-registered)
The four inputs to the fourth-down decision tool (F4). Raw game facts only (nflverse's own EPA/WP columns and betting
lines are never inputs); new raw table `state/pbp_state_{season}` for the game state. Development on 2016-2019 only
(`nfl/fourth/develop.py`; validation seasons 2018 and 2019):
- win probability: LightGBM, monotone in margin and strength, Brier 0.1491 vs. 0.1589 for a margin-by-time logistic.
- net punt: LightGBM L1, MAE 6.57 vs. 6.65 yards for a field-position bucket mean.
- fourth-down conversion: nothing beat the yards-to-go rate table (best logistic, with 3rd downs as extra rows, 0.2335
  vs. 0.2327), so the table is the model.
- field goal: a distance spline with roof, temperature, wind and season tied distance alone (0.1209 vs. 0.1207), so
  the distance-only logistic is the model.

Gates (Matt): win probability and punts must beat their baseline with the 90% interval of the edge (by game) above
zero; conversion and field goals pass if pooled ECE is below the 90th percentile of the ECE perfectly calibrated
predictions show on the same plays (a fixed ECE < 0.02 was dropped before registration: at ~550 attempts a season,
noise alone gives ~0.05). Backtest: walk-forward 2020-2025, expanding training from 2016; 2026 sealed. Harness
changes: Brier, log loss and ECE metrics, the calibration gate, fold dates in the next calendar year, an expanding
window that needs `first_train` (it silently trained on one season before; no card used it), and the "too good"
check now respects metrics where lower is better.

## 2026-11-17 · CFB system fit 1.1.9 → 1.1.10; CFB player projection 1.2.1 → 1.2.2 (patch: board openings)
Matt: the 2027 board said about 48% of Kansas's QB dropbacks were leaving with no portal entries; its starter (Isaiah
Marshall, 86% of dropbacks) is an underclassman who hasn't entered. `model.availability` (F0 role availability; the
board's "open" figure and its best-spots filter only, not the fit model or the projection) applied a leave rate by
experience alone, estimated from players in play-by-play who didn't play for the same team the next season. Two
problems: (1) the history included FCS, Division II and NAIA players who appear only in their one game against an FBS
team, nearly doubling the rates (third-year 0.46 vs. 0.27 on FBS programs); (2) for underclassmen the remaining
"departures" are mostly not departures. FBS starting QBs in their second or third year who skipped the portal, 2021-2024:
32 of 147 "gone", of whom about 9 left early for the NFL (elite prospects), about 8 were seniors whose earlier seasons
were outside our rosters, about 6 stayed but lost the job, were hurt or suspended, and about 5 transferred under an
unlinked portal record. Now: rates from FBS programs in both seasons, by position group and experience over the five
seasons before (a cell under 40 falls back to experience only); players in their first three seasons on a roster who
haven't entered the portal count as returning; early NFL entries aren't predicted. Kansas QB 0.483 -> 0.045 (its
fourth-year backup). FBS QB rooms, median open share 0.36 -> 0.12. The roster's class-year field can't replace our count
(2026: 20,030 players listed as year 1). Fit scores and projections unchanged; best landing spots shift with the filter.

## 2026-11-17 · CFB system fit 1.1.8 → 1.1.9; CFB player projection 1.2.0 → 1.2.1 (patch: portal board)
Best landing spots ranked by projected yards gave nearly every entrant the same programs (2026 window: every WR and TE had
San José State · Old Dominion · Troy; UTSA was in every QB's top 3). Projected yards are his share × the program's volume
× his rate, and volume and the level/development terms are the same for every entrant, so the order is the program's,
not his. Ranking by the fit probability instead had the same problem (Air Force/Army/Navy in most WR, TE, and RB lists:
programs where anyone "succeeds", and which signed no QB/RB/WR/TE transfers in 2023-2025). 1.1.9 ranks by relative fit:
his fit logit minus the program's mean logit at his position over all entrants, among programs with at least the median
open share at his position that signed 3+ skill-position transfers in the last three windows (Matt, Nov 17). Distinct
top-3 sets, 2026 window, all entrants: QB 38/245, RB 57/298, WR 83/588, TE 35/221 (before, among entrants at the
backup minimum: 9/88, 5/142, 1/249, 1/66). Players of a similar type still share many spots: fit sees a player through
a few style dimensions. Fit scores, program fit lists, and projections are unchanged (byte-identical program files).
Also: the coming window (2027) is shown before it opens, with programs' openings, styles, and coaching and no entrants
(Matt, Nov 17); teams_2027 falls back to 2026 membership until CFBD lists it. The ledger freeze still records lists
only once the window has entries. 1.2.1: `cfb/fit/portal.link` handles an empty portal; no output change.

## 2026-11-16 · CFB player projection 1.0.3 → 1.1.0 → 1.2.0; CFB system fit 1.1.6 → 1.1.7 → 1.1.8 (portal board)
1.1.0 (minor: a new output). `pit.whatif`: any entrant's projected season yards at any program, scored against that
program's projected room (returners, committed transfers, signees; himself removed if already projected there). For
committed transfers it reproduces project() exactly (share and yards differences 0.0). The board's projection trains on
outcome seasons through 2025 only (2026 is the Phase 2 vault). A program's room scales every entrant's share equally,
so the order at each program is nearly the same; the board shows each entrant's best landing spots as well (Matt).
1.2.0 (minor: training data). The board showed fourth-year backups near the top of program lists (Cole Gonzales, 1.9% of
Pitt's dropbacks, 9th at Boise State). Research (`cfb/proj/lowuse.py`, exploratory, seen seasons 2021-2025): low-usage
veterans win roles 13-25% of the time (other transfers 20-34%); moving down a level doesn't help except slightly at RB;
recruiting rating and size don't separate them (AUC 0.41-0.54; TE recruiting 0.64 on a small sample); last-season
efficiency and volume do (QB AUC 0.70, 0.65). The specific failure: QBs and TEs moving down a level were projected
1.25-1.57x the share they got, in the portal era only (development 2016-2020: QB down 0.92x). A transfer-experience x
last-share term didn't help in development and was dropped. 1.2.0 builds portal-era training seasons (2021 on) the May 1
way, like the projection itself (Matt), chosen with seen seasons; the 2026 vault confirms. Effect (seen data): QB r 0.532
(1.1.0 0.534), RB 0.514 (0.511), WR 0.489 (0.486), TE 0.441 (0.438); QB low-usage veterans moving down 1.28x -> 1.17x.
System fit 1.1.7 / 1.1.8 (patch): `board.py` adds projected yards beside the fit list (1.1.7), then shows projections and
best landing spots only for entrants who reached last season's backup minimum (1.1.8, Matt). Fit lists are unchanged.

## 2026-11-16 · CFB player profile 1.0.4 → 1.1.0 (minor: receiver measures); system fit 1.1.6, projection 1.0.3 (patch)
RECV-MEASURES. A new feature kind, `per_team_play`: a player's total of a value per team pass play in the games he played,
opponent-adjusted and shrunk like the others. New features: WR/TE receiving yards, receiving EPA, and catches per team
pass play; RB receiving yards per team pass play. They need only completions, which name the receiver in 99%+ of plays;
the per-target stats need incompletions, which 2021-2024 play text mostly doesn't attribute, so 12 target-based features
carry `biased_seasons: [2021-2024]` (values kept, not ranked). Year-to-year r for players who stayed (FIT-STABLE): WR
receiving yards per team pass play 0.571 and EPA 0.552 vs. EPA per target 0.279; TE 0.538 / 0.498 vs. -0.103; RB
receiving yards 0.522 vs. EPA per target 0.220. Profile validation: movers r 0.1811 -> 0.2079 (n 505 -> 698), all
players 0.2645 -> 0.3238, split-half 0.2952 -> 0.3261. System fit and the projection list the engine and features
file as code but read none of the new features: their inputs are unchanged (patch). The fit model's labels still use
EPA per target (`labels.py`, pre-registered); exploration reruns with the new measures follow.
The profile's label-permutation hook now also shuffles who got each play (models/cfb/player_profile/backtest.py, not
model code): per-team-play features carry usage, which an outcome-only shuffle keeps, so the 10-seed test failed with
nothing leaking. With both shuffled: mean shuffled-label r +0.014 (90% CI -0.016 to +0.044), passes.

## 2026-11-16 · CFB player projection 1.0.1 → 1.0.2, CFB system fit 1.1.4 → 1.1.5 (patch: DATA-QA part B)
Found by DATA-QA's new range assertions and id audit (Matt, Nov 16: ML best practice):
- Team play counts came from the event table, which only has a row when someone on the play was matched, so an unmatched
  lead back removed his team's runs (Mississippi State 8 run plays a game in 2020-2022). Counts now come straight from the
  play-by-play (same garbage-time rule): Mississippi State 2021 16 run and 51 pass plays a game.
- Receiving shares used a team total that wasn't clipped like each player's, so a team's shares summed to as much as 1.8.
- Roster positions of '?' (all of 2014) or blank (many FCS rosters) dropped players from every position pool (2-5% of FBS
  touches in 2014-2017, starters such as Javorius Allen and Nick Chubb). `cfb/fit/data.py` now infers the group from the
  role he played most when the listed position is unusable: lost FBS touches 5.1% -> 1.2% in 2014.
- Range assertions on team and player tables and on the projection fail the build; every season 2014-2025 passes.
- Convergence asserted: the share logit and the fit model's logistic regressions raise instead of warning.
Projection, 2021-2025 rerun locally (seen data; the official run is in CI on merge): QB 0.534, RB 0.511, WR 0.486, TE
0.438 (1.0.1: 0.534, 0.508, 0.487, 0.438); same gate outcome. System fit's backtest reruns in CI.

## 2026-11-16 · CFB player projection 1.0.0 → 1.0.1 (patch: receiving rate)
1.0.0's official pre-registered run (2021-2025): QB r 0.534 and RB 0.508 passed the strict per-position gate; WR 0.025
and TE 0.038 failed, near zero in the 2023 and 2024 folds. Cause: team receiving yards per dropback divided by matched
dropbacks, and UL Monroe 2022 (passers 3% matched; CFB-MATCH-FIX) had 1 matched dropback against 1,980 receiving yards,
so its rate was 1,980 and it corrupted the receiving-rate regression for every team in the folds that train on 2022.
1.0.1 divides by the team's pass plays (192 for UL Monroe 2022), independent of passer matching. QB and RB projections
are unchanged (they don't use the rate). Local rerun of 1.0.1 (the official one runs in CI on merge): QB 0.534, RB
0.508, WR 0.487 (vs. last-season yards 0.430, gain 90% CI +0.040 to +0.073), TE 0.438 (vs. 0.445, CI -0.045 to +0.031).
These seasons were looked at while debugging, so 1.0.1's 2021-2025 numbers are not a clean test; the 2026 vault is.
Tried while debugging and not adopted (both lowered QB and RB): training rosters built the May 1 way in the portal era,
and the roster's class year in experience (inconsistent with the departure rates' experience count).

## 2026-11-16 · CFB player projection 1.0.0 (new, pre-registered)
CFB-PROJ (PLAN-fit). Season yards for every QB, RB, WR, TE on an FBS team's projected roster as of May 1: team volume x
share x efficiency, with 80% intervals. Developed on 2016-2020 only (leave one season out): projected-yards r QB 0.531,
RB 0.550, WR 0.500, TE 0.488, each beating last-season yards and share-only with 90% intervals above zero. The
pre-registered backtest (2021-2025, strict in every position group, Matt) runs once in CI when this merges; until it
passes the model is experimental. Change from development: team receiving yards per dropback no longer uses the
projected QBs (changed development receiving r by under 0.001; the point-in-time version would need projected QBs for
every training season).

## 2026-11-15 · CFB player profile 1.0.3 → 1.0.4, CFB system fit 1.1.3 → 1.1.4 (patch: roster matching)
CFB-MATCH-FIX. `ingest/cfb/play_players.py` gave up whenever a name matched several roster entries. CFBD lists
placeholder entries (negative ids) through 2019 and sometimes two ids for one player, so whole teams went unmatched
(Iowa's passers 1-3% matched in 2017-2019, Kansas State 2% in 2019). Now, among same-name candidates: the real id,
then the one with stats that season, then with stats in the role's category, then a position that fits the role, then
offense, then the one with at least twice the role's volume; plus a shortened hyphenated name ("Zeon Chriss") and the
last word of a multi-word last name ("M.Van Buren Jr."). PARSER_VERSION 3 re-parses every cached season. Matched,
2014-2019: passers 97.8-99.7% -> 99.5-99.8%, rushers 95.4-97.2% -> 97.9-98.5%, targets 95.9-98.2% -> 99.2-99.7%.
FBS team-seasons with under 80% of passers matched: 17 -> 2 (2015-2025). Left unmatched: nicknames (Jo'quavious vs.
Woody Marks) and a player CFBD split evenly across two ids (UL Monroe's Chandler Rogers, 2022). Profile validation:
movers r 0.1795 -> 0.1811 (n 494 -> 505), all players 0.2651 -> 0.2645, split-half 0.2958 -> 0.2952. Found by CFB-PROJ
P4 (team pass totals near zero for Iowa and Kansas State). System fit's backtest was not rerun.

## 2026-11-15 · Backtest harness: label permutation over several seeds (CFB player profile)
No model changed. The player profile's fast backtest failed its label-permutation leakage test on sparvi-site #62
(r -0.134, row-bootstrap 90% CI -0.18 to -0.09) though nothing leaks: across 10 seeds the shuffled-label model averages
r -0.008 with a seed-to-seed sd of 0.071, wider than any bootstrap of one shuffle shows (about 150 players, ~7
correlated features each), so a single seed fails about 40% of the time. Matt (2026-11-15): average 10 seeds. The spec's
new `permutation_seeds` (default 1) runs that many shuffles and passes when the 90% interval of their mean (t, n-1 df)
includes zero; a leak present in every seed still fails (tests/test_backtest.py). The profile card sets 10. Stuff+ keeps 1:
each shuffle retrains its bagged model.

## 2026-11-15 · CFB player profile 1.0.2 → 1.0.3, CFB system fit 1.1.2 → 1.1.3 (patch: re-parse trigger)
`ingest/cfb/play_players.py` records a PARSER_VERSION per season and re-parses any cached season parsed by an older
one, so CI's cached 2025 file picks up the receiver fix. Parsing and every output are identical to 1.0.2 / 1.1.2;
metrics unchanged.

## 2026-11-15 · CFB player profile 1.0.1 → 1.0.2, CFB system fit 1.1.1 → 1.1.2 (patch: receiver parsing)
CFB-RECV-FIX. `ingest/cfb/play_players.py` (in both cards' code) missed the receiver in 2025's jersey-style play text
("to #81 X.Phipps caught at ASU49"), in "pass complete. Catch made by X", and in "A pass to B for 80 yds". Completions
naming a receiver: 2025 57.8% → 99.9%, 2026 3.4% → 99.9%, 2024 98.7% → 99.9%; 2014-2020 byte-identical. Found by
FIT-STABLE (receivers' yards per target fell from 11.2 in 2024 to 5.8 in 2025). A new data check requires >= 95% of
completions to name a receiver in every season. Profile validation: movers r 0.1818 → 0.1795 (n 480 → 494), all
players 0.2636 → 0.2651, split-half 0.2876 → 0.2958. On the portal board, WR entrants' total receiving yards rise
70,550 → 119,052. System fit's backtest was not rerun (its windows' outcomes are 2021-2025 and its inputs before 2025
are unchanged except a few hundred 2021-2024 rows). Not fixed (a source gap): CFBD's text names the receiver on only
21-54% of incompletions in 2021-2024, so target-based receiver stats are biased in those seasons.

## 2026-11-15 · CFB system fit 1.1.0 → 1.1.1 (patch: board display)
`cfb/portal/board.py` (in the card's code) also builds each program's top 25 among entrants who reached last season's
backup minimum, shown only when a viewer turns on the "Backup minimum" filter (Matt). The model's own list stays the
default and is the one frozen in the ledger; predictions are unchanged.

## 2026-11-15 · CFB system fit 1.0.1 → 1.1.0 (minor: the portal board's training data)
SC-3a. For the public portal board the model trains on transfer cohorts through 2024 only, so no 2026 outcome is used
before the Phase 2 vault opens (PLAN-fit); system and type dimensions are fit on complete seasons (through 2025), and a
live loader may read the season just played as of the board's date. `cfb/portal/board.py` joins the card's code. The
backtest path (fit_fold with no cap) is unchanged; still experimental on 1.0.0's result.

## 2026-11-15 · CFB system fit 1.0.0 → 1.0.1 (patch)
`cfb/fit/labels.py` (in the card's code) gains value over replacement (SYSTEM-FIT-EXPLORE, Matt: replacement = backups'
volume-weighted adjusted EPA/play per position, level, season). New columns only: the success labels, the model's
features, and its predictions are unchanged, so 1.0.0's backtest result stands; still experimental.

## 2026-11-15 · CFB system fit 1.0.0 (new, experimental: failed its pre-registered gate)
SYSTEM-FIT F4/F5. Pre-registered the same day before any model code (commit 7f3c350): recall at 10 on each FBS
program's portal board, windows 2021-2024, success within two years; must beat transfer rating, raw production, and
profile only, each with the 90% interval above zero (Matt). Run once (run 070d3961c0ba): fit 0.148 vs. 0.142 / 0.153 /
0.151 (random 0.045), n 311; edges +0.005, -0.005, -0.003, intervals about ±0.04. Failed. Secondary (cfb/fit/output/
f4.json): recall at 25 behind raw production (0.302 vs. 0.351); AUC among signees 0.644 vs. 0.628 / 0.668 / 0.611;
calibration overconfident at the top. The label-permutation leakage check also failed in that run (shuffled-label models beat
uniform random by +0.027 on average over 20 seeds): with a board-recall metric, any linear ranker puts players with real
production at the extremes. Matt approved a fair null: the mean credit of 20 random-direction rankers on the same
features. Rerun 28ef15e46370 (same code and data): primary identical (0.1479), every leakage check passes
(permutation edge over the null +0.028, 90% interval -0.013 to +0.072).

## 2026-11-15 · CFB player profile 1.0.0 → 1.0.1 (patch)
Share features (target share, carry share, dropback share, runs per dropback) now divide by the team's pass or run plays counted by play id instead of matched dropbacks, capped at 1. Found in SYSTEM-FIT F2: teams whose passers weren't matched to rosters (mostly FCS) produced shares up to 74. The opponent-adjusted features the pre-registered gate tests are unaffected.

## 2026-11-14 · Stuff+, Location+, Pitching+ 2.0.0: bagged training (PQ-STABLE)
**Why.** Checking free-agent grades (MLB-FA) against the published ones found per-pitcher Stuff+ up to 15 points apart between a local build and CI. Experiments (same data and code): Windows and Linux gave byte-identical models, so neither the OS nor floating point was the cause; changing only the seed (7 → 8) moved Stuff+ by 3.0 points on average and up to 64 (Pitching+ 2.1, up to 42). Each regressor was one LightGBM model early-stopped on a single 15% pitcher slice; the Stuff+ models stopped after 12–100 trees, and which pitchers fell in the slice (or a tiny data revision, which is how CI's cached raw data differed from the local copy) moved the stopping point and with it the grades. This corrects the 2026-11-13 entry below, which put cross-machine differences down to CPU models: the instability, not the hardware, made them large.

**What changed.** Each pitch-group regressor is now a bag of 10: pitchers split into 10 folds by a seeded shuffle, model f trains on the other nine folds and early-stops on fold f (member seed 700 + f), and predictions are the bag average. Stuff+ learning rate 0.03 → 0.005 (its models now run 130–650 trees). Model files are no longer committed (90 files; the version records keep their hashes). New `baseball/pitch_quality/stability.py` retrains with another seed and reports the per-pitcher differences.

**Results.** Seed sensitivity (562 MLB pitchers with 300+ pitches, seeds 7 vs 8): Stuff+ mean 0.35, max 3.5 (was 3.0 / 64); Pitching+ 0.38 / 2.7 (was 2.1 / 42); Location+ 0.12 / 0.6. For scale, a typical pitcher's 90% sampling interval is about ±2 points for Stuff+ and ±6 for Pitching+. Frozen eval set (r with next-season run value): Stuff+ 0.297 → 0.305, Pitching+ 0.318 → 0.318, Location+ 0.012 → 0.008. Split-half reliability unchanged (Stuff+ 0.994). Training takes about 64 minutes on 4 cores (was about 4); robustness.py about 25.

## 2026-11-14 · Pitching+ 1.0.1 → 2.0.0 (promoted by exception, Matt)
Frozen eval set: r_next_rv 0.3183 → 0.3179, lower by less than the ±0.01 reproduction tolerance, so `registry promote` declined. Matt approved promotion anyway (2026-11-14): 1.0.1's value was itself seed noise, and the shared training code (bagging) is what makes the grades stable. Live: not yet available (no ledger forecasts).

## 2026-11-14 · Location+ 1.0.1 → 2.0.0 (promoted by exception, Matt)
Frozen eval set: r_next_rv 0.01156 → 0.007667, lower by less than the ±0.01 reproduction tolerance, so `registry promote` declined. Matt approved promotion anyway (2026-11-14): 1.0.1's value was itself seed noise, and the shared training code (bagging) is what makes the grades stable. Live: not yet available (no ledger forecasts).

## 2026-11-14 · Stuff+ 1.0.1 → 2.0.0 (promoted)
Frozen eval set: Out-of-sample year over year: models refit on the earliest season only, grade the next season, and correlate with the season after's run value (MLB pitchers with enough pitches both years). Same split as the published Stuff+ validation.. r_next_rv: 0.297 → 0.3046. Live: not yet available (no ledger forecasts).

## 2026-11-13 · Stuff+ through the backtest harness (A5): reproduction
The harness reproduces `robustness.py` exactly on the same machine (n 324; r 0.3014 vs last-season RV 0.1926). The published 0.2970 was trained on a different CPU model: LightGBM trees differ slightly across hardware (also the cause of the refresh drift fixed in OPS-0). Not a leak or a harness bug. Reproduction standard (Matt): exact on one machine, ±0.01 in r across machines; official backtests run in CI. Pass gate pre-registered as strict (Matt): the edge over last season's run value must have a 90% interval excluding zero.

## 2026-11-13 · Stuff+, Location+, Pitching+ 1.0.0 → 1.0.1 (patch)
`build_features()` now takes batter heights as an argument instead of always reading the raw file, so the backtest harness (A5) can supply them through the point-in-time loader. With no argument it reads the same file as before: identical features and models; validation metrics unchanged (r_next_rv 0.297, 0.012, 0.318).

## 2026-11-13 · Registry started (A2): six models at v1.0.0
Registered as published: Stuff+, Location+, Pitching+ (out-of-sample r with next-season run value
0.297, 0.012, 0.318), Triple-A → MLB translation (hitter CV R² 0.046), NFL opponent adjustment
(next-week margin MAE 10.39), CFB power rating (MAE 12.55 vs. Elo 12.68). Records:
`models/<sport>/<model>/versions/1.0.0.json`. No model changed.

## 2026-11-13 · Stage layout (A0)
No model changed. Ingest moved to `ingest/`, data checks and contracts added; published outputs
verified byte-identical before and after on the same machine and data.
