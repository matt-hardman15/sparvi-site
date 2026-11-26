"""NFL plausibility tests (see plausibility/common.py).

Published (site/data/nfl): the opponent adjustment moves teams by a non-trivial but bounded amount and averages
zero; its penalty is inside the searched range and recovers most of a known schedule effect in simulation; offense
and defense totals agree; every game is in both teams' logs, mirrored; the fourth-down grid offers no impossible
choice and its labels follow its numbers; team fourth-down totals add up. Build only (the dbt database and the
schedule): no play credits a team that didn't play in the game.
"""
from __future__ import annotations

import os
from pathlib import Path

from plausibility.common import Suite, sd

EFF, GAMES, FOURTH, MATCHUPS = "Team efficiency", "Team pages and game logs", "Fourth-down decisions", "Offense or defense"
MU = "Matchup tool"
# sd(adjusted - raw) / sd(raw) EPA/play across the 32 teams, at the last week of a season. Below the floor the
# "opponent-adjusted" view is the raw view (Nov 2026, with the forecast penalty: 0.03 on offense, 0.04 on defense;
# with its own penalties the smallest in any season is 0.06, on offense, whose correction comes from the more heavily
# shrunk defenses). Above the ceiling the correction moves teams by more than an NFL schedule can (simulated truth:
# about 0.2).
ADJ_SIZE = (0.05, 0.45)
ADJ_MAX_MOVE = 0.12            # EPA/play: the largest correction for any team from week 4 on
ADJ_FROM_WEEK = 4
SIM_MIN_RECOVERED = 0.5        # share of a known schedule effect recovered over a simulated full season
EPA_RANGE = (-0.6, 0.6)        # season-to-date EPA/play from week 4 on
SR_RANGE = (0.25, 0.65)        # success rate from week 4 on
P_TOL = 2e-4                   # the grid's probabilities are rounded to 4 places
# The conversion model is the seasons' observed rate by yards to go (models/nfl/fourth_conversion), so one more yard
# can look slightly easier by chance (Nov 2026: 4th and 7, 42.7%, vs. 4th and 6, 40.7%). More than this is not noise.
CONVERT_NOISE = 0.03


def _team_weeks(suite: Suite, season: int):
    f = suite.load(f"site/data/nfl/team_week_{season}.json")
    return f["weeks"], {t["id"]: t["series"] for t in f["teams"]}


def efficiency(suite: Suite) -> None:
    meta = suite.load("site/data/nfl/meta.json")
    size_bad, mean_bad, move_bad, range_bad, mirror_bad, sizes = [], [], [], [], [], []
    for season in meta["seasons"]:
        weeks, teams = _team_weeks(suite, season)
        for k, wk in enumerate(weeks):
            for gt in ("no_gt", "all"):
                for side in ("off", "def"):
                    raw = [s[gt][f"{side}_epa"][k] for s in teams.values()]
                    adj = [s[gt][f"{side}_epa_adj"][k] for s in teams.values()]
                    pairs = [(r, a) for r, a in zip(raw, adj, strict=True) if r is not None or a is not None]
                    if any(r is None or a is None for r, a in pairs) or len(pairs) < len(teams) - 4:
                        range_bad.append(f"{season} week {wk}: {side} EPA/play missing (raw without adjusted, or 5+ teams)")
                        continue
                    raw, adj = [r for r, _ in pairs], [a for _, a in pairs]      # a team may not have played yet
                    diff = [a - r for a, r in zip(adj, raw, strict=True)]
                    if wk >= ADJ_FROM_WEEK and abs(sum(diff) / len(diff)) > 0.02:
                        mean_bad.append(f"{season} week {wk} {side}: mean correction {sum(diff) / len(diff):+.4f}")
                    if wk >= ADJ_FROM_WEEK:
                        if max(abs(d) for d in diff) > ADJ_MAX_MOVE:
                            move_bad.append(f"{season} week {wk} {side}: {max(abs(d) for d in diff):.3f}")
                        if not all(EPA_RANGE[0] <= v <= EPA_RANGE[1] for v in raw + adj):
                            range_bad.append(f"{season} week {wk} {side}: EPA/play outside {EPA_RANGE}")
                        sr = [v for v in (s[gt][f"{side}_sr"][k] for s in teams.values()) if v is not None]
                        if not all(SR_RANGE[0] <= v <= SR_RANGE[1] for v in sr):
                            range_bad.append(f"{season} week {wk} {side}: success rate outside {SR_RANGE}")
                    if k == len(weeks) - 1 and wk >= ADJ_FROM_WEEK:
                        ratio = sd(diff) / sd(raw)
                        if gt == "no_gt":
                            sizes.append(ratio)
                        if not ADJ_SIZE[0] <= ratio <= ADJ_SIZE[1]:
                            size_bad.append(f"{season} week {wk} {side} ({gt}): {ratio:.3f}")
                # Every play is one team's offense and another's defense: the league totals must agree.
                tot = {side: sum(s[gt][f"{side}_epa"][k] * s[gt][f"{side}_plays"][k] for s in teams.values()
                                 if s[gt][f"{side}_epa"][k] is not None) for side in ("off", "def")}
                n = sum(s[gt]["off_plays"][k] or 0 for s in teams.values())
                n_def = sum(s[gt]["def_plays"][k] or 0 for s in teams.values())
                if n != n_def or (n and abs(tot["off"] - tot["def"]) / n > 2e-4):
                    mirror_bad.append(f"{season} week {wk} ({gt}): offense {n:,} plays {tot['off']:+.1f} EPA,"
                                      f" defense {n_def:,} plays {tot['def']:+.1f}")
    eg = lambda bad: f"{len(bad)} cases, e.g. {bad[0]}"  # noqa: E731
    suite.check(EFF, "opponent adjustment has a non-trivial, bounded size", not size_bad and bool(sizes),
                eg(size_bad) + f" (allowed {ADJ_SIZE[0]:g} to {ADJ_SIZE[1]:g} of the spread between teams)" if size_bad
                else f"sd(adjusted - raw) is {min(sizes):.2f} to {max(sizes):.2f} of sd(raw) at each season's last week"
                     f" (allowed {ADJ_SIZE[0]:g} to {ADJ_SIZE[1]:g})")
    suite.check(EFF, f"no team's correction exceeds {ADJ_MAX_MOVE:g} EPA/play from week {ADJ_FROM_WEEK} on", not move_bad,
                eg(move_bad) if move_bad else "")
    suite.check(EFF, "the correction averages zero across teams", not mean_bad, eg(mean_bad) if mean_bad else "")
    suite.check(EFF, "EPA/play and success rate are in a possible range", not range_bad, eg(range_bad) if range_bad else "")
    suite.check(EFF, "league offense and defense totals agree (each play counted once on each side)", not mirror_bad,
                eg(mirror_bad) if mirror_bad else "")

    a = meta.get("adjustment", {})
    lam_s, sim = a.get("lambda_schedule"), a.get("simulation_recovered")
    suite.check(EFF, "the schedule correction has its own penalty, lighter than the forecast penalty",
                bool(lam_s) and max(lam_s) < a.get("lambda", 0),
                f"schedule {lam_s[0]} (offenses) and {lam_s[1]} (defenses), forecast {a.get('lambda')}" if lam_s
                else "meta.json has no lambda_schedule: the correction still uses the forecast penalty")
    suite.check(EFF, "in simulation the correction recovers most of a known schedule effect",
                bool(sim) and sim["full_season"] >= SIM_MIN_RECOVERED,
                f"{sim['full_season']:.0%} over a full season, {sim['through_week']:.0%} through this week"
                f" (needs {SIM_MIN_RECOVERED:.0%})" if sim else "no simulation result in meta.json")
    ordered = a.get("mae_ridge", 0) < min(a.get("mae_adjusted", 0), a.get("mae_raw", 0)) and \
        max(a.get("mae_adjusted", 99), a.get("mae_raw", 99)) < a.get("mae_home_only", 0)
    suite.check(EFF, "margin forecasts: shrunk ratings beat adjusted and raw, and all beat home field alone", ordered,
                f"MAE {a.get('mae_ridge', 0):.2f} / {a.get('mae_adjusted', 0):.2f} / {a.get('mae_raw', 0):.2f} /"
                f" {a.get('mae_home_only', 0):.2f}")


def game_logs(suite: Suite) -> None:
    meta = suite.load("site/data/nfl/meta.json")
    ids = {t["id"] for t in suite.load("site/data/nfl/teams.json")}
    bad, dup, n = [], [], 0
    for season in meta["seasons"]:
        logs = suite.load(f"site/data/nfl/team_games_{season}.json")["teams"]
        for team, rows in logs.items():
            wks = [r["week"] for r in rows]
            if len(wks) != len(set(wks)):
                dup.append(f"{season} {team}")
            for r in rows:
                n += 1
                opp = r["opp"]
                if opp not in ids or opp == team:
                    bad.append(f"{season} week {r['week']}: {team} vs. {opp!r}")
                    continue
                m = next((x for x in logs.get(opp, []) if x["week"] == r["week"]), None)
                # The opponent's log has the same game from the other side: scores exchanged, one home team, and
                # this team's offense is the opponent's defense (same plays, same EPA).
                if (m is None or m["opp"] != team or (m["pf"], m["pa"]) != (r["pa"], r["pf"])
                        or (r["home"] and m["home"])
                        or any(r.get(f"off_{k}") != m.get(f"def_{k}") for k in ("epa", "plays", "ng_epa", "ng_plays"))):
                    bad.append(f"{season} week {r['week']}: {team} vs. {opp}")
    suite.check(GAMES, "every game is in both teams' logs, mirrored (scores, home team, offense = opponent's defense)",
                not bad and n > 0, f"{len(bad)} of {n:,} rows, e.g. {bad[0]}" if bad else f"{n:,} team-games")
    suite.check(GAMES, "no team has two games in one week", not dup, f"{len(dup)}, e.g. {dup[0]}" if dup else "")


def fourth_downs(suite: Suite) -> None:
    f = suite.load("site/data/nfl/fourth_downs.json")
    grid, eng = f["grid"], f["engine"]
    fg_max = eng["fg_max_yardline"]
    from nfl.fourth.decide import PUNT_MIN_YARDLINE
    bad = []
    none_ys = [g["y"] for g in grid if g["call"] == "none"]
    called_ys = [g["y"] for g in grid if g["call"] != "none"]
    # "No call" is a stretch of field at the offense's own end, not scattered cells.
    no_call_from = min(none_ys) if none_ys else None
    if none_ys and called_ys and max(called_ys) >= no_call_from:
        bad.append(f"calls and no-calls overlap: a call at the {max(called_ys)}, no call at the {no_call_from}")
    if f.get("no_call_from") is not None and no_call_from != f["no_call_from"]:
        bad.append(f"the file says no call from the {f['no_call_from']} back, the grid stops calling at the {no_call_from}")
    if len(none_ys) > len(grid) / 2:
        bad.append(f"{len(none_ys)} of {len(grid)} situations have no call")
    for g in grid:
        y, t = g["y"], g["t"]
        where = f"4th and {t} at the {y}"
        if g["call"] == "none":      # deep in its own end the chart makes no call: then it must show no numbers either
            if any(g.get(k) is not None for k in ("go", "fg", "punt", "m", "lo", "hi", "kick", "pc", "pf")) or not t <= y <= 99:
                bad.append(f"{where}: no call, but numbers are shown")
            elif no_call_from is not None and y < no_call_from:
                bad.append(f"{where}: no call made closer than the offense's own {100 - no_call_from}")
            continue
        probs = [g[k] for k in ("go", "fg", "punt", "pc", "pf") if g.get(k) is not None]
        kicks = {k: g[k] for k in ("fg", "punt") if g.get(k) is not None}
        if t > y or t < 1 or not 1 <= y <= 99:
            bad.append(f"{where}: more yards to go than yards to the goal line")
        elif any(not 0 <= p <= 1 for p in probs) or g.get("go") is None:
            bad.append(f"{where}: a probability outside 0-1")
        elif (g.get("fg") is not None) != (y <= fg_max):
            bad.append(f"{where}: a field goal is offered only inside the {fg_max:g}")
        elif (g.get("punt") is not None) != (y >= PUNT_MIN_YARDLINE):
            bad.append(f"{where}: a punt is offered only from the {PUNT_MIN_YARDLINE} or farther")
        elif not kicks or g["kick"] not in kicks or kicks[g["kick"]] < max(kicks.values()) - P_TOL:
            bad.append(f"{where}: the best kick isn't the better of the kicks offered")
        elif abs(g["m"] - (g["go"] - kicks[g["kick"]])) > 2 * P_TOL:
            bad.append(f"{where}: margin isn't go minus the best kick")
        elif g["lo"] > g["hi"] or not {"go": g["lo"] >= 0, "kick": g["hi"] <= 0,
                                       "toss-up": g["lo"] <= 0 <= g["hi"]}.get(g["call"], False):
            bad.append(f"{where}: call {g['call']!r} doesn't follow its interval ({g['lo']}, {g['hi']})")
    cells = {(g["y"], g["t"]) for g in grid}
    suite.check(FOURTH, "no impossible decision in the grid (distance, options offered, probabilities, call vs. interval)",
                not bad and len(cells) == len(grid) and len(grid) > 0,
                f"{len(bad)} of {len(grid)} situations, e.g. {bad[0]}" if bad
                else f"{len(grid)} situations" + (f", no call from the offense's own {100 - no_call_from} back" if none_ys else ""))
    # A longer kick (among those offered) is never likelier to be good, and more yards to go is not likelier to be
    # converted beyond the noise in the observed rates.
    mono = []
    by_y = sorted({(g["y"], g["pf"]) for g in grid if g.get("fg") is not None})
    mono += [f"field goal from the {b[0]} ({b[1]}) likelier than from the {a[0]} ({a[1]})"
             for a, b in zip(by_y, by_y[1:], strict=False) if b[1] > a[1] + P_TOL]
    for y in sorted({g["y"] for g in grid}):
        col = sorted((g["t"], g["pc"]) for g in grid if g["y"] == y and g["pc"] is not None)
        mono += [f"at the {y}: 4th and {b[0]} converts more often ({b[1]}) than 4th and {a[0]} ({a[1]})"
                 for a, b in zip(col, col[1:], strict=False) if b[1] > a[1] + CONVERT_NOISE]
    suite.check(FOURTH, "longer kicks and longer conversions are never likelier", not mono,
                f"{len(mono)} cases, e.g. {mono[0]}" if mono else f"kicks offered inside the {fg_max:g}; conversions within {CONVERT_NOISE:g}")

    tb, by_season = [], {}
    for r in f["teams"]:
        ok = (r["clear_go"] + r["clear_kick"] + r["toss"] == r["n"] and r["went_go"] <= r["clear_go"]
              and r["went_kick"] <= r["clear_kick"] and 0 <= r["went"] <= r["n"] and r["cost"] >= 0
              and abs(r["cost"] - r["cost_passive"] - r["cost_aggr"]) < 1e-3
              and (r["rate"] is None or 0 <= r["rate"] <= 1))
        if not ok:
            tb.append(f"{r['season']} {r['team']}")
        s = by_season.setdefault(r["season"], {"n": 0, "clear_go": 0, "clear_kick": 0, "toss": 0, "cost": 0.0})
        for k in s:
            s[k] += r[k]
    for lg in f["league"]:
        s = by_season.get(lg["season"], {})
        if any(s.get(k) != lg[k] for k in ("n", "clear_go", "clear_kick", "toss")) or abs(s["cost"] - lg["cost"]) > 0.01:
            tb.append(f"{lg['season']} league total differs from the sum of its teams")
    suite.check(FOURTH, "team totals are coherent and add up to the league's", not tb and bool(f["teams"]),
                f"{len(tb)}, e.g. {tb[0]}" if tb else f"{len(f['teams'])} team-seasons")
    cb = [f"{c['season']} week {c['week']} {c['team']}" for c in f["costly"]
          if not (c["did"] in ("go", "fg", "punt") and 0 <= c["actual"] <= 1 and c["t"] <= c["y"]
                  and c["actual"] <= max(c["go"], c["kick"]) + P_TOL)]
    suite.check(FOURTH, "each costly call lost win probability against a choice that was available", not cb,
                f"{len(cb)}, e.g. {cb[0]}" if cb else f"{len(f['costly'])} calls")


def matchups(suite: Suite) -> None:
    d = suite.load("site/data/nfl/offense_defense.json")
    rows = []
    for g in d["games"]:
        for unit in ("away_off", "home_off"):
            for split, m in g[unit].items():
                rows.append({"split": split, "side": "off", "v": m["off"], "rank": m["off_rank"]})
                rows.append({"split": split, "side": "def", "v": -m["def"], "rank": m["def_rank"]})   # lower allowed is better
    suite.ranks_match(MATCHUPS, "unit ranks follow the adjusted EPA/play shown", rows, "rank", "v", by=("split", "side"),
                      exact=False)


def attribution(suite: Suite) -> None:
    name = "no team's offense or defense is credited in a game it didn't play"
    db = Path(os.environ.get("FOOTBALL_DB_PATH", suite.path("football/dbt/football.duckdb")))
    sched = suite.path("data/raw/nfl/schedules.parquet")
    if not suite.needs(GAMES, name, db, sched):
        return
    import duckdb
    con = duckdb.connect(str(db), read_only=True)
    try:
        if not con.sql("select 1 from information_schema.tables where table_name = 'stg_nfl_plays'").fetchone():
            suite.not_run(GAMES, name, "not run here: the football database has no NFL plays")
            return
        n, bad, eg = con.sql(f"""
            select count(*), count(*) filter (where s.game_id is null
                       or not ((p.offense = s.home_team and p.defense = s.away_team)
                            or (p.offense = s.away_team and p.defense = s.home_team))), min(p.game_id) filter (
                       where s.game_id is null or p.offense not in (s.home_team, s.away_team)
                          or p.defense not in (s.home_team, s.away_team) or p.offense = p.defense)
            from stg_nfl_plays p left join read_parquet('{sched.as_posix()}') s using (game_id)""").fetchone()
    finally:
        con.close()
    suite.check(GAMES, name, bad == 0 and n > 0,
                f"{bad:,} of {n:,} plays, e.g. game {eg}" if bad else f"{n:,} plays, each between its game's two teams")


def run(suite: Suite) -> None:
    efficiency(suite)
    game_logs(suite)
    attribution(suite)
    fourth_downs(suite)
    matchups(suite)
    matchup_tool(suite)


# -- Matchup tool ------------------------------------------------------------------------------------------------------
def matchup_tool(suite: Suite) -> None:
    d = suite.load("site/data/nfl/matchup.json")
    line, rows = d["line"], d["teams"]
    ids = [r["id"] for r in rows]
    suite.check(MU, "One rating per team in the matchup tool", len(ids) == len(set(ids)) == 32, f"{len(ids)} teams")
    # Home field has been worth about 1.5 to 2.5 points and a typical game misses by 10 to 11: outside these the line is not a fit of NFL margins.
    ok = 0.5 <= line["home_points"] <= 4 and 10 <= line["margin_sd"] <= 17 and line["points_per_net"] > 0
    suite.check(MU, "The margin line is NFL-sized (home field 0.5 to 4 points, spread 10 to 17, a better rating adds points)", ok, str(line))
    bad = [r["id"] for r in rows if abs(r["net"] - (r["off"] - r["def"])) > 0.00011]
    suite.check(MU, "A team's net rating is its offense less what its defense allows", not bad, f"e.g. {bad[0]}" if bad else "")
    suite.ranks_match(MU, "Rank follows the net rating in the matchup tool", rows, "rank", "net")
    gap = line["points_per_net"] * (max(r["net"] for r in rows) - min(r["net"] for r in rows))
    suite.within(MU, "The best team is a 3 to 30 point favorite over the worst on a neutral field", gap, 3, 30, ".1f", "points")
    suite.check(MU, "The matchup tool is through the same week as the rest of the NFL pages",
                d["through_week"] == suite.load("site/data/nfl/meta.json")["last_week"], f"week {d['through_week']}")
