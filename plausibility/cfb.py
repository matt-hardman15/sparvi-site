"""College football plausibility tests (see plausibility/common.py).

Published (site/data/cfb, cfb/adjusted_ratings/output/metrics.json, site/data/status): ranks follow the power rating,
the rating is centered and equals its two published parts, the blend weights on the page are the fitted ones, the
opponent adjustment has a sensible size, each game is in both teams' logs with the same score, every score-gap game
has a decision, and the portal board's windows are coherent. Build only (the dbt database): no play credits a team
that didn't play in the game.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from plausibility.common import Suite, sd

RATINGS, GAMES, PBP, PORTAL = "Team ratings", "Team pages and game logs", "Play-by-play integrity", "Portal board"
MU = "Matchup tool"
GT = "no_gt"
ADJ_SIZE = (0.10, 1.00)        # sd(adjusted - raw) / sd(raw) EPA/play across FBS teams, end of season
PART_MIN_SHARE = 0.05          # each part of the blend moves teams by at least this share of the rating's spread
PARTS_TOL = 0.11               # power vs. pw_epa + pw_elo, each rounded to a tenth of a point
DECISIONS = {"keep", "repair_teams", "repair_scores", "keep_no_score", "exclude"}
CLASSES = {"swapped", "over", "under"}


def _weeks(suite: Suite):
    """(season, week, rows) for every season file: one dict per FBS team with the week's values."""
    meta = suite.load("site/data/cfb/meta.json")
    for season in meta["seasons"]:
        f = suite.load(f"site/data/cfb/team_week_{season}.json")
        for k, wk in enumerate(f["weeks"]):
            rows = []
            for t in f["teams"]:
                s = t["series"][GT]
                rows.append({"team": t["id"], **{fld: (s[fld][k] if fld in s else None) for fld in f["fields"]}})
            yield season, wk, rows


def _corr(a: list[float], b: list[float]) -> float:
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    den = sd(a) * sd(b) * len(a)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True)) / den if den else 0.0


def ratings(suite: Suite) -> None:
    meta = suite.load("site/data/cfb/meta.json")
    pm = meta.get("power_model", {})
    b_net, b_elo = pm.get("points_per_net_epa"), pm.get("points_per_elo")
    suite.check(RATINGS, "blend weights are published and positive", bool(b_net and b_elo and b_net > 0 and b_elo > 0),
                f"{b_net:.2f} points per 1.00 net EPA/play, {b_elo:.4f} per Elo point" if b_net and b_elo else "missing from meta.json")
    mfile = suite.path("cfb/adjusted_ratings/output/metrics.json")
    if mfile.exists() and b_net and b_elo:
        fit = json.loads(mfile.read_text(encoding="utf-8"))["power_model"]
        same = all(abs(fit[k] - pm[k]) < 1e-9 for k in ("points_per_net_epa", "points_per_elo", "home_points"))
        suite.check(RATINGS, "published blend weights are the fitted ones", same,
                    "" if same else f"meta.json {pm} vs. metrics.json {fit}")
        prose = suite.path("site/_generated/cfb.yml").read_text(encoding="utf-8")
        want = [f'pm_net: "{fit["points_per_net_epa"]:.1f}"', f'pm_elo: "{fit["points_per_elo"]:.4f}"']
        suite.check(RATINGS, "blend weights printed on the page are the fitted ones", all(w in prose for w in want),
                    ", ".join(want))

    bad_rank, bad_center, bad_parts, n_weeks, n_rows = [], [], [], 0, 0
    part_share, implied_r, adj_size = [], [], {"off": [], "def": []}
    last = {}
    for season, wk, rows in _weeks(suite):
        last[season] = (wk, rows)
        rated = [r for r in rows if r.get("power") is not None and r.get("rank") is not None]
        if not rated:
            continue
        n_weeks += 1; n_rows += len(rated)
        # Ranks follow the rating: published power is rounded to a tenth and the rank comes from the unrounded value,
        # so the test is that a higher shown rating never has a worse rank, ranks start at 1, and a rank is one more
        # than the number of teams ranked above (ties share a rank: 1, 2, 2, 4).
        by_rank = sorted(rated, key=lambda r: r["rank"])
        for i, r in enumerate(by_rank):
            above = sum(x["rank"] < r["rank"] for x in rated)
            if r["rank"] != above + 1 or (i and r["power"] > by_rank[i - 1]["power"]):
                bad_rank.append(f"{season} week {wk}: {r['team']} power {r['power']} rank {r['rank']}")
                break
        m = sum(r["power"] for r in rated) / len(rated)
        if abs(m) > 0.1:
            bad_center.append(f"{season} week {wk}: mean {m:+.2f}")
        if all(r.get("pw_epa") is not None and r.get("pw_elo") is not None for r in rated):
            worst = max(abs(r["power"] - r["pw_epa"] - r["pw_elo"]) for r in rated)
            if worst > PARTS_TOL:
                bad_parts.append(f"{season} week {wk}: off by {worst:.2f}")
    suite.check(RATINGS, "power ranks follow the power rating (ranked on the unrounded value; ties share a rank)", not bad_rank and n_rows > 0,
                f"{len(bad_rank)} weeks, e.g. {bad_rank[0]}" if bad_rank else f"{n_rows:,} team-weeks in {n_weeks} weeks")
    suite.check(RATINGS, "power rating is centered on the average FBS team every week", not bad_center,
                f"{len(bad_center)} weeks, e.g. {bad_center[0]}" if bad_center else f"{n_weeks} weeks within 0.1 points of zero")

    # The two parts, end of each season. Where the files carry them: they sum to the rating. Always: the Elo part
    # implied by the published weight leaves an EPA part that is the efficiency rating (it tracks net adjusted EPA
    # per play), and neither part is negligible.
    for season, (_wk, rows) in last.items():
        rated = [r for r in rows if all(r.get(k) is not None for k in ("power", "elo", "off_epa_adj", "def_epa_adj"))]
        if len(rated) < 20 or not b_elo:
            continue
        mean_elo = sum(r["elo"] for r in rated) / len(rated)
        elo_part = [r["pw_elo"] if r.get("pw_elo") is not None else b_elo * (r["elo"] - mean_elo) for r in rated]
        epa_part = [r["pw_epa"] if r.get("pw_epa") is not None else r["power"] - e for r, e in zip(rated, elo_part, strict=True)]
        spread = sd([r["power"] for r in rated])
        part_share.append((min(sd(epa_part), sd(elo_part)) / spread, season))
        implied_r.append((_corr(epa_part, [r["off_epa_adj"] - r["def_epa_adj"] for r in rated]), season))
        for side in ("off", "def"):
            raw = [r[f"{side}_epa"] for r in rated if r.get(f"{side}_epa") is not None]
            diff = [r[f"{side}_epa_adj"] - r[f"{side}_epa"] for r in rated if r.get(f"{side}_epa") is not None]
            adj_size[side].append((sd(diff) / sd(raw), season))
    if not bad_parts:
        suite.check(RATINGS, "power rating equals its EPA part plus its Elo part", bool(part_share),
                    f"{len(part_share)} seasons; where the two parts aren't in a file yet they follow from the published weights")
    else:
        suite.check(RATINGS, "power rating equals its EPA part plus its Elo part", False, f"{len(bad_parts)} weeks, e.g. {bad_parts[0]}")
    lo_r = min(implied_r, default=(0, None))
    suite.check(RATINGS, "the EPA part of the power rating is the efficiency rating", lo_r[0] >= 0.95,
                f"correlation with net adjusted EPA/play at least {lo_r[0]:.3f} (lowest: {lo_r[1]}; expected 0.95 or more)")
    lo_s = min(part_share, default=(0, None))
    suite.check(RATINGS, "neither part of the blend is negligible", lo_s[0] >= PART_MIN_SHARE,
                f"smaller part's spread is at least {lo_s[0]:.0%} of the rating's (lowest: {lo_s[1]}; expected {PART_MIN_SHARE:.0%} or more)")
    for side, label in (("off", "offense"), ("def", "defense")):
        v = adj_size[side]
        lo, hi = min(v, default=(0, None)), max(v, default=(0, None))
        suite.check(RATINGS, f"opponent adjustment of {label} EPA/play has a non-trivial, bounded size",
                    bool(v) and ADJ_SIZE[0] <= lo[0] and hi[0] <= ADJ_SIZE[1],
                    f"sd(adjusted − raw) is {lo[0]:.0%} ({lo[1]}) to {hi[0]:.0%} ({hi[1]}) of sd(raw) across FBS teams, end of season"
                    f" (allowed {ADJ_SIZE[0]:.0%} to {ADJ_SIZE[1]:.0%})")

    gaps = meta.get("poll_gaps")
    if gaps and meta["season"] in last:
        f = suite.load(f"site/data/cfb/team_week_{meta['season']}.json")
        k = f["weeks"].index(gaps["week"]) if gaps["week"] in f["weeks"] else None
        rank = {t["id"]: t["series"][GT]["rank"][k] for t in f["teams"]} if k is not None else {}
        bad = [r["team"] for r in gaps["rows"] if rank.get(r["team"]) != r["power"] or r["gap"] != r["power"] - r["ap"]]
        suite.check(RATINGS, "AP-vs-power gaps use the published power ranks", not bad and k is not None,
                    f"{len(bad)} teams, e.g. {bad[0]}" if bad else f"{len(gaps['rows'])} ranked teams, week {gaps['week']}")


def game_logs(suite: Suite) -> None:
    """Every game between two FBS teams is in both teams' logs, same week, with the scores mirrored: a team is never
    shown in a game its opponent didn't play."""
    meta = suite.load("site/data/cfb/meta.json")
    bad, n, too_many = [], 0, []
    for season in meta["seasons"]:
        logs = suite.load(f"site/data/cfb/team_games_{season}.json")["teams"]
        for team, rows in logs.items():
            if len(rows) > 16:
                too_many.append(f"{season} {team}: {len(rows)} games")
            for g in rows:
                if g["opp"] == team:
                    bad.append(f"{season} week {g['week']}: {team} plays itself")
                if g["opp"] not in logs:
                    continue
                n += 1
                mirror = [x for x in logs[g["opp"]] if x["week"] == g["week"] and x["opp"] == team
                          and x.get("pf") == g.get("pa") and x.get("pa") == g.get("pf")]
                if not mirror or (g.get("home") and mirror[0].get("home")):
                    bad.append(f"{season} week {g['week']}: {team} vs {g['opp']}")
    suite.check(GAMES, "every FBS-vs-FBS game is in both teams' logs with the same score", not bad and n > 0,
                f"{len(bad)} of {n:,} team-games, e.g. {bad[0]}" if bad else f"{n:,} team-games")
    suite.check(GAMES, "no team has more games in a season than a season holds", not too_many,
                too_many[0] if too_many else "16 at most")


def pbp(suite: Suite) -> None:
    listed = suite.load("site/data/status/cfb.json").get("score_gaps", [])
    f = suite.path("site/data/status/cfb_pbp_integrity.json")
    if not f.exists():
        suite.check(PBP, "every score-gap game has a documented decision", not listed, f"{len(listed)} games listed, no decision table")
        return
    t = json.loads(f.read_text(encoding="utf-8"))
    by_id = {g["id"]: g for g in t["games"]}
    allowed = DECISIONS | (set() if t.get("built") else {"pending"})
    missing = [g for g in listed if g["id"] not in by_id]
    undecided = [g for g in t["games"] if g.get("decision") not in allowed or g.get("class") not in CLASSES
                 or (g["class"] != "swapped" and g["decision"] in ("repair_teams", "repair_scores", "keep_no_score", "pending"))]
    pending = sum(g["decision"] == "pending" for g in t["games"])
    counts = {}
    for g in t["games"]:
        counts[g["decision"]] = counts.get(g["decision"], 0) + 1
    detail = (f"{len(missing)} listed games without a decision, e.g. {missing[0]['id']}" if missing else
              f"{len(undecided)} games with no valid decision, e.g. {undecided[0]['id']}" if undecided else
              f"{len(t['games'])} games: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
              + (f" ({pending} swapped games are repaired at the next build)" if pending else ""))
    suite.check(PBP, "every score-gap game has a documented decision", not missing and not undecided and bool(t.get("rule")), detail)
    swapped_listed = [g for g in t["games"] if g["class"] == "swapped"]
    wrong = [g["id"] for g in swapped_listed if g["decision"] == "keep"]
    suite.check(PBP, "no game with home and away exchanged is used as it came", not wrong,
                f"{len(wrong)} games, e.g. {wrong[0]}" if wrong
                else f"{len(swapped_listed)} swapped games; the repair is in the pipeline and takes effect at the next build (the ratings shown still use them as they came)" if pending
                else f"{len(swapped_listed)} swapped games, each repaired or its running score ignored")

    # Build only: the plays themselves.
    name = "no team's offense or defense is credited in a game it didn't play"
    db = Path(os.environ.get("FOOTBALL_DB_PATH", suite.path("football/dbt/football.duckdb")))
    raw = suite.path("data/raw/cfb")
    if not suite.needs(PBP, name, db, raw):
        return
    import duckdb
    con = duckdb.connect(str(db), read_only=True)
    try:
        if not con.sql("select 1 from information_schema.tables where table_name = 'stg_cfb_plays'").fetchall():
            suite.not_run(PBP, name, "not run here: the football database has no college plays")
            return
        g = f"read_parquet('{(raw / 'games_*.parquet').as_posix()}')"
        n, bad, eg = con.sql(f"""
            select count(*), count(*) filter (where g.id is null or p.offense = p.defense
                     or not ((p.offense = g.homeTeam and p.defense = g.awayTeam) or (p.offense = g.awayTeam and p.defense = g.homeTeam))),
                   min(p.game_id) filter (where g.id is null or p.offense = p.defense
                     or not ((p.offense = g.homeTeam and p.defense = g.awayTeam) or (p.offense = g.awayTeam and p.defense = g.homeTeam)))
            from stg_cfb_plays p left join {g} g on g.id = p.game_id""").fetchone()
        suite.check(PBP, name, bad == 0 and n > 0, f"{bad:,} of {n:,} plays, e.g. game {eg}" if bad else f"{n:,} plays, each between its game's two teams")
    finally:
        con.close()


def portal(suite: Suite) -> None:
    idx = suite.load("site/data/cfb/portal/index.json")["windows"]
    years = [w["window"] for w in idx]
    suite.check(PORTAL, "windows are listed once each, newest first", years == sorted(set(years), reverse=True) and bool(years), str(years))
    bad, unopened = [], []
    for w in idx:
        b = suite.load(f"site/data/cfb/portal/board_{w['window']}.json")
        if not (len(b["players"]) == w["players"] == b["entrants"]):
            bad.append(f"{w['window']}: index says {w['players']} entrants, the board has {len(b['players'])}")
        if w["players"] == 0:
            # An empty window is one that hasn't opened: it must say when it opens, and that can't be before its build date.
            if not w.get("opens") or w["opens"] < w["as_of"][:7]:
                unopened.append(f"{w['window']}: no entrants as of {w['as_of']}, opens {w.get('opens', 'not stated')}")
        for p in b["players"]:
            if (p.get("epa_pct") is not None and not 0 <= p["epa_pct"] <= 100) or (p.get("share") is not None and not 0 <= p["share"] <= 1):
                bad.append(f"{w['window']}: {p['name']} has percentile {p.get('epa_pct')} and share {p.get('share')}")
                break
    suite.check(PORTAL, "entrant counts match the boards, and shares and percentiles are in range", not bad,
                bad[0] if bad else ", ".join(f"{w['window']}: {w['players']:,} entrants" for w in idx))
    suite.check(PORTAL, "a window with no entrants is labeled with when it opens", not unopened,
                unopened[0] if unopened else ", ".join(f"{w['window']} opens {w['opens']}" for w in idx if w["players"] == 0) or "every window has entrants")


def matchup_tool(suite: Suite) -> None:
    meta = suite.load("site/data/cfb/meta.json")
    pm = meta.get("power_model", {})
    # A typical FBS game misses a good forecast by 12 to 13 points (root mean square about 16); home field is 2 to 3.
    ok = pm.get("margin_sd") is not None and 12 <= pm["margin_sd"] <= 20 and 1 <= pm.get("home_points", 0) <= 5
    suite.check(MU, "The margin's spread and home field are college-football-sized (spread 12 to 20 points, home 1 to 5)", ok,
                f"spread {pm.get('margin_sd')}, home {pm.get('home_points')}")
    wk = suite.load(f"site/data/cfb/team_week_{meta['season']}.json")
    slugs = {t["id"] for t in suite.load("site/data/cfb/teams.json")}
    rated = [(t["id"], t["series"][GT]["power"][-1]) for t in wk["teams"] if t["series"][GT]["power"][-1] is not None]
    lost = [n for n, _ in rated if n not in slugs]
    suite.check(MU, "Every team in the matchup tool has a team page", len(rated) >= 100 and not lost, f"e.g. {lost[0]}" if lost else f"{len(rated)} teams")
    if rated:
        gap = max(p for _, p in rated) - min(p for _, p in rated)
        suite.within(MU, "The best team is a 25 to 80 point favorite over the worst on a neutral field", gap, 25, 80, ".1f", "points")


def run(suite: Suite) -> None:
    ratings(suite)
    matchup_tool(suite)
    game_logs(suite)
    pbp(suite)
    portal(suite)
