"""Baseball plausibility tests, on the committed site data (no build intermediates needed).

Scales, as the pages state them (site/assets/js/sparvi.js S.SCALES, glossary "Scales"):
  - Stuff+, Location+, Pitching+ and RV+ are on the MLB scale at both levels: 100 and the 10-point SD are the
    qualified MLB pitchers of the season before (baseball/pitch_quality/train.py `scale`, meta.scale_reference;
    the snapshot marts reuse the same reference). So the MLB league mean must sit at 100, and the Triple-A mean must
    sit below it. RV+ is that ruler on results against the pitcher's own level, so it sits near 100 at both levels.
  - Percentiles in the Triple-A explorer and the Rule 5 board are on the Triple-A scale: rank among qualified
    Triple-A players, so the qualified pool must average 50 and follow the sorted values.

Tolerances are set from the 2026 files with room for a normal season, and each says what it allows for:
  MLB_FINAL_TOL   2.0  the scale is frozen on the prior season, so the current season drifts a little (2026 final
                       date: 99.1 to 99.3 pitch-weighted, all four metrics).
  MLB_WEEKLY_TOL  4.0  early weeks are small samples and velocity builds through spring (2026: 96.6 to 102.9 across
                       every weekly snapshot with 100+ qualified pitchers, both windows).
  AAA_BAND   (75, 98)  Triple-A on the MLB scale (2026: Stuff+ 90 to 94, Location+ 85 to 91, Pitching+ 83 to 86 across
                       weekly snapshots). At 98+ Triple-A would be graded like the majors; under 75 it would be 2.5 MLB
                       SDs short, which no full league is.
  AAA_RV_BAND (94, 106) Triple-A results against Triple-A hitters on the MLB ruler (2026: 96.6 to 102.5).
  GAP_RATIO (0.75, 1.5) Pitching+ is its own model, so its Triple-A shortfall in runs need not equal the Stuff+ shortfall
                       plus the Location+ one, but it must be close (2026: 0.56 vs 0.49 runs per 100 pitches, ratio 1.14).
                       This is the test that the Triple-A Pitching+ mean (about 85, below both components) comes from
                       the scale and not from a miscalculation: the shortfalls add, the SD they are divided by doesn't.
"""
from __future__ import annotations

import math

from plausibility.common import Suite, sd, wmean

DATA = "site/data/baseball"
PLUS = ["stuff_plus", "loc_plus", "pitching_plus", "rv_plus"]
GRADES = PLUS[:3]
NAMES = {"stuff_plus": "Stuff+", "loc_plus": "Location+", "pitching_plus": "Pitching+", "rv_plus": "RV+"}
REF_KEY = {"stuff_plus": "stuff", "loc_plus": "loc", "pitching_plus": "pitching", "rv_plus": "rv"}
MLB_FINAL_TOL, MLB_WEEKLY_TOL = 2.0, 4.0
AAA_BAND, AAA_RV_BAND, GAP_RATIO = (75.0, 98.0), (94.0, 106.0), (0.75, 1.5)
MIN_QUALIFIED = 100          # weekly tests start once this many pitchers (or hitters) qualify
ORGS = 30

LB, AAA, R5, PP, FARM, TR, FA, TWO = ("Stuff+ leaderboard", "Triple-A explorer", "Rule 5 board", "Player pages",
                                       "Farm systems", "Triple-A translation", "Free-agent pitchers", "Two-week signal")
ALT = "Triple-A altitude"


def snap_rows(snap: dict, window: str, i: int, present: str) -> list[dict]:
    """Rows at snapshot i (what Sparvi.snapRows gives the pages): players with a value for `present` then."""
    out = []
    for p in snap["players"]:
        s = p["series"].get(window)
        if s and s[present][i] is not None:
            out.append({"id": p["id"], "name": p.get("name"), **{k: s[k][i] for k in snap["fields"]}})
    return out


def pitch_rows(snap: dict, i: int) -> list[dict]:
    return [{"id": p["id"], "pt": p["pt"], "group": p["group"], **{k: p["series"][k][i] for k in snap["fields"]}}
            for p in snap["pitches"] if p["series"]["pitches"][i] is not None]


def league_mean(rows: list[dict], key: str, weight: str = "pitches") -> float | None:
    return wmean([r[key] for r in rows], [r[weight] if r[key] is not None else None for r in rows])


def monotone(rows: list[dict], value: str, pctl: str) -> tuple[int, bool]:
    """(number of adjacent pairs out of order, higher value = higher percentile?) among rows sorted by value."""
    v = sorted((r[value], r[pctl]) for r in rows if r[value] is not None and r[pctl] is not None)
    if len(v) < 3:
        return 0, True
    up = v[-1][1] >= v[0][1]
    bad = sum(1 for a, b in zip(v, v[1:], strict=False) if a[0] < b[0] and ((b[1] < a[1]) if up else (b[1] > a[1])))
    return bad, up


def run(suite: Suite) -> None:
    meta = suite.load(f"{DATA}/meta.json")
    ref = meta["scale_reference"]
    leaderboard(suite, meta, ref)
    explorer(suite, meta)
    rule5(suite, meta)
    player_pages(suite)
    farm(suite)
    translation(suite)
    free_agents(suite)
    altitude(suite)
    two_week(suite)


# -- Stuff+ leaderboard (and every page that shows the grades) -----------------------------------------------------------
def leaderboard(suite: Suite, meta: dict, ref: dict) -> None:
    g = {(lvl, win): suite.load(f"{DATA}/snapshots_grades_{lvl}{'' if win == 'std' else '_last30'}.json")
         for lvl in ("mlb", "aaa") for win in ("std", "last30")}
    final = {lvl: snap_rows(g[lvl, "std"], "std", len(g[lvl, "std"]["dates"]) - 1, "pitches") for lvl in ("mlb", "aaa")}
    mlb, aaa = final["mlb"], final["aaa"]

    # MLB scale: the MLB league mean is 100 (season to date, final date), and 10 points is one SD among qualifiers.
    for k in PLUS:
        suite.mean_near(LB, f"MLB league mean of {NAMES[k]} is 100 on the MLB scale (final date, pitch-weighted)",
                        [r[k] for r in mlb], 100, MLB_FINAL_TOL, [r["pitches"] if r[k] is not None else None for r in mlb])
    q = [r for r in mlb if r["qualified"]]
    for k in PLUS:
        s = sd([r[k] for r in q if r[k] is not None])
        suite.within(LB, f"Qualified MLB pitchers' {NAMES[k]} has an SD near 10 (10 points = 1 SD)", s, 7.5, 12.5, ".1f", "SD")

    # ...and stays there every week, both windows, once enough pitchers qualify.
    worst, n_weeks = (0.0, ""), 0
    for win in ("std", "last30"):
        snap = g["mlb", win]
        for i, d in enumerate(snap["dates"]):
            rows = snap_rows(snap, win, i, "pitches")
            if sum(1 for r in rows if r["qualified"]) < MIN_QUALIFIED:
                continue
            n_weeks += 1
            for k in PLUS:
                m = league_mean(rows, k)
                if m is not None and abs(m - 100) > worst[0]:
                    worst = (abs(m - 100), f"{NAMES[k]} {m:.1f} on {d} ({win})")
    suite.check(LB, f"MLB league mean of every + metric within {MLB_WEEKLY_TOL:g} of 100 at every weekly snapshot",
                n_weeks > 0 and worst[0] <= MLB_WEEKLY_TOL, f"largest gap: {worst[1] or 'none'}; {n_weeks} snapshots x windows")

    # Triple-A on the MLB scale: below MLB, inside the band, every week.
    means = {k: league_mean(aaa, k) for k in PLUS}
    mlb_means = {k: league_mean(mlb, k) for k in PLUS}
    for k in GRADES:
        m = means[k]
        suite.check(LB, f"Triple-A league mean of {NAMES[k]} is below MLB's and inside {AAA_BAND[0]:g}-{AAA_BAND[1]:g} (MLB scale)",
                    m is not None and AAA_BAND[0] <= m <= AAA_BAND[1] and m < mlb_means[k],
                    f"Triple-A {m:.1f}, MLB {mlb_means[k]:.1f} (final date, pitch-weighted)")
    suite.within(LB, "Triple-A league mean of RV+ is near 100 (MLB ruler, results against Triple-A hitters)", means["rv_plus"],
                 *AAA_RV_BAND, ".1f", "mean")
    out, n_weeks = [], 0
    for win in ("std", "last30"):
        snap = g["aaa", win]
        for i, d in enumerate(snap["dates"]):
            rows = snap_rows(snap, win, i, "pitches")
            if sum(1 for r in rows if r["qualified"]) < MIN_QUALIFIED:
                continue
            n_weeks += 1
            for k in PLUS:
                m, (lo, hi) = league_mean(rows, k), AAA_RV_BAND if k == "rv_plus" else AAA_BAND
                if m is None or not lo <= m <= hi:
                    out.append(f"{NAMES[k]} {m} on {d} ({win})")
    suite.check(LB, "Triple-A league means stay inside their bands at every weekly snapshot", n_weeks > 0 and not out,
                f"{len(out)} outside, e.g. {out[0]}" if out else f"{n_weeks} snapshots x windows")

    # Pitching+ below both components in Triple-A: explained by the scale. In runs per 100 pitches the Pitching+
    # shortfall must be about the Stuff+ shortfall plus the Location+ one.
    gap = {k: (100 - means[k]) / 10 * ref[REF_KEY[k]]["sd"] for k in GRADES}
    parts = gap["stuff_plus"] + gap["loc_plus"]
    ratio = gap["pitching_plus"] / parts if parts > 0 else math.nan
    suite.check(LB, "Triple-A Pitching+ shortfall equals the Stuff+ and Location+ shortfalls combined, in runs",
                parts > 0 and GAP_RATIO[0] <= ratio <= GAP_RATIO[1],
                f"Pitching+ {gap['pitching_plus']:.3f} vs Stuff+ {gap['stuff_plus']:.3f} + Location+ {gap['loc_plus']:.3f}"
                f" = {parts:.3f} runs per 100 pitches (ratio {ratio:.2f}, allowed {GAP_RATIO[0]:g} to {GAP_RATIO[1]:g});"
                f" in MLB SDs: {gap['pitching_plus'] / ref['pitching']['sd']:.2f} vs"
                f" {gap['stuff_plus'] / ref['stuff']['sd']:.2f} and {gap['loc_plus'] / ref['loc']['sd']:.2f}")
    sds = {k: ref[REF_KEY[k]]["sd"] for k in GRADES}
    suite.check(LB, "The Pitching+ reference SD is larger than either component's and smaller than their sum",
                max(sds["stuff_plus"], sds["loc_plus"]) < sds["pitching_plus"] < sds["stuff_plus"] + sds["loc_plus"],
                f"Stuff+ {sds['stuff_plus']:.3f}, Location+ {sds['loc_plus']:.3f}, Pitching+ {sds['pitching_plus']:.3f} runs per 100 pitches")
    suite.check(LB, "The frozen scale is the season before the one shown",
                ref["applies_to_season"] == meta["season"] and ref["reference_season"] == meta["season"] - 1,
                f"reference {ref['reference_season']}, applied to {ref['applies_to_season']}, site season {meta['season']}")

    # Qualified pitchers' grades are in a believable range (unqualified ones can be anything: a few pitches). The
    # Triple-A floor is lower: its wildest qualified pitchers really are 9 MLB SDs short on Location+ (2026 lowest:
    # 10 and 12, two pitchers who walked 27% and 23% of their batters; under Location+ 2.0.0 the same two were lowest,
    # at 25 and 29), which no one would be allowed to do for 150 MLB pitches.
    for lvl, rows in final.items():
        lo = 40 if lvl == "mlb" else 5
        vals = [(r[k], r["name"], k) for r in rows if r["qualified"] for k in PLUS if r[k] is not None]
        bad = [v for v in vals if not lo <= v[0] <= 160]
        suite.check(LB, f"Qualified {lvl.upper()} pitchers' + grades are between {lo} and 160", bool(vals) and not bad,
                    f"{len(bad)} outside, e.g. {bad[0][1]} {NAMES[bad[0][2]]} {bad[0][0]}" if bad else f"{len(vals):,} grades")

    # Pitch-type grades: MLB scale within each pitch group.
    for lvl in ("mlb", "aaa"):
        pt = suite.load(f"{DATA}/snapshots_pitches_{lvl}.json")
        rows = pitch_rows(pt, len(pt["dates"]) - 1)
        for grp in ("fastball", "breaking", "offspeed"):
            gr = [r for r in rows if r["group"] == grp]
            ms = {k: league_mean(gr, k) for k in GRADES}
            shown = ", ".join(f"{NAMES[k]} {ms[k]:.1f}" for k in GRADES)
            if lvl == "mlb":
                suite.check(LB, f"MLB {grp} pitch grades average 100 within their group (±3)",
                            all(abs(ms[k] - 100) <= 3 for k in GRADES), shown)
            else:
                suite.check(LB, f"Triple-A {grp} pitch grades average below 100 and above 80 (MLB scale)",
                            all(80 <= ms[k] < 100 for k in GRADES), shown)

    # Counting columns only go up over the season, and the season table agrees with the last snapshot.
    for lvl in ("mlb", "aaa"):
        drops = 0
        for p in g[lvl, "std"]["players"]:
            s = p["series"]["std"]
            for k in ("pitches", "games", "bf"):
                v = [x for x in s[k] if x is not None]
                drops += sum(1 for a, b in zip(v, v[1:], strict=False) if b < a)
        suite.check(LB, f"{lvl.upper()} season-to-date pitches, games, and batters faced never decrease from week to week",
                    drops == 0, f"{drops} decreases" if drops else f"{len(g[lvl, 'std']['players']):,} pitchers")
    table = suite.load(f"{DATA}/stuff_pitchers.json")
    keys = [(r["id"], r["level"]) for r in table]
    suite.check(LB, "One row per pitcher and level in the season table", len(keys) == len(set(keys)), f"{len(keys):,} rows")
    by = {(r["id"], lvl.upper()): r for lvl, rows in final.items() for r in rows}
    diff = [r for r in table for k in PLUS
            if (r["id"], r["level"]) in by and by[r["id"], r["level"]][k] is not None and r[k] is not None
            and abs(by[r["id"], r["level"]][k] - r[k]) > 0.051]
    matched = sum((r["id"], r["level"]) in by for r in table)
    suite.check(LB, "Season-table grades equal the final snapshot's (one scale in both files)",
                not diff and matched >= 0.95 * len(table),
                f"{len(diff)} differ, e.g. {diff[0]['name']}" if diff else f"{matched:,} of {len(table):,} pitchers matched")


# -- Triple-A explorer -------------------------------------------------------------------------------------------------
def explorer(suite: Suite, meta: dict) -> None:
    for side, present, pool in (("pitchers", "pitches", "pitchers"), ("hitters", "pa", "hitters")):
        for win in ("std", "last30"):
            snap = suite.load(f"{DATA}/snapshots_{side}_aaa{'' if win == 'std' else '_last30'}.json")
            pct = [f for f in snap["fields"] if f.endswith("_pctl")]
            rates = [f for f in snap["fields"] if f.endswith("_pct")]
            out_range, off_mean, disorder, bad_rate, n_snap, teams_max = [], [], [], [], 0, 0
            for i, d in enumerate(snap["dates"]):
                rows = snap_rows(snap, win, i, present)
                teams_max = max(teams_max, len({r["team"] for r in rows if r["team"]}))
                for r in rows:
                    out_range += [f"{r['name']} {k} {r[k]} on {d}" for k in pct if r[k] is not None and not 0 <= r[k] <= 100]
                    bad_rate += [f"{r['name']} {k} {r[k]} on {d}" for k in rates if r[k] is not None and not 0 <= r[k] <= 1]
                q = [r for r in rows if r["qualified"]]
                if len(q) < MIN_QUALIFIED:
                    continue
                n_snap += 1
                for k in pct:
                    m = wmean([r[k] for r in q])
                    if m is None or abs(m - 50) > 2:
                        off_mean.append(f"{k} {m} on {d}")
                    bad, _ = monotone(q, k[:-5], k)
                    if bad:
                        disorder.append(f"{k} on {d}: {bad} pairs out of order")
            w = "season to date" if win == "std" else "last 30 days"
            suite.check(AAA, f"{side.title()}: percentiles are between 0 and 100 ({w})", not out_range,
                        f"{len(out_range)} outside, e.g. {out_range[0]}" if out_range else f"{len(pct)} percentile columns")
            suite.check(AAA, f"{side.title()}: qualified Triple-A {pool} average the 50th percentile (Triple-A scale, {w})",
                        n_snap > 0 and not off_mean,
                        f"{len(off_mean)} off by more than 2, e.g. {off_mean[0]}" if off_mean else f"{n_snap} snapshots, {len(pct)} columns")
            suite.check(AAA, f"{side.title()}: percentiles follow the sorted values among qualified {pool} ({w})",
                        n_snap > 0 and not disorder, disorder[0] if disorder else f"{n_snap} snapshots, {len(pct)} columns")
            suite.check(AAA, f"{side.title()}: rates are between 0 and 1 ({w})", not bad_rate,
                        f"{len(bad_rate)} outside, e.g. {bad_rate[0]}" if bad_rate else f"{len(rates)} rate columns")
            if win == "std":
                suite.check(AAA, f"{side.title()}: no more than {ORGS} Triple-A clubs", 0 < teams_max <= ORGS, f"{teams_max} clubs")
                ids = [p["id"] for p in snap["players"]]
                suite.check(AAA, f"{side.title()}: one row per player", len(ids) == len(set(ids)), f"{len(ids):,} players")

    # Percentile orientation: higher is always better (the page says so).
    p = suite.load(f"{DATA}/snapshots_pitchers_aaa.json")
    q = [r for r in snap_rows(p, "std", len(p["dates"]) - 1, "pitches") if r["qualified"]]
    want = {"k_pct": True, "csw_pct": True, "whiff_pct": True, "bb_pct": False, "hard_hit_pct": False, "xwoba": False}
    wrong = [k for k, up in want.items() if monotone(q, k, f"{k}_pctl")[1] != up]
    h = suite.load(f"{DATA}/snapshots_hitters_aaa.json")
    qh = [r for r in snap_rows(h, "std", len(h["dates"]) - 1, "pa") if r["qualified"]]
    want_h = {"xwoba": True, "ev90": True, "hard_hit_pct": True, "zone_contact_pct": True, "bb_pct": True, "chase_pct": False,
              "k_pct": False}
    wrong += [f"hitter {k}" for k, up in want_h.items() if monotone(qh, k, f"{k}_pctl")[1] != up]
    suite.check(AAA, "Percentiles point the right way: higher is better for every column", not wrong,
                f"reversed: {', '.join(wrong)}" if wrong else f"{len(want)} pitcher and {len(want_h)} hitter columns")
    velo = [r["fb_velo"] for r in q if r["fb_velo"] is not None]
    suite.within(AAA, "Qualified Triple-A pitchers' average fastball velocity is believable (mph)", wmean(velo), 88, 97, ".1f", "mean")
    xw = wmean([r["xwoba"] for r in qh])
    suite.within(AAA, "Qualified Triple-A hitters' average xwOBA is believable", xw, 0.28, 0.38, ".3f", "mean")


# -- Rule 5 board ------------------------------------------------------------------------------------------------------
def rule5(suite: Suite, meta: dict) -> None:
    pit, hit = suite.load(f"{DATA}/rule5_pitchers.json"), suite.load(f"{DATA}/rule5_hitters.json")
    table = {r["id"]: r for r in suite.load(f"{DATA}/stuff_pitchers.json") if r["level"] == "AAA"}
    both = [r for r in pit if r["player_id"] in table]
    diff = [r["name"] for r in both for k in GRADES
            if r[k] is not None and table[r["player_id"]][k] is not None and abs(r[k] - table[r["player_id"]][k]) > 0.051]
    suite.check(R5, "Pitchers' Stuff+, Location+, Pitching+ equal their Triple-A leaderboard grades (MLB scale)",
                not diff and len(both) >= 0.9 * len(pit),
                f"{len(diff)} differ, e.g. {diff[0]}" if diff else f"{len(both)} of {len(pit)} pitchers matched")
    q = [r for r in pit if r["pitches"] >= 300 and r["stuff_plus"] is not None]
    m = wmean([r["stuff_plus"] for r in q])
    suite.within(R5, "Eligible pitchers (300+ pitches) average below 100 Stuff+: unprotected Triple-A arms, MLB scale", m, 80, 100,
                 ".1f", "mean")
    for name, rows in (("pitchers", pit), ("hitters", hit)):
        ids = [r["player_id"] for r in rows]
        suite.check(R5, f"One row per player ({name})", len(ids) == len(set(ids)) and len(ids) > 0, f"{len(ids)} players")
        orgs = {r["org"] for r in rows}
        suite.check(R5, f"Every player belongs to one of {ORGS} organizations ({name})", 0 < len(orgs) <= ORGS and None not in orgs,
                    f"{len(orgs)} organizations")
        bad = [r["name"] for r in rows if r["status"] not in meta["rule5"]["statuses"]]
        suite.check(R5, f"Every eligibility status is one the page explains ({name})", not bad,
                    f"e.g. {bad[0]}" if bad else ", ".join(meta["rule5"]["statuses"]))
        ages = [r["name"] for r in rows if r["age"] is not None and not 18 <= r["age"] <= 45]
        suite.check(R5, f"Ages are between 18 and 45 ({name})", not ages, f"e.g. {ages[0]}" if ages else "")
    org_p = {r["player_id"]: r["org"] for r in pit}
    clash = [r["name"] for r in hit if r["player_id"] in org_p and org_p[r["player_id"]] != r["org"]]
    suite.check(R5, "A player on both lists is in the same organization on each", not clash, f"e.g. {clash[0]}" if clash else "")
    bad = [r["name"] for r in pit if None not in (r["k_pct"], r["bb_pct"], r["k_bb_pct"])
           and abs(r["k_bb_pct"] - (r["k_pct"] - r["bb_pct"])) > 0.0011]
    suite.check(R5, "K-BB% is strikeout rate minus walk rate", not bad, f"e.g. {bad[0]}" if bad else f"{len(pit)} pitchers")
    pct = [k for k in hit[0] if k.endswith("_pctl")]
    bad = [f"{r['name']} {k} {r[k]}" for r in hit for k in pct if r[k] is not None and not 0 <= r[k] <= 100]
    suite.check(R5, "Hitters' percentiles are between 0 and 100 (Triple-A scale)", not bad, f"e.g. {bad[0]}" if bad else f"{len(pct)} columns")
    sn = suite.load(f"{DATA}/snapshots_hitters_aaa.json")
    last = {r["id"]: r for r in snap_rows(sn, "std", len(sn["dates"]) - 1, "pa")}
    cmp_ = [(r, last[r["player_id"]]) for r in hit if r["player_id"] in last]
    diff = [r["name"] for r, s in cmp_ for k in pct if k in s and r[k] is not None and s[k] is not None and abs(r[k] - s[k]) > 0.5]
    suite.check(R5, "Hitters' percentiles equal the Triple-A explorer's at the final date", not diff and len(cmp_) >= 0.9 * len(hit),
                f"{len(diff)} differ, e.g. {diff[0]}" if diff else f"{len(cmp_)} of {len(hit)} hitters matched")


# -- Player pages ------------------------------------------------------------------------------------------------------
def player_pages(suite: Suite) -> None:
    urls, index = suite.load(f"{DATA}/player_urls.json"), suite.load(f"{DATA}/players_index.json")
    suite.check(PP, "One entry per player in the index", len(index) == len(set(index)), f"{len(index):,} players")
    shown = {r["id"] for r in suite.load(f"{DATA}/stuff_pitchers.json")}
    shown |= {r["player_id"] for f in ("rule5_pitchers", "rule5_hitters", "aaa_pitchers", "aaa_hitters")
              for r in suite.load(f"{DATA}/{f}.json")}
    missing = [i for i in shown if str(i) not in urls]
    suite.check(PP, "Every player listed in a tool has a player page", not missing,
                f"{len(missing)} without one, e.g. id {missing[0]}" if missing else f"{len(shown):,} players")
    slugs = [u for v in urls.values() for u in v]
    suite.check(PP, "No two players share a page address", len(slugs) == len(set(slugs)), f"{len(slugs):,} addresses")


# -- Farm systems ------------------------------------------------------------------------------------------------------
def farm(suite: Suite) -> None:
    rows = suite.load(f"{DATA}/farm_orgs.json")
    bad = [f"{r['org']} {r['metric']} {r['season']}" for r in rows if not r["lo"] <= r["score"] <= r["hi"]]
    suite.check(FARM, "Every organization's score is inside its own interval", not bad, f"e.g. {bad[0]}" if bad else f"{len(rows)} scores")
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["metric"], r["season"]), []).append(r)
    short = [k for k, g in groups.items() if len({r["org"] for r in g}) != ORGS or len(g) != ORGS]
    suite.check(FARM, f"{ORGS} organizations, once each, for every metric and season", not short,
                f"e.g. {short[0]}" if short else f"{len(groups)} metric-seasons")
    grew = [f"{r['org']} {r['metric']} {r['season']}" for r in rows if abs(r["score"]) > abs(r["raw"]) + 1e-9]
    suite.check(FARM, "Shrinking never makes a score larger than the raw improvement", not grew, f"e.g. {grew[0]}" if grew else "")
    # Scores are improvements relative to the league, in each metric's own units: the 30 organizations center near 0.
    off = []
    for k, g in groups.items():
        m, s = wmean([r["score"] for r in g]), sd([r["raw"] for r in g])
        if s == 0 or abs(m) > 0.25 * s:
            off.append(f"{k[0]} {k[1]}: mean {m:.3f} vs raw SD {s:.3f}")
    suite.check(FARM, "League-average improvement is about zero for every metric and season", not off,
                off[0] if off else f"{len(groups)} metric-seasons")


# -- Triple-A translation ----------------------------------------------------------------------------------------------
def translation(suite: Suite) -> None:
    fits = suite.load(f"{DATA}/translation_fits.json")
    bad, n = [], 0
    for side in ("hitter", "pitcher"):
        for k, v in fits[side].items():
            n += 1
            if not (0 <= v["slope"] <= 1.5 and v["band"][0] < 0 < v["band"][1]):
                bad.append(f"{side} {k}: slope {v['slope']}, band {v['band']}")
    suite.check(TR, "Every translation line has a slope between 0 and 1.5 and a band around it", not bad, bad[0] if bad else f"{n} lines")
    # The small-sample line (50-149 PA/BF, aaa_translation 2.0.0): it must start where the main line's minimum leaves off,
    # shrink more at 50 than at 149, widen its range as the sample shrinks, and send an average Triple-A line to a
    # believable MLB one (within the range the main line gives for the same value).
    sm = fits.get("small")
    if suite.check(TR, "The small-sample translation covers the plate appearances just under the main line's minimum",
                   bool(sm) and sm["max_pa"] + 1 == fits["min_aaa"] and 25 <= sm["min_pa"] < sm["max_pa"],
                   f"{sm['min_pa']} to {sm['max_pa']}, main line from {fits['min_aaa']}" if sm else "no small-sample constants"):
        bad, n = [], 0
        for side in ("hitter", "pitcher"):
            for k, c in sm[side].items():
                n += 1
                half = [sm["z80"] * math.sqrt(c["base"] + c["slope"] ** 2 * c["talent"] * (1 - pa / (pa + c["k"]))) for pa in (sm["min_pa"], sm["max_pa"])]
                at_mean = c["intercept"] + c["slope"] * c["mean"]
                main = fits[side].get(k)
                ok = (0 < c["slope"] <= 1.5 and 5 <= c["k"] <= 1000 and c["talent"] > 0 and c["base"] >= 0 and half[0] > half[1] > 0
                      and (main is None or main["band"][0] <= at_mean - (main["intercept"] + main["slope"] * c["mean"]) <= main["band"][1]))
                if not ok:
                    bad.append(f"{side} {k}: slope {c['slope']}, k {c['k']}, half-widths {half}, average line -> {at_mean}")
        suite.check(TR, "Every small-sample line has a positive slope, a believable reliability, a range that widens as the "
                        "sample shrinks, and agrees with the main line for an average player", not bad, bad[0] if bad else f"{n} lines")
    # A league-average Triple-A pitcher on the MLB scale translates to a below-average big leaguer, not an average one.
    g = suite.load(f"{DATA}/snapshots_grades_aaa.json")
    rows = snap_rows(g, "std", len(g["dates"]) - 1, "pitches")
    for k in ("stuff_plus", "pitching_plus"):
        if k in fits["pitcher"]:
            f = fits["pitcher"][k]
            t = f["intercept"] + f["slope"] * league_mean(rows, k)
            suite.within(TR, f"The average Triple-A {NAMES[k]} translates to a below-average MLB grade (MLB scale)", t, 75, 100, ".1f",
                         "translated")
    pairs = suite.load(f"{DATA}/translation_pairs_pitcher.json")
    vals = [r[f"{k}_{lv}"] for r in pairs for k in GRADES for lv in ("aaa", "mlb") if r.get(f"{k}_{lv}") is not None]
    bad = [v for v in vals if not 20 <= v <= 180]
    suite.check(TR, "Pitcher pairs' grades are between 20 and 180 at both levels", bool(vals) and not bad,
                f"{len(bad)} outside, e.g. {bad[0]}" if bad else f"{len(vals):,} grades")
    up = [r for r in pairs if r.get("stuff_plus_aaa") is not None and r.get("stuff_plus_mlb") is not None]
    d = wmean([r["stuff_plus_mlb"] - r["stuff_plus_aaa"] for r in up])
    # 2026: +3.9. Pitchers who moved up grade a little higher in the majors than they did in Triple-A (they are called
    # up when they are throwing well, and the leagues' balls and tracking differ), so the scale is shared but not
    # perfectly level-neutral. Past 6 points (0.6 MLB SD) "directly comparable" would no longer be a fair description.
    suite.within(TR, "The same pitcher's Stuff+ is about the same at both levels (one MLB scale, no level adjustment)", d, -6, 6,
                 "+.1f", "mean MLB minus Triple-A")


# -- Free-agent pitchers -----------------------------------------------------------------------------------------------
def free_agents(suite: Suite) -> None:
    fa = suite.load(f"{DATA}/free_agents.json")["pitchers"]
    bad = [r["name"] for r in fa for k in ("stuff", "loc", "pitching") if not r[f"{k}_lo"] <= r[f"{k}_plus"] <= r[f"{k}_hi"]]
    suite.check(FA, "Every grade is inside its own interval", not bad, f"e.g. {bad[0]}" if bad else f"{len(fa)} pitchers")
    bad = [r["name"] for r in fa if abs(r["gap"] - (r["rv_plus"] - r["pitching_plus"])) > 0.051]
    suite.check(FA, "The results gap is RV+ minus Pitching+", not bad, f"e.g. {bad[0]}" if bad else "")
    ids = [r["id"] for r in fa]
    suite.check(FA, "One row per pitcher", len(ids) == len(set(ids)), f"{len(ids)} pitchers")
    m = wmean([r["stuff_plus"] for r in fa], [r["pitches"] for r in fa])
    suite.within(FA, "Free agents' average Stuff+ is near the MLB average (they are MLB pitchers, MLB scale)", m, 92, 108, ".1f", "mean")


# -- Triple-A grades at altitude ---------------------------------------------------------------------------------------
def altitude(suite: Suite) -> None:
    d = suite.load(f"{DATA}/altitude.json")
    rows, parks = d["pitchers"], d["parks"]
    ids = [r["id"] for r in rows]
    suite.check(ALT, "One row per pitcher in the altitude table", len(ids) == len(set(ids)) and len(ids) >= 300, f"{len(ids)} pitchers")
    bad = [r["name"] for r in rows if abs(r["stuff_adj"] - (r["stuff_plus"] + r["delta"])) > 0.051]
    suite.check(ALT, "The adjusted Stuff+ is the published Stuff+ plus the change", not bad, f"e.g. {bad[0]}" if bad else f"{len(rows)} pitchers")
    pub = {r["id"]: r["stuff_plus"] for r in suite.load(f"{DATA}/stuff_pitchers.json") if r["level"] == "AAA"}
    bad = [r["name"] for r in rows if pub.get(r["id"]) != r["stuff_plus"]]
    suite.check(ALT, "The as-recorded Stuff+ in the altitude table is the leaderboard's", not bad, f"e.g. {bad[0]}" if bad else "")
    bad = [p["club"] for p in parks if not (0.6 <= p["lo"] <= p["factor"] <= p["hi"] <= 1.1)]
    suite.check(ALT, "Every park records between 60% and 110% of MLB movement, inside its own interval", bool(parks) and not bad,
                f"e.g. {bad[0]}" if bad else f"{len(parks)} parks, {min(p['factor'] for p in parks):.2f} to {max(p['factor'] for p in parks):.2f}")
    hi, lo = [p["factor"] for p in parks if p["altitude"]], [p["factor"] for p in parks if not p["altitude"]]
    suite.check(ALT, "The high-altitude parks record less movement than every other park", bool(hi) and bool(lo) and max(hi) < min(lo),
                f"altitude up to {max(hi):.3f}, others from {min(lo):.3f}" if hi and lo else "missing")
    # A park correction is a few points: a change beyond 20 would mean the factors were applied wrong (e.g. multiplied).
    big = [r["name"] for r in rows if abs(r["delta"]) > 20]
    suite.check(ALT, "No pitcher's grade changes by more than 20 points", not big,
                f"e.g. {big[0]}" if big else f"largest {max(abs(r['delta']) for r in rows):.1f}")


# -- Two-week signal ---------------------------------------------------------------------------------------------------
def two_week(suite: Suite) -> None:
    res = suite.load(f"{DATA}/two_week_results.json")
    rs = [(side, k, v["r"]) for side in ("hitters", "pitchers") for k, v in res[side].items() if isinstance(v, dict) and "r" in v]
    bad = [f"{s} {k} r {r}" for s, k, r in rs if r is None or not -0.2 <= r <= 1]
    suite.check(TWO, "Block-to-block correlations are between -0.2 and 1", bool(rs) and not bad, bad[0] if bad else f"{len(rs)} metrics")
    velo = res["pitchers"].get("fb_velo", {}).get("r")
    rest = [r for s, k, r in rs if s == "pitchers" and k != "fb_velo"]
    suite.check(TWO, "Fastball velocity repeats more than any other pitcher number", velo is not None and all(velo > r for r in rest),
                f"velocity r {velo}, next {max(rest):.3f}" if rest and velo is not None else "missing")
