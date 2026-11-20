"""Before/after report for a college football rebuild: what a change to the CFB pipeline did to the published numbers.

    python -m cfb.qa.rebuild_report origin/main > report.md     # after cfb/build.sh, in the changed checkout

"Before" is each file as committed at the base ref; "after" is the working tree (just rebuilt). Reads only what is
committed or published (metrics.json, the site's team-week files, the status files): aggregates, no plays.
Used by .github/workflows/cfb-rebuild-report.yml, which posts the output on the pull request.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter

from sparvi.config import ROOT, cfb

METRICS = "cfb/adjusted_ratings/output/metrics.json"
INTEGRITY = "site/data/status/cfb_pbp_integrity.json"
GT = "no_gt"


def before(ref: str, rel: str):
    r = subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    return json.loads(r.stdout) if r.returncode == 0 else None


def after(rel: str):
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _get(d, dotted):
    for k in dotted.split("."):
        d = d.get(k) if isinstance(d, dict) else None
    return d


def metrics_table(b: dict, a: dict) -> list[str]:
    rows = [("Next-week margin MAE, power rating", "backtest.power.mae", ".3f"), ("MAE, Elo alone", "backtest.elo.mae", ".3f"),
            ("MAE, efficiency alone", "backtest.ours.mae", ".3f"), ("MAE, schedule-adjusted EPA", "backtest.schedule_adjusted.mae", ".3f"),
            ("MAE, raw EPA", "backtest.raw_epa.mae", ".3f"), ("MAE, closing line", "backtest.line_close.mae", ".3f"),
            ("Power minus Elo MAE", "backtest.power_minus_elo_mae", "+.3f"), ("Backtest games", "backtest.n_games", ",d"),
            ("WEPA test MAE, ours", "wepa_test.ours.mae", ".3f"), ("Ridge lambda", "lambda", ",d"),
            ("Blend: points per 1.00 net EPA/play", "power_model.points_per_net_epa", ".2f"),
            ("Blend: points per Elo point", "power_model.points_per_elo", ".4f"),
            ("Blend: home field, points", "power_model.home_points", ".2f"), ("Blend: games fit", "power_model.n_games", ",d")]
    out = ["| | Before | After | Change |", "|---|---|---|---|"]
    for label, path, fmt in rows:
        x, y = _get(b, path), _get(a, path)
        if x is None and y is None:
            continue
        plain = fmt.replace("+", "")
        shown = ["n/a" if v is None else format(v, plain) for v in (x, y)]
        d = "none" if x is None or y is None or x == y else format(y - x, "+" + plain)
        out.append(f"| {label} | {shown[0]} | {shown[1]} | {d} |")
    ci = _get(a, "backtest.power_minus_elo_mae_ci90")
    if ci:
        out += ["", f"Power minus Elo, 90% interval after: {ci[0]:+.3f} to {ci[1]:+.3f}."]
    return out


def ratings(ref: str, season: int) -> list[str]:
    rel = f"site/data/cfb/team_week_{season}.json"
    b, a = before(ref, rel), after(rel)
    if not b or not a:
        return [f"No {season} team-week file on one side."]
    wk = min(len(b["weeks"]), len(a["weeks"])) - 1
    at = lambda f: {t["id"]: {k: t["series"][GT][k][wk] for k in ("power", "rank", "off_epa_adj", "def_epa_adj")}  # noqa: E731
                    for t in f["teams"]}
    x, y = at(b), at(a)
    both = [t for t in y if t in x and x[t]["power"] is not None and y[t]["power"] is not None]
    moved = sorted(both, key=lambda t: -abs(y[t]["power"] - x[t]["power"]))
    rank_moves = [t for t in both if x[t]["rank"] != y[t]["rank"]]
    out = [f"{season}, week {a['weeks'][wk]}, {len(both)} rated teams: power changed for "
           f"{sum(abs(y[t]['power'] - x[t]['power']) >= 0.05 for t in both)} teams (largest "
           f"{max((abs(y[t]['power'] - x[t]['power']) for t in both), default=0):.1f} points); rank changed for "
           f"{len(rank_moves)} (largest {max((abs(y[t]['rank'] - x[t]['rank']) for t in both), default=0)} places).", ""]
    if moved and abs(y[moved[0]]["power"] - x[moved[0]]["power"]) >= 0.05:
        out += ["| Team | Power before | after | Rank before | after |", "|---|---|---|---|---|"]
        out += [f"| {t} | {x[t]['power']:+.1f} | {y[t]['power']:+.1f} | {x[t]['rank']} | {y[t]['rank']} |" for t in moved[:10]
                if abs(y[t]["power"] - x[t]["power"]) >= 0.05]
    return out


def all_seasons(ref: str, seasons: list[int]) -> list[str]:
    """Team-weeks whose power or adjusted EPA changed, by season: where the repaired games sit."""
    out = ["| Season | Team-weeks | Power changed | Largest, points | Adjusted EPA changed | Largest, EPA/play |", "|---|---|---|---|---|---|"]
    for s in seasons:
        rel = f"site/data/cfb/team_week_{s}.json"
        b, a = before(ref, rel), after(rel)
        if not b or not a:
            continue
        bt = {t["id"]: t["series"][GT] for t in b["teams"]}
        n = pw = ep = 0
        dp = de = 0.0
        for t in a["teams"]:
            if t["id"] not in bt:
                continue
            for k in range(min(len(a["weeks"]), len(b["weeks"]))):
                n += 1
                p0, p1 = bt[t["id"]]["power"][k], t["series"][GT]["power"][k]
                if p0 is not None and p1 is not None and p0 != p1:
                    pw += 1; dp = max(dp, abs(p1 - p0))
                for f in ("off_epa_adj", "def_epa_adj"):
                    e0, e1 = bt[t["id"]][f][k], t["series"][GT][f][k]
                    if e0 is not None and e1 is not None and e0 != e1:
                        ep += 1; de = max(de, abs(e1 - e0))
        out.append(f"| {s} | {n:,} | {pw:,} | {dp:.1f} | {ep:,} | {de:.3f} |")
    return out


def integrity() -> list[str]:
    t = after(INTEGRITY)
    if not t:
        return ["No integrity table was written."]
    if not t.get("built"):
        return ["The integrity table is not from a build (`built: false`): the rebuild did not run."]
    out = [f"{t['games_checked']:,} games checked, {t['plays']:,} plays kept, {t['plays_in_repaired_games']:,} of them in repaired games.",
           "", "By class: " + ", ".join(f"{k} {v:,}" for k, v in t["by_class"].items()) + ".",
           "By decision (games that don't match): " + ", ".join(f"{k} {v:,}" for k, v in t["by_decision"].items()) + ".", ""]
    listed = Counter((g["class"], g["decision"]) for g in t["games"])
    out += ["Listed on /status/: " + ", ".join(f"{c} → {d}: {n}" for (c, d), n in sorted(listed.items())) + ".", "",
            "| Season | Game | Result | Play-by-play | Decision |", "|---|---|---|---|---|"]
    out += [f"| {g['season']} | {g['away']} at {g['home']} | {g['result']} | {g['pbp']} | {g['decision']} |"
            for g in t["games"] if g["class"] == "swapped" or g["decision"] != "keep"]
    return out


def plausibility() -> list[str]:
    p = after("site/data/status/cfb_plausibility.json")
    if not p:
        return ["No plausibility result."]
    bad = [c for c in p["checks"] if c["ok"] is False]
    out = [f"{p['n_checks']} tests, {p['n_failed']} failed, {p.get('n_not_run', 0)} not run."]
    out += [f"- FAILED {c['tool']}: {c['name']}: {c['detail']}" for c in bad]
    out += [f"- {c['tool']}: {c['name']}: {c['detail']}" for c in p["checks"] if c["tool"] == "Play-by-play integrity" and c["ok"]]
    return out


def main() -> int:
    ref = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    cfg = cfb()
    b, a = before(ref, METRICS), after(METRICS)
    lines = ["## College football rebuild: before and after", "",
             f"Before: files as committed at `{ref}`. After: `cfb/build.sh` on this branch, from the cached raw data "
             "(no CFBD calls). If the cache is newer than the last committed refresh, some difference is new games, "
             "not this change: compare the backtest game counts.", "", "### Backtest and blend", ""]
    lines += metrics_table(b or {}, a or {}) if a else ["metrics.json is missing after the build."]
    lines += ["", "### Play-by-play integrity", ""] + integrity()
    lines += ["", "### Current ratings", ""] + ratings(ref, cfg.season)
    lines += ["", "### Every season", ""] + all_seasons(ref, cfg.seasons)
    lines += ["", "### Plausibility tests on the rebuilt data", ""] + plausibility()
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
