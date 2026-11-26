"""Plausibility tests for the college basketball pages (CBB-PAGES-1): can the published ratings be true?

Published tests only (they read site/data/cbb). Bounds are from the 2026-27 ratings as first published and from what
the quantities are: a net rating is points per 100 possessions against an average Division I team, so across all of
Division I it averages about zero, and no team is 80 points per 100 possessions from average.
"""
from __future__ import annotations

from plausibility.common import Suite, sd, wmean

DATA = "site/data/cbb"
RATINGS = "Basketball team ratings"
TEAMS = "Basketball team pages"


def run(suite: Suite) -> None:
    d, meta = suite.load(f"{DATA}/ratings.json"), suite.load(f"{DATA}/meta.json")
    rows = d["teams"]
    ids = [r["id"] for r in rows]
    suite.check(RATINGS, "One row per Division I team", len(ids) == len(set(ids)) and 300 <= len(ids) <= 400 and len(ids) == meta["teams"],
                f"{len(ids)} teams")
    bad = [r["team"] for r in rows if abs(r["net"] - (r["off"] + r["def"])) > 0.11]
    suite.check(RATINGS, "Net rating is offense plus defense", not bad, f"e.g. {bad[0]}" if bad else f"{len(rows)} teams")
    suite.ranks_match(RATINGS, "Rank follows the net rating", rows, "rank", "net", exact=False)      # ratings are published rounded
    m = wmean([r["net"] for r in rows])
    suite.within(RATINGS, "The average Division I team's net rating is near zero", m, -4, 4, ".2f", "mean")
    s = sd([r["net"] for r in rows])
    # The spread was 15.0 points per 100 possessions on Nov 25 2026 (about six games a team, on top of last season's
    # ratings). Under 4 would mean the ratings were flattened toward league average; over 25, that they were not shrunk.
    suite.within(RATINGS, "Net ratings spread like team strength does (4 to 25 points per 100 possessions)", s, 4, 25, ".1f", "sd")
    far = [r["team"] for r in rows if abs(r["net"]) > 80 or abs(r["off"]) > 60 or abs(r["def"]) > 60]
    suite.check(RATINGS, "No team is implausibly far from average", not far, f"e.g. {far[0]}" if far else f"largest net {max(abs(r['net']) for r in rows):.1f}")
    off = [r["team"] for r in rows if not 55 <= r["tempo"] <= 82]
    suite.check(RATINGS, "Every team's tempo is between 55 and 82 possessions", not off, f"e.g. {off[0]}" if off else f"league {meta['league_tempo']}")
    short = [r["team"] for r in rows if r["w"] + r["l"] < r["games"] or min(r["w"], r["l"], r["games"]) < 0]
    suite.check(RATINGS, "No team has more rated games than games played", not short, f"e.g. {short[0]}" if short else "")
    games = suite.load(f"{DATA}/team_games.json")["teams"]
    slugs = [r.get("slug") for r in rows]
    suite.check(TEAMS, "Every team has its own page address", all(slugs) and len(set(slugs)) == len(slugs), f"{len(set(slugs))} addresses")
    bad = [r["team"] for r in rows
           if (r["w"], r["l"]) != (sum(g[4] == "final" and g[5] > g[6] for g in games.get(str(r["id"]), [])),
                                   sum(g[4] == "final" and g[5] < g[6] for g in games.get(str(r["id"]), [])))]
    suite.check(TEAMS, "Every team's record is the results listed on its page", not bad, f"e.g. {bad[0]}" if bad else f"{len(rows)} teams")
    odd = [r["team"] for r in rows for g in games.get(str(r["id"]), [])
           if g[4] == "final" and not (10 <= min(g[5], g[6]) and max(g[5], g[6]) <= 180)]      # seen through Nov 24 2026: 17 to 143
    suite.check(TEAMS, "Every final score is a basketball score (10 to 180 points a side)", not odd, f"e.g. {odd[0]}" if odd else "")
    ids = {r["id"] for r in rows}
    lost = [r["team"] for r in rows for g in games.get(str(r["id"]), []) if g[1] is not None and g[1] not in ids]
    suite.check(TEAMS, "Every linked opponent is a team with a page", not lost, f"e.g. {lost[0]}" if lost else "")
    suite.within(RATINGS, "Home court is worth between 1 and 6 points", meta["home_court_points"], 1, 6, ".1f", "points")
    suite.check(RATINGS, "The model's version and state are published", meta.get("model", {}).get("state") in ("validated", "experimental")
                and bool(meta.get("model", {}).get("version")), str(meta.get("model", {}).get("state")))
