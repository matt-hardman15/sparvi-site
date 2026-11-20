"""Play-by-play vs. official result, per game: the classes the ratings' integrity rule uses.

The rule itself (what each class does to the plays) lives in football/dbt/models/intermediate/int_cfb_game_integrity.sql;
this is the same classification in Python, for the ingest checks (which run before dbt) and the tests that hold the two
together. CFBD's games table is the authority for who played and the final score.
"""
from __future__ import annotations

CLASSES = ("match", "swapped", "over", "under")
DECISIONS = ("keep", "repair_teams", "repair_scores", "keep_no_score", "exclude")
SWAP_EVIDENCE_POINTS = 7      # dbt var cfb_pbp_swap_evidence_points (tested equal)

EXPLAIN = {
    "match": "the play-by-play reaches the official score",
    "swapped": "the play-by-play reaches the official score with home and away exchanged",
    "over": "a running score goes above the team's official final: a bad score field",
    "under": "the play-by-play stops short of the official score: plays or scores missing",
}
WHAT = {
    "keep": "kept: teams, yards and EPA don't depend on the score fields; a running score above a team's official "
            "final is ignored for garbage time",
    "repair_teams": "repaired: offense and defense exchanged (the scoring plays fit the official result only that way)",
    "repair_scores": "repaired: the two score fields exchanged (the scoring plays fit the official result as labeled)",
    "keep_no_score": "kept as labeled, running score ignored (the scoring plays can't tell which field is exchanged)",
    "exclude": "excluded from the ratings",
    "pending": "repair at the next build: teams or score fields, whichever the game's scoring plays show is exchanged",
}


def classify(home_points: float, away_points: float, pbp_home: float, pbp_away: float) -> str:
    if (pbp_home, pbp_away) == (home_points, away_points):
        return "match"
    if (pbp_home, pbp_away) == (away_points, home_points) and home_points != away_points:
        return "swapped"
    if pbp_home > home_points or pbp_away > away_points:
        return "over"
    return "under"


def swap_decision(home_points: float, away_points: float, est_home: float, est_away: float) -> str:
    """For a swapped game: which field is exchanged, from points by labeled offense (TDs x 7 + field goals x 3)."""
    d_keep = abs(est_home - home_points) + abs(est_away - away_points)
    d_swap = abs(est_home - away_points) + abs(est_away - home_points)
    if d_keep - d_swap >= SWAP_EVIDENCE_POINTS:
        return "repair_teams"
    if d_swap - d_keep >= SWAP_EVIDENCE_POINTS:
        return "repair_scores"
    return "keep_no_score"
