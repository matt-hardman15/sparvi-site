"""What the published ratings expect from a game between two teams (CBB-MATCHUP): the arithmetic the matchup page does
in the browser, here in Python so a test can hold the two equal.

    expected(a, b, site, league) -> {'poss', 'a_pts', 'b_pts', 'margin', 'total', 'lo80', 'hi80', 'p_a'}

`a` and `b` are rows of site/data/cbb/ratings.json (off, def, tempo); `site` is 'a' (team A at home), 'b' or 'n'
(neutral); `league` is meta.json's `matchup` block: league efficiency and tempo, home court per 100 possessions, and the
spread of the margin. It is model.forecast with the published ratings: tempo there is a team's own term, here it is
published as possessions against an average opponent (league tempo plus the term), so the two teams' sum has one league
tempo too many. The second stage (three-point luck, availability, continuity, form) is not in it: those need a game day.
"""
from __future__ import annotations

import math

Z80 = 1.2815515655446004


def expected(a: dict, b: dict, site: str, league: dict) -> dict:
    hc = {"a": 1.0, "b": -1.0, "n": 0.0}[site]
    poss = a["tempo"] + b["tempo"] - league["tempo"]
    ea = league["efficiency"] + a["off"] - b["def"] + league["home_court"] * hc
    eb = league["efficiency"] + b["off"] - a["def"] - league["home_court"] * hc
    pa, pb = poss * ea / 100, poss * eb / 100
    margin, sd = pa - pb, league["margin_sd"]
    return {"poss": poss, "a_pts": pa, "b_pts": pb, "margin": margin, "total": pa + pb, "lo80": margin - Z80 * sd, "hi80": margin + Z80 * sd,
            "p_a": 0.5 * (1 + math.erf(margin / (sd * math.sqrt(2))))}
