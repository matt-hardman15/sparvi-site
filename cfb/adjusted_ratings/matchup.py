"""What the college football power rating expects from a game between two teams (CFB-MATCHUP): the arithmetic the
matchup page does in the browser, here in Python so a test can hold the two equal.

The power rating is already in points against an average FBS team on a neutral field, so the expected margin is the
gap in two teams' ratings plus home field. The spread around it is the root mean squared error of the rating's
next-week forecast in its published backtest (cfb/adjusted_ratings/output/metrics.json), which the export puts in
meta.json beside home field. A margin and a win probability only: the rating does not model points scored.

    expected(a, b, site, model) -> {'margin', 'lo80', 'hi80', 'p_a'}

`a`, `b`: {'power': points}; `site`: 'a' (team A at home), 'b' or 'n' (neutral); `model`: meta.json's `power_model`
(`home_points`, `margin_sd`).
"""
from __future__ import annotations

import math

Z80 = 1.2815515655446004


def expected(a: dict, b: dict, site: str, model: dict) -> dict:
    hc = {"a": 1.0, "b": -1.0, "n": 0.0}[site]
    margin = a["power"] - b["power"] + model["home_points"] * hc
    sd = model["margin_sd"]
    return {"margin": margin, "lo80": margin - Z80 * sd, "hi80": margin + Z80 * sd,
            "p_a": 0.5 * (1 + math.erf(margin / (sd * math.sqrt(2))))}
