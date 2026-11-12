"""The published snapshot files agree with the season-level files at the final date.

The exact (full-precision) check runs in dbt on synthetic data (test_pipeline.py). This one runs on
the real exported JSON for 20 random players per table, comparing at the precision each file is
published at.
"""
import json
import random

import pytest

from sparvi.config import ROOT

DATA = ROOT / "site" / "data" / "baseball"
N = 20


def load(name):
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def final(snap: dict, pid, field, window="std"):
    s = next(p for p in snap["players"] if p["id"] == pid)["series"][window]
    return s[field][-1]


def close(a, b, tol):
    return (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= tol + 1e-12)


@pytest.mark.parametrize("level", ["MLB", "AAA"])
def test_pitcher_grades_role_and_counts(level):
    snap, season = load(f"snapshots_grades_{level.lower()}"), [r for r in load("stuff_pitchers") if r["level"] == level]
    for r in random.Random(7).sample(season, N):
        pid = r["id"]
        for g in ["stuff_plus", "loc_plus", "pitching_plus", "rv_plus"]:
            assert close(final(snap, pid, g), r[g], 0), (pid, g)
        assert (final(snap, pid, "role"), final(snap, pid, "games"), final(snap, pid, "starts")) == (r["role"], r["g"], r["gs"]), pid


def test_triple_a_pitcher_rates_and_percentiles():
    snap = load("snapshots_pitchers_aaa")
    # League files carry only what the tools show; check every rate they carry.
    rates = [m for m in ["k_pct", "bb_pct", "csw_pct", "whiff_pct", "chase_pct", "hard_hit_pct", "xwoba"] if m in snap["fields"]]
    assert len(rates) >= 6
    for r in random.Random(11).sample(load("aaa_pitchers"), N):
        pid = r["player_id"]
        assert final(snap, pid, "bf") == r["bf"]
        for m in rates:
            # Season file has 4 decimals, snapshots 3: equal to within the coarser rounding.
            assert close(final(snap, pid, m), r[m], 0.0005), (pid, m)
            assert final(snap, pid, m + "_pctl") == r[m + "_pctl"], (pid, m + "_pctl")
        assert close(final(snap, pid, "fb_velo"), r["fb_velo"], 0.05), (pid, "fb_velo")


def test_triple_a_hitter_rates_and_percentiles():
    snap = load("snapshots_hitters_aaa")
    rates = [m for m in ["chase_pct", "zone_swing_pct", "zone_contact_pct", "whiff_pct", "k_pct", "bb_pct", "hard_hit_pct", "xwoba"]
             if m in snap["fields"]]
    assert len(rates) >= 6
    for r in random.Random(13).sample(load("aaa_hitters"), N):
        pid = r["player_id"]
        assert final(snap, pid, "pa") == r["pa"]
        for m in rates:
            assert close(final(snap, pid, m), r[m], 0.0005), (pid, m)
            assert final(snap, pid, m + "_pctl") == r[m + "_pctl"], (pid, m + "_pctl")
        assert close(final(snap, pid, "ev90"), r["ev90"], 0.05), (pid, "ev90")


@pytest.mark.parametrize("level", ["MLB", "AAA"])
def test_pitch_type_grades(level):
    snap = load(f"snapshots_pitches_{level.lower()}")
    by_key = {(p["id"], p["pt"]): p["series"] for p in snap["pitches"]}
    season = [r for r in load("stuff_pitches") if r["level"] == level]
    for r in random.Random(17).sample(season, N):
        s = by_key[(r["id"], r["pt"])]
        assert all(close(s[g][-1], r[g], 0) for g in ["stuff_plus", "loc_plus", "pitching_plus"]), (r["id"], r["pt"])
        assert s["pitches"][-1] >= r["n"], (r["id"], r["pt"])


@pytest.mark.parametrize("name", ["snapshots_grades_mlb", "snapshots_grades_aaa", "snapshots_pitchers_aaa", "snapshots_hitters_aaa"])
def test_cumulative_counts_never_decrease(name):
    count = "pa" if "hitters" in name else "pitches"
    for p in load(name)["players"]:
        v = [x for x in p["series"]["std"][count] if x is not None]
        assert v == sorted(v), p["id"]


def test_player_files_hold_every_metric_at_the_final_date():
    """Fields left out of the league files live in the player files; spot-check them the same way."""
    for r in random.Random(19).sample(load("aaa_hitters"), N):
        s = json.loads((DATA / "players" / f"{r['player_id']}.json").read_text(encoding="utf-8"))["timeline"]["AAA"]["hitter"]["std"]
        for m in ["zone_swing_pct", "whiff_pct"]:
            assert close(s[m][-1], r[m], 0.0005) and s[m + "_pctl"][-1] == r[m + "_pctl"], (r["player_id"], m)
