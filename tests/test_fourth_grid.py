"""The fourth-down chart's standard situations (nfl/fourth/decisions.grid) and the published grid they produce
(site/data/nfl/fourth_downs.json): checks a reader would make by eye. Added with the Nov. 2026 audit, after the chart
called 4th-and-3 through 4th-and-7 from inside the offense's own 10 a clear go.

The published-grid checks run on the committed JSON, so a rebuild that breaks one fails CI before it deploys.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl.fourth import decide as D
from nfl.fourth import decisions as FD

ROOT = Path(__file__).resolve().parents[1]
BINARY = ["home", "recv_2h", "half2", "ot", "indoors"]          # 0/1 inputs of the tree models
GO_SLACK = 0.0          # the published go curve has never stepped down; keep it that way
PUNT_SLACK = 0.0025     # the win-probability model is trees fit to noisy outcomes: steps this small aren't a slope
PUNT_STEADY_FROM = 55   # closer than this, more punts reach the end zone and a touchback can cost field position


# ---- the situations fed to the engine ---------------------------------------------------------------------------
def test_grid_flags_are_zero_or_one():
    """A tree model doesn't average an in-between flag: it puts 0.5 on one side of every split. recv_2h = 0.5, which
    stays 0.5 when the ball changes hands, made whichever team had the ball next the one receiving the 2nd-half
    kickoff, and tilted every cell toward going for it."""
    g = FD.grid(2026)
    for c in BINARY:
        assert set(g[c].unique()) <= {0.0, 1.0}, c


def test_grid_averages_each_cell_over_home_and_the_second_half_kickoff():
    g = FD.grid(2026)
    n = g.groupby(["yardline_100", "ydstogo"]).agg(rows=("home", "size"), home=("home", "mean"), recv=("recv_2h", "mean"),
                                                   combos=("home", lambda h: len(set(zip(h, g.loc[h.index, "recv_2h"], strict=True)))))
    assert (n["rows"] == 4).all() and (n["combos"] == 4).all() and (n["home"] == 0.5).all() and (n["recv"] == 0.5).all()
    assert len(n) == sum(1 for y in FD.GRID_YARDLINES for t in FD.GRID_YTG if t <= y)


def test_grid_is_the_situation_the_page_states():
    """Tie game, 10:00 left in the 2nd quarter (40:00 left in the game), 4th down, all timeouts, even teams, outdoors."""
    g = FD.grid(2026)
    one = {c: g[c].unique().tolist() for c in ["score_differential", "half_seconds_remaining", "game_seconds_remaining",
                                               "down", "posteam_timeouts_remaining", "defteam_timeouts_remaining",
                                               "strength", "indoors", "half2", "ot"]}
    assert one == {"score_differential": [0.0], "half_seconds_remaining": [600.0], "game_seconds_remaining": [2400.0],
                   "down": [4.0], "posteam_timeouts_remaining": [3.0], "defteam_timeouts_remaining": [3.0],
                   "strength": [0.0], "indoors": [0.0], "half2": [0.0], "ot": [0.0]}
    assert set(D.STATE) <= set(g.columns) and (g["ydstogo"] <= g["yardline_100"]).all()
    page = (ROOT / "site" / "nfl" / "fourth-downs" / "index.qmd").read_text(encoding="utf-8")
    assert "tie game, 10:00 left in the second quarter" in page


def test_grid_table_averages_a_cells_rows_within_each_refit():
    g = FD.grid(2026)
    g = g[(g["yardline_100"] == 97) & (g["ydstogo"] == 3)].reset_index(drop=True)
    r = pd.DataFrame({"wp_go": [0.40, 0.44, 0.38, 0.42], "wp_fg": np.nan, "wp_punt": [0.42, 0.44, 0.40, 0.42],
                      "p_convert": 0.5, "p_fg": 0.9, "punt_net": 44.0})
    r["margin"] = r["wp_go"] - r["wp_punt"]
    boot = np.array([[0.01, -0.03, 0.01, -0.03], [0.02, 0.02, 0.02, 0.02], [-0.05, -0.01, -0.05, -0.01]])
    (row,) = FD._grid_table(g, r, boot)
    assert (row["yardline_100"], row["ydstogo"], row["kick"]) == (97, 3, "punt")
    assert row["wp_go"] == pytest.approx(0.41) and row["wp_punt"] == pytest.approx(0.42) and row["margin"] == pytest.approx(-0.01)
    assert row["margin_lo"] < 0 < row["margin_hi"] and row["call"] == "toss-up"       # refit means: -0.01, 0.02, -0.03


# ---- the published grid -----------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def grid() -> pd.DataFrame:
    return pd.DataFrame(json.loads((ROOT / "site" / "data" / "nfl" / "fourth_downs.json").read_text(encoding="utf-8"))["grid"])


def test_published_grid_is_complete(grid):
    assert len(grid) == sum(1 for y in range(1, 100) for t in range(1, 11) if t <= y)
    assert set(grid["call"]) <= {"go", "kick", "toss-up"}
    assert np.allclose(grid["m"], grid["go"] - grid[["fg", "punt"]].max(axis=1), atol=2e-4)
    assert ((grid["lo"] <= grid["hi"]).all() and (grid.loc[grid["call"] == "go", "lo"] >= 0).all()
            and (grid.loc[grid["call"] == "kick", "hi"] <= 0).all())    # rounded to 4 places


def test_no_clear_go_on_fourth_and_five_or_more_inside_the_own_10(grid):
    """Failing there hands the opponent the ball in scoring range; teams have tried it a handful of times in ten
    seasons, almost all in desperation."""
    deep = grid[(grid["y"] >= 90) & (grid["t"] >= 5)]
    assert len(deep) == 60 and not (deep["call"] == "go").any(), deep[deep["call"] == "go"][["y", "t", "m", "lo"]]


def test_go_gets_better_toward_the_opponents_goal(grid):
    """At a fixed distance to go, the chance to win by going rises as the ball moves toward the opponent's goal."""
    for t, s in grid.sort_values("y", ascending=False).groupby("t"):
        go = s["go"].to_numpy()
        assert (np.diff(go) >= -GO_SLACK).all(), (t, s[["y", "go"]].to_numpy()[np.r_[False, np.diff(go) < -GO_SLACK]])
        assert go[-1] > go[0] + 0.10


def test_punt_gets_better_with_field_position(grid):
    """Field position is yards from the goal, so the same holds for every distance to go: one check a column."""
    p = grid[grid["punt"].notna()].groupby("y")["punt"].agg(["min", "max"])
    assert (p["max"] - p["min"] < 1e-9).all()                       # a punt doesn't depend on the distance to go
    wp = p["max"].sort_index(ascending=False)                       # own 1 first
    far = wp[wp.index >= PUNT_STEADY_FROM].to_numpy()
    assert (np.diff(far) >= -PUNT_SLACK).all(), wp[wp.index >= PUNT_STEADY_FROM][np.r_[False, np.diff(far) < -PUNT_SLACK]]
    assert wp.loc[60] > wp.loc[99] + 0.03



# ---- the fitted engine (needs the raw play-by-play: skipped where data/raw/nfl isn't populated, as in check.yml) ----
@pytest.fixture(scope="module")
def fitted():
    from sparvi.config import RAW, nfl
    season = nfl().season
    if not all((RAW / "nfl" / f"pbp_{s}.parquet").exists() and (RAW / "nfl" / "state" / f"pbp_state_{s}.parquet").exists()
               for s in range(FD.FIRST_SEASON, season)):
        pytest.skip("raw NFL play-by-play not on disk")
    from backtest.asof import Loader
    eng, _ = D.engine_for(Loader(), season, FD.FIRST_SEASON, f"{season}-09-01")
    return eng, FD.grid(season)


def test_failing_at_the_own_3_is_far_worse_than_punting_from_it_in_the_fitted_engine(fitted):
    eng, g = fitted
    s = g[(g["yardline_100"] == 97) & (g["ydstogo"] == 3)].reset_index(drop=True)
    no = np.zeros(len(s), bool)
    fail = eng._after(s, 100 - 97.0, 0, no, eng.runoff["go"]).mean()          # the opponent's ball at the 3
    kept = eng._after(s, 94.0, 0, ~no, eng.runoff["go"]).mean()               # 1st and 10 at the own 6
    punt = eng.punt_wp(s)[0].mean()
    assert fail < punt - 0.05, (fail, punt)
    assert fail < 0.40 < kept, (fail, kept)
    assert 38 < eng.punt_wp(s)[1].mean() < 48                                 # net yards from the end zone, 2016-2025: 44
