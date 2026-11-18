"""The fourth-down decision engine (nfl/fourth/decide.py) on stand-in models with known answers: possession flips,
the end of the game and of the half, which choices are offered, and that each choice is an expectation over its
outcomes rather than one plugged-in result."""
import numpy as np
import pandas as pd
import pytest

from nfl.fourth import decide as D
from nfl.fourth import models as M

N = len(D.QUANTILES)


def _wp(_model, st):
    """A stand-in win probability: the score margin, plus a little for field position."""
    return 1 / (1 + np.exp(-(st["score_differential"].to_numpy(float) / 7 + (50 - st["yardline_100"].to_numpy(float)) / 200)))


@pytest.fixture()
def eng(monkeypatch):
    monkeypatch.setattr(M, "predict_wp", _wp)
    monkeypatch.setattr(M, "predict_fg_baseline", lambda m, d: np.full(len(d), m))
    monkeypatch.setattr(M, "predict_punt", lambda m, d: np.full(len(d), m))
    return D.Engine(wp=None, conv=pd.Series({k: 0.5 for k in range(1, 12)}), fg=0.8, punt=40.0,
                    extra=np.zeros((len(D.YTG_BUCKETS), N)), short=np.zeros((len(D.YTG_BUCKETS), N)),
                    punt_resid=np.zeros((len(D.FP_BUCKETS), N)), p_block=0.0, fg_snap=18.0,
                    runoff={"go": 6.0, "fg": 4.0, "punt": 9.0}, kick_start=70.0, seasons=(2024,))


def sit(**kw) -> pd.DataFrame:
    base = dict(season=2025, score_differential=0.0, game_seconds_remaining=2700.0, half_seconds_remaining=900.0, half2=0.0,
                ot=0.0, down=4.0, ydstogo=2.0, yardline_100=40.0, posteam_timeouts_remaining=3.0,
                defteam_timeouts_remaining=2.0, home=1.0, recv_2h=0.0, strength=1.5, indoors=0.0, wind=5.0)
    return pd.DataFrame([base | kw])


def test_change_of_possession_flips_the_state(eng):
    s = sit(score_differential=3.0)
    # Opponent's ball at its own 30 (70 from the goal), trailing by 3: the offense gets one minus its chance.
    got = eng._after(s, 70, 0, [False], 0)[0]
    opp = _wp(None, pd.DataFrame({"score_differential": [-3.0], "yardline_100": [70.0]}))[0]
    assert got == pytest.approx(1 - opp)
    # Keeping the ball changes nothing but the spot.
    assert eng._after(s, 38, 0, [True], 0)[0] == pytest.approx(_wp(None, pd.DataFrame({"score_differential": [3.0], "yardline_100": [38.0]}))[0])


def test_timeouts_home_and_strength_swap_with_the_ball(eng, monkeypatch):
    seen = {}
    monkeypatch.setattr(M, "predict_wp", lambda m, st: seen.update(st=st.copy()) or np.full(len(st), 0.5))
    eng._after(sit(), 70, 3, [False], 4)
    st = seen["st"].iloc[0]
    assert (st["posteam_timeouts_remaining"], st["defteam_timeouts_remaining"]) == (2.0, 3.0)
    assert (st["home"], st["recv_2h"], st["strength"], st["score_differential"]) == (0.0, 1.0, -1.5, -3.0)
    assert (st["down"], st["ydstogo"], st["game_seconds_remaining"]) == (1.0, 10.0, 2696.0)


def test_a_kick_on_the_last_play_is_worth_its_make_probability(eng):
    s = sit(score_differential=-2.0, half2=1.0, game_seconds_remaining=3.0, half_seconds_remaining=3.0, yardline_100=25.0)
    wp, p = eng.kick(s)
    assert p[0] == 0.8 and wp[0] == pytest.approx(0.8)         # good: wins by 1; missed: loses
    tied = sit(score_differential=-3.0, half2=1.0, game_seconds_remaining=3.0, half_seconds_remaining=3.0, yardline_100=25.0)
    assert eng.kick(tied)[0][0] == pytest.approx(0.8 * 0.5)    # good: overtime counts one half


def test_end_of_the_first_half_goes_to_the_second_half_kickoff(eng):
    base = dict(score_differential=4.0, half_seconds_remaining=0.0, game_seconds_remaining=1800.0, yardline_100=12.0)
    a = eng.value(sit(recv_2h=1.0, **base))[0]      # the team with the ball receives: its ball at the kickoff start
    b = eng.value(sit(recv_2h=0.0, **base))[0]      # the other team receives
    kick = lambda sd: _wp(None, pd.DataFrame({"score_differential": [sd], "yardline_100": [70.0]}))[0]  # noqa: E731
    assert a == pytest.approx(kick(4.0)) and b == pytest.approx(1 - kick(-4.0))


def test_choices_offered_by_field_position(eng):
    r = eng.decide(pd.concat([sit(yardline_100=20.0), sit(yardline_100=45.0), sit(yardline_100=80.0)]))
    assert np.isnan(r["wp_punt"][0]) and not np.isnan(r["wp_fg"][0])           # too close to punt
    assert not np.isnan(r["wp_punt"][1]) and not np.isnan(r["wp_fg"][1])
    assert np.isnan(r["wp_fg"][2]) and not np.isnan(r["wp_punt"][2])           # too far to kick
    assert list(r["best_kick"]) == ["fg", r["best_kick"][1], "punt"]
    assert r["margin"].notna().all() and set(r["best"]) <= {"go", "fg", "punt"}
    assert np.allclose(r["margin"], r["wp_go"] - r["wp_kick"])


def test_going_for_it_is_an_expectation_over_yards_gained(eng):
    s = sit(yardline_100=40.0, ydstogo=2.0)
    flat = eng.go(s)[0][0]
    # Converted plays that gain 0 extra half the time and 20 extra the other half: the value is the average of the
    # two, which is not the value at the average gain (the stand-in is curved in field position).
    b = D._bucket([2.0], D.YTG_BUCKETS)[0]
    eng.extra[b, : N // 2], eng.extra[b, N // 2:] = 0.0, 20.0
    spread = eng.go(s)[0][0]
    v = lambda y: _wp(None, pd.DataFrame({"score_differential": [0.0], "yardline_100": [y]}))[0]  # noqa: E731
    fail = 1 - v(60.0)
    assert flat == pytest.approx(0.5 * v(38.0) + 0.5 * fail)
    assert spread == pytest.approx(0.5 * (0.5 * v(38.0) + 0.5 * v(18.0)) + 0.5 * fail)


def test_a_gain_past_the_goal_line_is_a_touchdown(eng):
    s = sit(yardline_100=3.0, ydstogo=3.0)
    v = _wp(None, pd.DataFrame({"score_differential": [-7.0], "yardline_100": [70.0]}))[0]      # opponent after the kickoff
    fail = 1 - _wp(None, pd.DataFrame({"score_differential": [0.0], "yardline_100": [97.0]}))[0]
    assert eng.go(s)[0][0] == pytest.approx(0.5 * (1 - v) + 0.5 * fail)


def test_punts_include_touchbacks_and_blocks(eng):
    s = sit(yardline_100=35.0)                      # a 40-yard net would cross the goal line: touchback, net 15
    wp, net = eng.punt_wp(s)
    assert net[0] == pytest.approx(35.0 - D.TOUCHBACK)
    eng.p_block = 1.0                               # always blocked: the opponent's ball 10 yards behind the line
    v = _wp(None, pd.DataFrame({"score_differential": [0.0], "yardline_100": [100 - 45.0]}))[0]
    assert eng.punt_wp(s)[0][0] == pytest.approx(1 - v)
