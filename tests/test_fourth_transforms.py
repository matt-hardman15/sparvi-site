"""Where the ball is, and whose win probability is read, after every change of possession in the fourth-down engine
(nfl/fourth/decide.py): a failed conversion, a punt (returned, touchback, from the end zone, blocked) and a missed or
blocked field goal. Written for the Nov. 2026 audit of the chart's "go" calls inside the offense's own 10.

Field position is yards from the goal line a team is attacking. A team that turns the ball over `y` yards from the
goal gives the opponent the ball 100 - y yards from the goal the opponent attacks, and the offense's win probability
is one minus the opponent's from there.

The stand-in win probability records every state it is asked about, so each test checks the state handed to the
model (spot, score, down), not only the number that comes back.
"""
import numpy as np
import pandas as pd
import pytest

from nfl.fourth import decide as D
from nfl.fourth import fg as F
from nfl.fourth import models as M

N = len(D.QUANTILES)
KICK_START, FG_SNAP = 70.0, 18.0


def wp_of(yardline, margin=0.0):
    """Stand-in win probability of the team with the ball: better closer to the goal, better with a lead."""
    return 1 / (1 + np.exp(-(np.asarray(margin, float) / 7 + (50 - np.asarray(yardline, float)) / 60)))


@pytest.fixture()
def seen(monkeypatch):
    """Every state the engine asks the win-probability model about, in order."""
    log = []

    def predict(_model, st):
        log.append(st.copy())
        return wp_of(st["yardline_100"], st["score_differential"])

    monkeypatch.setattr(M, "predict_wp", predict)
    monkeypatch.setattr(F, "predict", lambda m, d: np.full(len(d), m))
    monkeypatch.setattr(M, "predict_punt", lambda m, d: np.full(len(d), m))
    return log


def engine(**kw) -> D.Engine:
    base = dict(wp=None, conv=pd.Series({k: 0.5 for k in range(1, 12)}), fg=0.0, fg_max=45.0, punt=40.0,
                extra=np.zeros((len(D.YTG_BUCKETS), N)), short=np.zeros((len(D.YTG_BUCKETS), N)),
                punt_resid=np.zeros((len(D.FP_BUCKETS), N)), p_block=0.0, fg_snap=FG_SNAP,
                runoff={"go": 6.0, "fg": 4.0, "punt": 9.0}, kick_start=KICK_START, seasons=(2024,))
    return D.Engine(**(base | kw))


def sit(**kw) -> pd.DataFrame:
    base = dict(season=2025, score_differential=0.0, game_seconds_remaining=2400.0, half_seconds_remaining=600.0, half2=0.0,
                ot=0.0, down=4.0, ydstogo=3.0, yardline_100=40.0, posteam_timeouts_remaining=3.0,
                defteam_timeouts_remaining=2.0, home=1.0, recv_2h=0.0, strength=1.5, indoors=0.0, wind=5.0)
    return pd.DataFrame([base | kw])


def spots(log) -> list[tuple[float, float, float]]:
    """(yards from goal, score margin, home) of each distinct state asked about, in first-seen order."""
    out = []
    for st in log:
        for r in st[["yardline_100", "score_differential", "home"]].itertuples(index=False):
            if tuple(r) not in out:
                out.append(tuple(r))
    return out


# ---- failed conversion ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("y", [1, 3, 10, 35, 50, 65, 90, 97, 99])
def test_failed_conversion_gives_the_opponent_the_ball_at_100_minus_y(seen, y):
    eng = engine(conv=pd.Series({k: 0.0 for k in range(1, 12)}))     # never converts: go is the failure branch alone
    s = sit(yardline_100=float(y), ydstogo=1.0, score_differential=3.0)
    go = eng.go(s)[0][0]
    # Stopped for no gain: the opponent, trailing by 3 and on the road, has a 1st down 100 - y from its goal.
    # (The engine also values the touchdown branch on every call, used or not: the opponent down 10 after a kickoff.)
    opp = [x for x in seen if (x["home"] == 0).all() and (x["score_differential"] == -3).all()]
    assert opp and all((x["yardline_100"] == 100 - y).all() and (x["down"] == 1).all() for x in opp)
    assert spots(seen)[-1][0] == 100 - y or (float(100 - y), -3.0, 0.0) in spots(seen)
    assert go == pytest.approx(1 - wp_of(100 - y, -3.0))             # the offense gets one minus the opponent's chance


def test_failure_at_the_own_3_is_the_opponents_ball_at_the_3_not_the_97(seen):
    eng = engine(conv=pd.Series({k: 0.0 for k in range(1, 12)}))
    go = eng.go(sit(yardline_100=97.0, ydstogo=3.0))[0][0]
    assert go == pytest.approx(1 - wp_of(3.0)) and go < 0.35         # a disaster, valued as one
    assert go < 1 - wp_of(97.0) - 0.3                                # and nothing like the mirrored spot


def test_failed_conversion_moves_the_spot_by_the_yards_gained_or_lost(seen):
    b = D._bucket([3.0], D.YTG_BUCKETS)[0]
    short = np.zeros((len(D.YTG_BUCKETS), N))
    short[b, : N // 2], short[b, N // 2:] = -6.0, 2.0                # half sacks for 6, half gains of 2 (short of 3)
    eng = engine(conv=pd.Series({k: 0.0 for k in range(1, 12)}), short=short)
    go = eng.go(sit(yardline_100=40.0, ydstogo=3.0))[0][0]
    # A 6-yard loss from the 40 leaves the ball 46 out: the opponent is 54 from its goal. A 2-yard gain: 62.
    assert go == pytest.approx(0.5 * (1 - wp_of(54.0)) + 0.5 * (1 - wp_of(62.0)))


def test_a_failed_gain_never_reaches_the_line_to_gain(seen):
    short = np.full((len(D.YTG_BUCKETS), N), 9.0)                    # the table says 9 yards; it was 4th and 2
    eng = engine(conv=pd.Series({k: 0.0 for k in range(1, 12)}), short=short)
    assert eng.go(sit(yardline_100=40.0, ydstogo=2.0))[0][0] == pytest.approx(1 - wp_of(61.0))   # capped at 1 yard


def test_a_loss_into_the_own_end_zone_is_the_opponents_ball_at_the_1(seen):
    # Safeties aren't modeled; the stand-in is the worst spot on the field, not a spot off it.
    short = np.full((len(D.YTG_BUCKETS), N), -8.0)
    eng = engine(conv=pd.Series({k: 0.0 for k in range(1, 12)}), short=short)
    assert eng.go(sit(yardline_100=97.0, ydstogo=3.0))[0][0] == pytest.approx(1 - wp_of(1.0))


# ---- converted --------------------------------------------------------------------------------------------------
def test_conversion_keeps_the_ball_past_the_line_to_gain_from_the_offenses_side(seen):
    extra = np.full((len(D.YTG_BUCKETS), N), 4.0)
    eng = engine(conv=pd.Series({k: 1.0 for k in range(1, 12)}), extra=extra)
    go = eng.go(sit(yardline_100=97.0, ydstogo=3.0, score_differential=-4.0))[0][0]
    # 3 to gain plus 4 more from the own 3: 1st and 10 at the own 10 (90 out), still trailing by 4, still at home.
    own = [x for x in seen if (x["home"] == 1).all()]
    assert all((x["yardline_100"] == 90).all() and (x["score_differential"] == -4).all() and (x["ydstogo"] == 10).all()
               for x in own)
    assert go == pytest.approx(wp_of(90.0, -4.0))                    # read directly: no one-minus


def test_a_touchdown_hands_the_opponent_a_kickoff_down_seven_more(seen):
    eng = engine(conv=pd.Series({k: 1.0 for k in range(1, 12)}))
    go = eng.go(sit(yardline_100=2.0, ydstogo=2.0, score_differential=-3.0))[0][0]
    assert go == pytest.approx(1 - wp_of(KICK_START, -4.0))          # was down 3, now up 4: the opponent trails by 4


# ---- punts ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("y", "net", "opp_at"), [
    (60, 40, 80),        # own 40, 40 net: the opponent's own 20
    (70, 45, 75),
    (97, 44, 47),        # from the own 3: the ball is at the punting team's 47, so the opponent is 47 from scoring
    (99, 40, 41),        # out of the end zone
    (99, 25, 26),        # a shank from the end zone: the opponent is already in field-goal range
])
def test_punt_puts_the_opponent_at_100_minus_where_it_ends(seen, y, net, opp_at):
    wp, got_net = engine(punt=float(net)).punt_wp(sit(yardline_100=float(y)))
    assert got_net[0] == pytest.approx(net)
    # The punt itself is one state: the opponent's ball, there. (The blocked-punt state is valued too, at weight 0.)
    blocked = (float(np.clip(100 - (y + D.BLOCK_YARDS), 1, 99)), 0.0, 0.0)
    assert [x for x in spots(seen) if x != blocked] == [(float(opp_at), 0.0, 0.0)]
    assert wp[0] == pytest.approx(1 - wp_of(opp_at))


@pytest.mark.parametrize(("y", "net"), [(35, 40), (45, 45), (50, 60), (40, 41)])
def test_a_punt_into_the_end_zone_is_a_touchback_at_the_opponents_20(seen, y, net):
    wp, got_net = engine(punt=float(net)).punt_wp(sit(yardline_100=float(y)))
    assert got_net[0] == pytest.approx(y - D.TOUCHBACK)
    blocked = (float(100 - (y + D.BLOCK_YARDS)), 0.0, 0.0)
    assert [x for x in spots(seen) if x != blocked] == [(80.0, 0.0, 0.0)]   # 80 yards from the goal the opponent attacks
    assert wp[0] == pytest.approx(1 - wp_of(80.0))


def test_a_punt_downed_at_the_1_is_not_a_touchback(seen):
    wp, net = engine(punt=44.0).punt_wp(sit(yardline_100=45.0))
    assert net[0] == pytest.approx(44.0) and wp[0] == pytest.approx(1 - wp_of(99.0))


def test_a_punt_is_an_expectation_over_where_it_lands(seen):
    b = D._bucket([97.0], D.FP_BUCKETS)[0]
    resid = np.zeros((len(D.FP_BUCKETS), N))
    resid[b, : N // 2], resid[b, N // 2:] = -20.0, 10.0
    wp, net = engine(punt=44.0, punt_resid=resid).punt_wp(sit(yardline_100=97.0))
    assert net[0] == pytest.approx(0.5 * 24 + 0.5 * 54)
    assert wp[0] == pytest.approx(0.5 * (1 - wp_of(27.0)) + 0.5 * (1 - wp_of(57.0)))


@pytest.mark.parametrize(("y", "opp_at"), [(60, 30), (85, 5), (89, 1), (97, 1), (99, 1)])
def test_a_blocked_punt_is_the_opponents_ball_behind_the_line(seen, y, opp_at):
    # BLOCK_YARDS behind the line of scrimmage; from deep in its own end that is the 1 (no safety or score modeled).
    wp, _ = engine(p_block=1.0).punt_wp(sit(yardline_100=float(y)))
    assert wp[0] == pytest.approx(1 - wp_of(opp_at))
    assert (float(opp_at), 0.0, 0.0) in spots(seen)


def test_block_and_punt_are_weighted_by_the_block_rate(seen):
    s = sit(yardline_100=70.0)
    wp = engine(p_block=0.25).punt_wp(s)[0][0]
    assert wp == pytest.approx(0.75 * (1 - wp_of(70.0)) + 0.25 * (1 - wp_of(20.0)))


# ---- field goals ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("y", "opp_at"), [
    (30, 62),            # snapped at the 30, kicked from the 38: the opponent's ball there, 62 from its goal
    (40, 52),
    (12, 80),            # kicked from the 20: the same spot
    (5, 80),             # kicked from inside the 20: the opponent gets it at its 20
    (1, 80),
])
def test_a_missed_or_blocked_field_goal_is_the_opponents_ball_at_the_spot_of_the_kick_or_its_20(seen, y, opp_at):
    wp, p = engine(fg=0.0).kick(sit(yardline_100=float(y), score_differential=2.0))      # never good
    assert p[0] == 0.0 and wp[0] == pytest.approx(1 - wp_of(opp_at, -2.0))
    assert (float(opp_at), -2.0, 0.0) in spots(seen)


def test_a_made_field_goal_is_a_kickoff_to_the_opponent_down_three_more(seen):
    wp, _ = engine(fg=1.0).kick(sit(yardline_100=30.0, score_differential=-1.0))
    assert wp[0] == pytest.approx(1 - wp_of(KICK_START, -2.0))


# ---- the opponent's state is the offense's, mirrored ------------------------------------------------------------
def test_every_possession_change_reads_the_opponents_win_probability(seen, monkeypatch):
    """A win probability that knows which team it is asked about: 0.9 for the original offense (home here), 0.2 for
    the opponent. Every turnover branch must come back as 1 - 0.2."""
    monkeypatch.setattr(M, "predict_wp", lambda m, st: np.where(st["home"] == 1, 0.9, 0.2))
    s = sit(yardline_100=40.0)
    never = pd.Series({k: 0.0 for k in range(1, 12)})
    assert engine(conv=never).go(s)[0][0] == pytest.approx(0.8)
    assert engine().punt_wp(s)[0][0] == pytest.approx(0.8)
    assert engine(p_block=1.0).punt_wp(s)[0][0] == pytest.approx(0.8)
    assert engine(fg=0.0).kick(s)[0][0] == pytest.approx(0.8)
    assert engine(fg=1.0).kick(s)[0][0] == pytest.approx(0.8)        # after the kickoff it is the opponent's ball too
    assert engine(conv=pd.Series({k: 1.0 for k in range(1, 12)})).go(s)[0][0] == pytest.approx(0.9)


def test_the_flags_that_belong_to_a_team_change_sides_with_the_ball(seen):
    engine().punt_wp(sit(recv_2h=1.0, strength=2.5, posteam_timeouts_remaining=3.0, defteam_timeouts_remaining=1.0))
    st = seen[-1].iloc[0]
    assert (st["home"], st["recv_2h"], st["strength"]) == (0.0, 0.0, -2.5)
    assert (st["posteam_timeouts_remaining"], st["defteam_timeouts_remaining"]) == (1.0, 3.0)


# ---- what any sane win probability must then imply --------------------------------------------------------------
YARDS = np.arange(99.0, 44.0, -1.0)      # own 1 to the opponent's 45: a 40-yard net stays out of the end zone


def test_go_gets_better_as_the_ball_moves_toward_the_opponents_goal(seen):
    eng = engine()
    for ytg in (1.0, 3.0, 5.0, 10.0):
        y = np.arange(99.0, ytg, -1.0)
        s = pd.concat([sit(yardline_100=v, ydstogo=ytg) for v in y], ignore_index=True)
        assert (np.diff(eng.go(s)[0]) > 0).all()


def test_punt_gets_better_with_field_position(seen):
    s = pd.concat([sit(yardline_100=v) for v in YARDS], ignore_index=True)
    wp = engine().punt_wp(s)[0]
    assert (np.diff(wp) > 0).all() and wp[-1] > wp[0] + 0.05
    # Closer in, a fixed net crosses the goal line and a touchback (the 20) is worse than a punt downed at the 1:
    # the one place a punt may get worse with better field position.
    assert engine().punt_wp(sit(yardline_100=40.0))[0][0] < engine().punt_wp(sit(yardline_100=41.0))[0][0]


def test_failing_at_the_own_3_is_far_worse_than_punting_from_it(seen):
    s = sit(yardline_100=97.0, ydstogo=3.0)
    fail = engine(conv=pd.Series({k: 0.0 for k in range(1, 12)})).go(s)[0][0]
    punt = engine(punt=44.0).punt_wp(s)[0][0]
    assert fail < punt - 0.10
