"""SYSTEM-FIT F0: portal linking rules, leave rates, and role availability on small hand-built inputs."""
import pandas as pd

from cfb.fit import data as D
from cfb.fit.portal import link


def _roster(rows):
    return pd.DataFrame(rows, columns=["id", "firstName", "lastName", "team"])


def test_portal_link_order_and_uniqueness():
    rosters = {
        2023: _roster([(1, "Jalen", "Smith Jr.", "Alpha"), (2, "Jalen", "Smith", "Bravo"), (3, "Chris", "Lee", "Alpha"),
                       (4, "Cam", "Lee", "Alpha"), (5, "Tom", "Ray", "Miami (OH)")]),
        2022: _roster([(6, "Ben", "Old", "Alpha")]),
    }
    portal = pd.DataFrame([
        {"season": 2024, "firstName": "Jalen", "lastName": "Smith", "origin": "Alpha", "destination": "X", "transferDate": "2023-12-10"},
        {"season": 2024, "firstName": "Chris", "lastName": "Lee", "origin": "Alpha", "destination": None, "transferDate": "2023-12-11"},
        {"season": 2024, "firstName": "Ben", "lastName": "Old", "origin": "Alpha", "destination": None, "transferDate": "2023-12-12"},
        {"season": 2024, "firstName": "Tom", "lastName": "Ray", "origin": "Miami", "destination": None, "transferDate": "2023-12-13"},
        {"season": 2024, "firstName": "C.", "lastName": "Lee", "origin": "Alpha", "destination": None, "transferDate": "2023-12-14"},
    ])
    out = link(portal, rosters).set_index("firstName")
    assert out.loc["Jalen", "player_id"] == "1" and out.loc["Jalen", "link_method"] == "origin_t1"   # suffix ignored
    assert out.loc["Chris", "player_id"] == "3"
    assert out.loc["Ben", "link_method"] == "origin_t2"
    assert out.loc["Tom", "link_method"] == "any_t1"
    assert pd.isna(out.loc["C.", "player_id"])            # "C. Lee" matches two players: left unlinked, never guessed


def test_leave_rates_and_availability():
    w = pd.DataFrame({"team": ["A"] * 4, "player_id": ["1", "2", "3", "4"], "share": [0.5, 0.3, 0.1, 0.1], "group": ["WR"] * 4})
    exp = pd.Series({"1": 4, "2": 2, "3": 2, "4": 5})
    played_next = {("A", "2"), ("A", "3")}
    rates = D.leave_rates([(w, set(), exp, played_next)])
    assert rates == {2: 0.0, 4: 1.0, 5: 1.0}
    ev = pd.DataFrame({"opp": ["target"] * 10, "offense": ["A"] * 10, "player_id": ["1"] * 5 + ["2"] * 3 + ["3"] * 2,
                       "play_id": [str(i) for i in range(10)], "game_id": [1] * 10})
    pos = pd.Series({"1": "WR", "2": "WR", "3": "WR"})
    portal = pd.DataFrame({"player_id": ["2"], "transferDate": ["2024-01-05T00:00:00Z"]})
    agg, rows = D.role_availability(ev, pos, portal, exp, pd.Timestamp("2024-02-01"), {2: 0.25, 4: 0.6})
    a = agg.set_index(["team", "group"]).loc[("A", "WR")]
    assert abs(a["lost_portal"] - 0.3) < 1e-9                       # player 2 entered the portal: certain
    assert abs(a["expected_other_exits"] - (0.5 * 0.6 + 0.2 * 0.25)) < 1e-9
    assert abs(a["open_share"] + a["returning"] - 1) < 1e-9
