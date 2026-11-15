"""SYSTEM-FIT F3: transfer detection (id, new id, portal) and the success label rules."""
import pandas as pd

from cfb.fit import labels as LB


def _ros(rows):
    return pd.DataFrame(rows, columns=["id", "firstName", "lastName", "team", "position"])


def test_transfers_three_sources():
    rosters = {
        2021: _ros([(1, "Ann", "Moved", "A", "QB"), (2, "Bo", "Newid", "A", "QB"), (3, "Cy", "Stays", "A", "WR"),
                    (5, "Dup", "Name", "A", "RB"), (6, "Dup", "Name", "C", "RB")]),
        2022: _ros([(1, "Ann", "Moved", "B", "QB"), (99, "Bo", "Newid", "B", "QB"), (3, "Cy", "Stays", "A", "WR"),
                    (50, "Ed", "Portal", "B", "WR"), (77, "Dup", "Name", "B", "RB")]),
    }
    portal = pd.DataFrame([{"season": 2022, "firstName": "Ed", "lastName": "Portal", "origin": "Z", "destination": "B"},
                           {"season": 2022, "firstName": "Cy", "lastName": "Stays", "origin": "A", "destination": "A"}])  # withdrew
    tr = LB.transfers(rosters, 2022, portal).set_index("player_id")
    assert tr.loc["1", "source"] == "id" and tr.loc["1", "team_from"] == "A"
    assert tr.loc["99", "source"] == "reid" and tr.loc["99", "prev_id"] == "2"       # new id, same person
    assert tr.loc["50", "source"] == "portal" and tr.loc["50", "team_from"] == "Z" and pd.isna(tr.loc["50", "prev_id"])
    assert "3" not in tr.index                                                       # stayed
    assert "77" not in tr.index                                                      # ambiguous name: not guessed


def test_role_and_success_rules():
    rows = []
    # n = his dropbacks; each team has 100 pass plays. raw is the in-games-he-played share: q2's 0.9 must not count.
    for pid, team, n, raw, epa, suc in (("q1", "B", 60, 0.6, 0.3, 0.5), ("q2", "B", 40, 0.9, 0.2, 0.4), ("q3", "C", 90, 0.9, -0.1, 0.3),
                                        ("q4", "D", 80, 0.8, 0.1, 0.45)):
        rows += [{"season": 2022, "player_id": pid, "group": "QB", "team": team, "feature": "qb_dropback_share", "n": n, "raw": raw, "adj": raw},
                 {"season": 2022, "player_id": pid, "group": "QB", "team": team, "feature": "qb_epa", "raw": epa, "adj": epa},
                 {"season": 2022, "player_id": pid, "group": "QB", "team": team, "feature": "qb_success", "raw": suc, "adj": suc}]
    plays = pd.Series({("B", "pass"): 100, ("C", "pass"): 100, ("D", "pass"): 100})
    out = LB.role_and_output(pd.DataFrame(rows), 2022, {"B": "SEC", "C": "SEC", "D": "SEC"}, plays).set_index("player_id")
    assert out.loc["q1", "won_role"] and not out.loc["q2", "won_role"]              # 50% of the season's dropbacks
    assert out.loc["q1", "tier"] == "Power"
    assert out.loc["q1", "above_average"] and not out.loc["q3", "above_average"]   # vs. the starters' average
    tr = pd.DataFrame([{"player_id": "q1", "group": "QB", "team_to": "B"}, {"player_id": "q2", "group": "QB", "team_to": "B"},
                       {"player_id": "q9", "group": "QB", "team_to": "B"}])
    lab = LB.label(tr, out.reset_index()).set_index("player_id")
    assert lab.loc["q1", "success"] and not lab.loc["q2", "success"]
    assert not lab.loc["q9", "played"] and not lab.loc["q9", "success"]             # never played at the new school


def test_follow_stops_when_he_leaves():
    def ros(rows):
        return pd.DataFrame(rows, columns=["id", "team"])
    rosters = {2022: ros([("a", "B"), ("b", "B")]), 2023: ros([("a", "B"), ("b", "C")]), 2024: ros([("a", "B")])}
    def outc(s, won):
        return pd.DataFrame([{"player_id": "a", "group": "QB", "team": "B", "share": .6 if won else .1, "won_role": won,
                              "composite": .5, "tier": "Power", "bar": .2, "above_average": True, "season": s}])
    outcomes = {2022: outc(2022, False), 2023: outc(2023, True), 2024: outc(2024, True)}
    tr = pd.DataFrame([{"player_id": p, "prev_id": None, "group": "QB", "position": "QB", "team_from": "A", "team_to": "B",
                        "source": "id", "firstName": "X", "lastName": p, "season": 2022} for p in ("a", "b")])
    y = LB.follow(LB.label(tr, outcomes[2022]), rosters, outcomes, last=2023)
    a = y[y["player_id"] == "a"].set_index("year")
    assert list(a.index) == [1, 2] and not a.loc[1, "success"] and a.loc[2, "success"]   # 2024 is past `last`
    assert a.loc[2, "lastName"] == "a"
    assert list(y.loc[y["player_id"] == "b", "year"]) == [1]                           # left for C: record ends


def test_id_on_two_rosters_settled_by_plays():
    rosters = {2020: _ros([(1, "Ann", "Two", "A", "TE"), (2, "Bo", "Two", "A", "TE")]),
               2021: _ros([(1, "Ann", "Two", "A", "TE"), (1, "Ann", "Two", "B", "TE"), (2, "Bo", "Two", "A", "TE"), (2, "Bo", "Two", "B", "TE")])}
    tr = LB.transfers(rosters, 2021, played={2021: {"1": "B"}}).set_index("player_id")
    assert tr.loc["1", "team_to"] == "B"                     # played for B: moved
    assert "2" not in tr.index                               # no plays to settle it: not guessed
