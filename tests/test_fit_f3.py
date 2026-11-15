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
    portal = pd.DataFrame([{"season": 2022, "firstName": "Ed", "lastName": "Portal", "origin": "Z", "destination": "B"}])
    tr = LB.transfers(rosters, 2022, portal).set_index("player_id")
    assert tr.loc["1", "source"] == "id" and tr.loc["1", "team_from"] == "A"
    assert tr.loc["99", "source"] == "reid" and tr.loc["99", "prev_id"] == "2"       # new id, same person
    assert tr.loc["50", "source"] == "portal" and tr.loc["50", "team_from"] == "Z" and pd.isna(tr.loc["50", "prev_id"])
    assert "3" not in tr.index                                                       # stayed
    assert "77" not in tr.index                                                      # ambiguous name: not guessed


def test_role_and_success_rules():
    rows = []
    for pid, team, share, epa, suc in (("q1", "B", 0.6, 0.3, 0.5), ("q2", "B", 0.4, 0.2, 0.4), ("q3", "C", 0.9, -0.1, 0.3),
                                       ("q4", "D", 0.8, 0.1, 0.45)):
        rows += [{"season": 2022, "player_id": pid, "group": "QB", "team": team, "feature": "qb_dropback_share", "raw": share, "adj": share},
                 {"season": 2022, "player_id": pid, "group": "QB", "team": team, "feature": "qb_epa", "raw": epa, "adj": epa},
                 {"season": 2022, "player_id": pid, "group": "QB", "team": team, "feature": "qb_success", "raw": suc, "adj": suc}]
    out = LB.role_and_output(pd.DataFrame(rows), 2022, {"B": "SEC", "C": "SEC", "D": "SEC"}).set_index("player_id")
    assert out.loc["q1", "won_role"] and not out.loc["q2", "won_role"]              # 50% of dropbacks
    assert out.loc["q1", "tier"] == "Power"
    assert out.loc["q1", "above_average"] and not out.loc["q3", "above_average"]   # vs. the starters' average
    tr = pd.DataFrame([{"player_id": "q1", "group": "QB", "team_to": "B"}, {"player_id": "q2", "group": "QB", "team_to": "B"},
                       {"player_id": "q9", "group": "QB", "team_to": "B"}])
    lab = LB.label(tr, out.reset_index()).set_index("player_id")
    assert lab.loc["q1", "success"] and not lab.loc["q2", "success"]
    assert not lab.loc["q9", "played"] and not lab.loc["q9", "success"]             # never played at the new school
