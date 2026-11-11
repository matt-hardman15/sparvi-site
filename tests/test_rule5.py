"""Rule 5 eligibility logic (baseball/rule5/analysis.py) and the published board."""
import importlib.util
import json
from datetime import date

import pandas as pd
import pytest

from sparvi.config import ROOT

spec = importlib.util.spec_from_file_location("r5", ROOT / "baseball" / "rule5" / "analysis.py")
r5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r5)

MLB = {111, 134, 137}
ENDS = {"2021": "2021-08-31", "2022": "2022-08-23", "2023": "2023-08-22", "2024": "2024-08-20"}


def person(birth, *tx, drafts=()):
    return {"birthDate": birth, "drafts": list(drafts),
            "transactions": [{"date": d, "typeCode": c, "toTeam": {"id": team, "name": f"Team {team}"}} for d, c, team in tx]}


def test_june5_is_the_one_before_signing():
    assert r5.june5_before(date(2022, 7, 29)) == date(2022, 6, 5)
    assert r5.june5_before(date(2022, 1, 15)) == date(2021, 6, 5)
    assert r5.june5_before(date(2022, 6, 5)) == date(2022, 6, 5)


@pytest.mark.parametrize("birth,signed,expected_age,expected_from", [
    ("2004-05-01", "2022-07-29", 18, 2026),   # high-school draftee: 18 -> 5 seasons (2022-26)
    ("2001-01-18", "2022-07-24", 21, 2025),   # college draftee: 21 -> 4 seasons (2022-25)
    ("2005-03-01", "2022-01-15", 16, 2026),   # international signing in January counts 2022
    ("2005-03-01", "2022-09-10", 17, 2027),   # signed after the complex-league season ended: 2023 is season one
    ("2003-06-10", "2022-07-01", 18, 2026),   # turned 19 after June 5: still 18 on the date that counts
])
def test_eligibility_window(birth, signed, expected_age, expected_from):
    out = r5.estimate(person(birth, (signed, "SGN", 134)), ENDS, MLB)
    assert (out["age_at_signing"], out["eligible_from"], out["basis"]) == (expected_age, expected_from, "signing")


def test_signing_with_a_non_mlb_team_is_ignored_and_draft_year_is_the_fallback():
    p = person("2003-03-01", ("2021-02-01", "SFA", 999), drafts=[{"isDrafted": True, "year": "2022", "team": {"name": "X"}}])
    out = r5.estimate(p, ENDS, MLB)
    assert out["basis"] == "draft" and out["signing_date"] == "2022-07-15"


def test_contract_status_uses_the_latest_join_or_exit():
    assert r5.contract_status(person("2000-01-01", ("2022-07-01", "SGN", 134), ("2026-11-04", "DFA", 134))) == "Free agent"
    # Released and re-signed the same day: under contract.
    assert r5.contract_status(person("2000-01-01", ("2026-05-01", "REL", 134), ("2026-05-01", "SFA", 111))) is None
    assert r5.contract_status(person("2000-01-01", ("2022-07-01", "SGN", 134))) is None


# ---- the committed board ------------------------------------------------------------------
OUT = ROOT / "baseball" / "rule5" / "output"
DATA = ROOT / "site" / "data" / "baseball"


def test_no_board_player_is_on_a_40_man_roster():
    el = pd.read_csv(OUT / "eligibility.csv")
    on40 = set(el.loc[el["on_40man"], "player_id"])
    for f in ("rule5_pitchers.json", "rule5_hitters.json"):
        ids = {r["player_id"] for r in json.loads((DATA / f).read_text(encoding="utf-8"))}
        assert not ids & on40, f"{f}: {sorted(ids & on40)[:5]}"


def test_board_holds_only_eligible_statuses():
    for f in ("rule5_pitchers.json", "rule5_hitters.json"):
        assert {r["status"] for r in json.loads((DATA / f).read_text(encoding="utf-8"))} <= {"Eligible", "Likely eligible", "Unknown"}


def test_validation_meets_target():
    res = json.loads((OUT / "results.json").read_text(encoding="utf-8"))["validation"]
    assert res["n"] >= 10 and res["agreement"] >= 0.8
