"""SC-3a portal board: market tier rules, the ledger freeze (ranking records only when a list changes), and that the
public site data never carries recruiting-service ratings."""
import json
from types import SimpleNamespace

import pandas as pd

from cfb.portal import board as B
from cfb.portal import freeze as F
from sparvi import ledger
from sparvi.config import ROOT


def _p(**kw):
    base = dict(group="QB", won_role=False, volume=0, hs_stars=None, level="FCS", above_average=False, vor=0.0,
                composite=None, power_bar=0.3)
    return SimpleNamespace(**{**base, **kw})


def test_market_tier_rules():
    assert B.market_tier(_p(level="Power", won_role=True, above_average=True, volume=300)) == 1
    assert B.market_tier(_p(level="FCS", hs_stars=5, vor=5.0)) == 1                 # high recruit producing
    assert B.market_tier(_p(level="G5", won_role=True, volume=300, composite=0.4)) == 2   # Power-level production
    assert B.market_tier(_p(level="Power", volume=60)) == 2                         # Power rotation (backup minimum 50)
    assert B.market_tier(_p(level="G5", won_role=True, volume=300, composite=0.1)) == 3
    assert B.market_tier(_p(level="FCS", hs_stars=4, vor=-1.0)) == 3                # high recruit, little production
    assert B.market_tier(_p(level="FCS", won_role=True, volume=300)) == 4
    assert B.market_tier(_p(level="Power", volume=10)) == 4                         # thin sample


def _res():
    b = pd.DataFrame({"player": ["a", "b", "c", "a", "b", "c"], "team": ["X"] * 3 + ["Y"] * 3, "group": ["QB"] * 6,
                      "yards": [300, 100, 200] * 2, "eff": [0.1, None, 0.3] * 2, "stars": [3, 4, None] * 2, "rating": [0.8, 0.9, None] * 2})
    progs = [{"team": "X", "fit": {"QB": [["c", "similar"], ["a", None]]}}, {"team": "Y St.", "fit": {"QB": [["a", None]], "RB": []}}]
    return {"window": 2027, "board": b, "programs": progs}


def test_freeze_writes_rankings_only_when_a_list_changes(tmp_path):
    day = pd.Timestamp("2027-01-03")
    recs = F.records(_res(), "1.1.0", [], day)
    assert [r["forecast_id"] for r in recs] == ["cfb-portal-2027-x-qb-20270103", "cfb-portal-2027-y-st-qb-20270103"]
    r = recs[0]
    assert r["detail"]["list"] == ["c", "a"] and r["baselines"]["production"] == ["a", "c", "b"]
    assert r["baselines"]["profile"] == ["c", "a", "b"] and r["baselines"]["transfer_rating"] == ["b", "a", "c"]
    gate = lambda m: {"model": m, "version": "1.1.0", "state": "experimental", "reason": "test"}  # noqa: E731
    assert ledger.append_forecasts("cfb", 2027, recs, root=tmp_path, gate=gate) == 2
    existing, _ = ledger.read("cfb", 2027, tmp_path)
    assert F.records(_res(), "1.1.0", existing, day + pd.Timedelta(days=1)) == []      # nothing changed: nothing written
    res = _res()
    res["programs"][0]["fit"]["QB"] = [["a", None], ["c", "similar"]]
    again = F.records(res, "1.1.0", existing, day + pd.Timedelta(days=1))
    assert [x["entity_id"] for x in again] == ["2027:X:QB"]


def test_site_data_has_no_recruiting_ratings():
    """Matt, Nov 15: 247Sports stars and ratings feed the market tier only; the public files never carry them."""
    d = ROOT / "site" / "data" / "cfb" / "portal"
    for f in d.glob("board_*.json"):
        players = json.loads(f.read_text(encoding="utf-8"))["players"]
        assert players and not ({"stars", "rating", "hs_stars"} & set(players[0]))


def test_freeze_writes_projection_lists_separately(tmp_path):
    """Projected-yards lists (cfb_player_projection) are frozen beside the fit lists, each changing on its own."""
    day = pd.Timestamp("2027-01-03")
    res = _res()
    res["programs"][0]["proj"] = {"QB": [["a", 2100, 900, 3300, 0.6], ["c", 1500, 400, 2700, 0.4]]}
    recs = F.records(res, "1.1.0", [], day, proj_version="1.1.0")
    proj = [r for r in recs if r["model"] == F.PROJ_MODEL]
    assert [r["forecast_id"] for r in proj] == ["cfb-proj-2027-x-qb-20270103"]
    assert proj[0]["detail"]["projections"] == [["a", 2100, 900, 3300], ["c", 1500, 400, 2700]]
    assert proj[0]["target"] == "season_yards" and proj[0]["resolves_after"] == "2028-03-01"
    gate = lambda m: {"model": m, "version": "1.1.0", "state": "experimental", "reason": "test"}  # noqa: E731
    ledger.append_forecasts("cfb", 2027, recs, root=tmp_path, gate=gate)
    existing, _ = ledger.read("cfb", 2027, tmp_path)
    assert F.records(res, "1.1.0", existing, day + pd.Timedelta(days=1), proj_version="1.1.0") == []
    res["programs"][0]["proj"]["QB"].reverse()
    again = F.records(res, "1.1.0", existing, day + pd.Timedelta(days=1), proj_version="1.1.0")
    assert [(r["model"], r["entity_id"]) for r in again] == [(F.PROJ_MODEL, "2027:X:QB:proj")]
