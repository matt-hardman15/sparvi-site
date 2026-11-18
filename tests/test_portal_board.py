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
    seen = 0
    for f in d.glob("board_*.json"):          # a coming window has no entrants yet; the check must not be vacuous overall
        for pl in json.loads(f.read_text(encoding="utf-8"))["players"]:
            assert not ({"stars", "rating", "hs_stars"} & set(pl)), f.name
            seen += 1
    assert seen > 0


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


# ---- best landing spots by relative fit (Matt, Nov 17) -------------------------------------------------------------
def _fit_board():
    import pandas as pd
    # Two entrants, four programs. "Easy" scores high for everyone (a program-wide effect); the entrants differ in where
    # they fit better than the other does.
    rows = []
    p = {("A", "Easy"): .9, ("A", "X"): .6, ("A", "Y"): .3, ("A", "Closed"): .8,
         ("B", "Easy"): .9, ("B", "X"): .3, ("B", "Y"): .6, ("B", "Closed"): .8}
    for (pl, tm), v in p.items():
        rows.append({"player": pl, "team": tm, "group": "WR", "p": v})
    return pd.DataFrame(rows)


def test_best_spots_rank_by_relative_fit_not_program_wide_scores():
    import pandas as pd
    b = _fit_board()
    avail = pd.Series({("Easy", "WR"): .5, ("X", "WR"): .5, ("Y", "WR"): .5, ("Closed", "WR"): .01})
    intake = pd.Series({"Easy": 10, "X": 10, "Y": 10, "Closed": 10})
    best = B.best_spots(b, avail, intake, {("X", "A"): 812}, n=1)
    assert best["A"] == [["X", 812]] and best["B"] == [["Y", None]]       # each his own program, not "Easy"
    assert "Closed" not in {s[0] for v in B.best_spots(b, avail, intake, {}).values() for s in v}   # below-median opening


def test_best_spots_skip_programs_that_dont_take_transfers():
    import pandas as pd
    b = _fit_board()
    avail = pd.Series({(t, "WR"): .5 for t in ("Easy", "X", "Y", "Closed")})
    best = B.best_spots(b, avail, pd.Series({"Easy": 10, "Y": 10, "Closed": 10, "X": 0}), {})
    assert "X" not in {s[0] for s in best["A"]}


def test_ledger_freeze_waits_for_entries_but_the_site_shows_the_coming_window(monkeypatch):
    import pandas as pd

    from cfb.portal import export as X
    from sparvi.config import cfb
    monkeypatch.setattr(X.M, "portal", lambda *a, **k: pd.DataFrame())     # no entries yet
    nxt = cfb().season + 1
    day = pd.Timestamp("2026-11-17")
    assert nxt in dict(X.windows(day)) and nxt not in dict(X.windows(day, require_entries=True))


def test_freeze_refreshes_the_board_but_records_nothing_before_entries(monkeypatch):
    from cfb.portal import export as X
    calls = {"export": 0, "ledger": 0}
    monkeypatch.setattr(X, "windows", lambda day, require_entries=False: [] if require_entries else [(2027, day)])
    monkeypatch.setattr(X, "export", lambda t, d: calls.__setitem__("export", calls["export"] + 1) or {"window": t, "result": {}})
    monkeypatch.setattr(X, "write_index", lambda done: None)
    monkeypatch.setattr(F.ledger, "append_forecasts", lambda *a, **k: calls.__setitem__("ledger", calls["ledger"] + 1))
    monkeypatch.setattr(F, "cfb", lambda: SimpleNamespace(season=2026))
    assert F.main() == 0 and calls == {"export": 1, "ledger": 0}


def test_leave_rates_fall_back_when_a_cell_is_thin():
    from cfb.fit import model as M
    d = pd.DataFrame({"group": ["QB"] * 10 + ["WR"] * 100, "exp": [4] * 110, "gone": [True] * 10 + [False] * 100})
    r = M.leave_rates_by_group(d)
    assert r[("WR", 4)] == 0.0                         # 100 rows: its own rate
    assert r[("QB", 4)] == r[("*", 4)] == 10 / 110     # 10 rows: the pooled rate
    assert M.UNDERCLASS == 3                           # underclassmen not in the portal count as returning (Matt, Nov 17)
