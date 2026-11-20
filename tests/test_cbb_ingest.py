"""College basketball ingest (CBB-0a): the pull's windows and caps, casting to the contract, and the snapshot tables.
No network: responses are made up, shaped like the source's."""
from datetime import UTC, datetime

import pandas as pd
import pytest
import yaml

import ingest.cbb.pull as pull
from ingest.snapshot import cbb_tables
from sparvi.config import ROOT, cbb

CONTRACT = yaml.safe_load((ROOT / "contracts" / "cbb_ingest.yml").read_text(encoding="utf-8"))["tables"]


def test_windows_cover_the_season_once():
    cfg = cbb()
    for days in (None, 7):
        w = pull.windows(days)
        assert w[0][0] == pd.Timestamp(cfg.start, tz="UTC") and w[-1][1] == pd.Timestamp(cfg.end, tz="UTC") + pd.Timedelta(days=1)
        assert all(a[1] == b[0] for a, b in zip(w, w[1:], strict=False)), "no gap and no overlap between windows"
    assert max((b - a).days for a, b in pull.windows(7)) == 7 and max((b - a).days for a, b in pull.windows(None)) <= 31


def test_requests_skip_the_future_and_cache_only_finished_windows():
    cfg = cbb()
    now = datetime.fromisoformat(cfg.start).replace(tzinfo=UTC) + pd.Timedelta(days=45)
    r = pull.requests_for(now)
    games = [(prm, done) for t, _, prm, done in r if t == "games"]
    assert len(games) == 2 and games[0][1] and not games[1][1], "the first month is final, the current one is not"
    assert all(pd.Timestamp(prm["startDateRange"]) <= now for _, _, prm, _ in r if "startDateRange" in prm)
    weekly = [done for t, _, _, done in r if t == "player_games"]
    assert len(weekly) == 7 and weekly[:5] == [True] * 5 and weekly[-1] is False
    assert not any(done for t, _, _, done in r if t.startswith("ratings")), "ratings are overwritten at the source: always pulled"


def test_frames_are_cast_to_the_contract():
    game = {"season": 2027, "seasonType": "regular", "startDate": "2026-11-14T17:00:00.000Z", "homeTeamId": 3, "homeTeam": "A",
            "awayTeamId": 4, "awayTeam": "B", "homeScore": 65, "awayScore": 63}
    line = {"provider": "X", "spread": -3, "overUnder": None, "homeMoneyline": -150, "awayMoneyline": 130, "spreadOpen": -2.5,
            "overUnderOpen": None, "somethingNew": 1}
    raw = [{"gameId": 1, **game, "lines": [line, {**line, "spread": -3.5, "overUnder": 140}]},
           {"gameId": 2, **game, "lines": []}]
    df = pull.conform(pull.FRAMES["lines"](raw), "lines", CONTRACT["lines"])
    assert len(df) == 1 and df["spread"].iloc[0] == -3.5, "one row per game x provider: the later one"
    assert "somethingNew" not in df.columns
    want = CONTRACT["lines"]["columns"]
    assert list(df.columns) == list(want)
    assert {c: pull.TYPES[t] for c, t in want.items()} == {c: str(df[c].dtype) for c in df.columns}
    # A column that is empty today keeps its contract type (an all-missing column has no type of its own).
    g = pull.conform(pd.DataFrame({"id": [9], "homeSeed": [None], "tournament": [None], "homePeriodPoints": [[30, 28]], "status": ["final"]}),
                     "games", CONTRACT["games"])
    assert str(g["homeSeed"].dtype) == "Int64" and str(g["tournament"].dtype) == "string" and g["homePeriodPoints"].iloc[0] == "[30,28]"


def test_a_window_at_the_api_cap_stops_the_pull(monkeypatch, tmp_path):
    class Fake:
        calls = 0

        def get(self, ep, prm, done):
            return [{"id": i, "gameId": i} for i in range(pull.CAPS[ep])] if ep == "/games" else []
    monkeypatch.setattr(pull, "OUT", tmp_path)
    with pytest.raises(RuntimeError, match="cap"):
        pull.pull(Fake())


def test_snapshot_tables_match_the_contract():
    s = cbb().season
    tables = {t.name: t for t in cbb_tables()}
    assert set(tables) == {f"cbb_{k}_{s}" for k in CONTRACT}, "every contract table is snapshotted, under a cbb_ name"
    for k, spec in CONTRACT.items():
        t = tables[f"cbb_{k}_{s}"]
        assert list(t.key) == spec["key"] and t.path.name == f"{k}_{s}.parquet"
        assert all(c in spec["columns"] for c in spec["key"])
    from ingest.common.snapshots import MIN_MASS_UPDATE
    assert len(CONTRACT) == 10 and MIN_MASS_UPDATE > 400, "the ratings tables (about 365 teams) change wholesale every night"
    games = pd.DataFrame({"startDate": ["2026-11-14T21:00:00.000Z", "2026-11-28T21:00:00.000Z"], "status": ["final", "scheduled"]})
    at = tables[f"cbb_games_{s}"].available_at(games, pd.Timestamp("2026-11-20T12:00:00", tz="UTC"))
    assert list(at) == [pd.Timestamp("2026-11-15"), pd.Timestamp("2026-11-20")], "a result the day after; a scheduled game when first seen"


def test_cbb_season_is_the_year_it_ends():
    c = cbb()
    assert c.end.startswith(str(c.season)) and c.start.startswith(str(c.season - 1)) and c.start < c.end
    assert c.call_reserve >= 500, "basketball must leave football its share of the quota"
