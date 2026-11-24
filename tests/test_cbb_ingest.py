"""College basketball ingest (CBB-0a): the pull's windows and caps, casting to the contract, and the snapshot tables.
No network: responses are made up, shaped like the source's."""
from datetime import UTC, datetime

import pandas as pd
import pytest
import yaml

import ingest.cbb.checks as checks
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
    tables = {t.name: t for t in cbb_tables() if t.name.endswith(f"_{s}")}        # past seasons: the history tests below
    current = {k: v for k, v in CONTRACT.items() if k != "portal" or (pull.OUT / f"portal_{s}.parquet").exists()}   # portal: from the spring window
    assert set(tables) == {f"cbb_{k}_{s}" for k in current}, "every contract table is snapshotted, under a cbb_ name"
    for k, spec in current.items():
        t = tables[f"cbb_{k}_{s}"]
        assert list(t.key) == spec["key"] and t.path.name == f"{k}_{s}.parquet"
        assert all(c in spec["columns"] for c in spec["key"])
    from ingest.common.snapshots import MIN_MASS_UPDATE
    assert len(CONTRACT) == 11 and MIN_MASS_UPDATE > 400, "the ratings tables (about 365 teams) change wholesale every night"
    games = pd.DataFrame({"startDate": ["2026-11-14T21:00:00.000Z", "2026-11-28T21:00:00.000Z"], "status": ["final", "scheduled"]})
    at = tables[f"cbb_games_{s}"].available_at(games, pd.Timestamp("2026-11-20T12:00:00", tz="UTC"))
    assert list(at) == [pd.Timestamp("2026-11-15"), pd.Timestamp("2026-11-20")], "a result the day after; a scheduled game when first seen"


def test_cbb_season_is_the_year_it_ends():
    c = cbb()
    assert c.end.startswith(str(c.season)) and c.start.startswith(str(c.season - 1)) and c.start < c.end
    assert c.call_reserve >= 500, "basketball must leave football its share of the quota"


# ---- past seasons (CBB-HIST) ----------------------------------------------------------------------------------------
def test_a_past_season_has_the_same_windows_a_year_earlier():
    cfg = cbb()
    s = cfg.season - 3
    for days in (None, 7):
        now, then = pull.windows(days), pull.windows(days, s)
        assert then[0][0] == pd.Timestamp(cfg.start, tz="UTC") - pd.DateOffset(years=3)
        assert then[-1][1] == pd.Timestamp(cfg.end, tz="UTC") - pd.DateOffset(years=3) + pd.Timedelta(days=1)
        assert all(a[1] == b[0] for a, b in zip(then, then[1:], strict=False)) and abs(len(then) - len(now)) <= 1
    assert pull.windows(7) == pull.windows(7, cfg.season)


def test_past_season_requests_are_final_and_leave_out_the_ratings():
    cfg = cbb()
    s = cfg.history_seasons[0]
    r = pull.requests_for(season=s)
    assert all(done for _, _, _, done in r), "a past season's windows are final: cached for good"
    assert not any(t.startswith("ratings") for t, _, _, _ in r), "the source overwrites ratings: a past season's would be end-of-season values"
    assert {t for t, _, _, _ in r} == {"teams", "rankings", "roster", "portal", "games", "team_games", "lines", "player_games"}
    assert {t for t, _, _, _ in r} == set(checks.HISTORY_TABLES), "every past-season table pulled is checked"
    assert all(prm.get("season", prm.get("year")) == s for _, _, prm, _ in r)
    first = min(pd.Timestamp(prm["startDateRange"]) for _, _, prm, _ in r if "startDateRange" in prm)
    assert first.year == s - 1 and first.month == 11


def test_the_default_pull_asks_only_for_the_current_season():
    """The daily workflow runs the pull with no arguments: it must never ask for a past season."""
    s = cbb().season
    assert all(prm.get("season", prm.get("year")) == s for _, _, prm, _ in pull.requests_for())
    wf = (ROOT / ".github" / "workflows" / "cbb.yml").read_text(encoding="utf-8")
    assert "--history" not in wf


def test_mismatch_names_a_missing_column_and_a_value_of_another_type():
    spec = {"columns": {"id": "BIGINT", "spread": "DOUBLE", "provider": "VARCHAR", "neutral": "BOOLEAN"}, "key": ["id"]}
    good = pd.DataFrame({"id": [1, 2], "spread": [-3.5, None], "provider": ["X", None], "neutral": [True, False], "extra": [1, 2]})
    assert pull.mismatch(good, spec) == []
    bad = pd.DataFrame({"id": [1, 2.5], "spread": ["-3.5", "pk"], "provider": [7, "X"]})
    m = pull.mismatch(bad, spec)
    assert any(x == "no column neutral" for x in m)
    assert any(x.startswith("id is BIGINT") for x in m) and any(x.startswith("spread is DOUBLE") for x in m)
    assert any(x.startswith("provider is VARCHAR") for x in m)


def test_a_past_season_that_does_not_fit_the_contract_is_not_written(monkeypatch, tmp_path):
    s = cbb().history_seasons[-1]

    class Fake:
        calls = 0

        def get(self, ep, prm, done):
            return [{"id": 1, "school": "A"}] if ep == "/teams" else []      # far fewer columns than the contract's
    monkeypatch.setattr(pull, "OUT", tmp_path)
    with pytest.raises(pull.ContractMismatch, match="teams: no column"):
        pull.pull(Fake(), season=s)
    assert not list(tmp_path.glob("*.parquet"))


def test_history_snapshot_tables_only_where_the_file_exists(monkeypatch, tmp_path):
    import ingest.snapshot as snap
    s = cbb().history_seasons[-1]
    (tmp_path / "cbb").mkdir()
    monkeypatch.setattr(snap, "RAW", tmp_path)
    assert snap.cbb_history_tables() == []
    for t in ("games", "lines", "roster"):
        pd.DataFrame({"x": [1]}).to_parquet(tmp_path / "cbb" / f"{t}_{s}.parquet")
    tables = {t.name: t for t in snap.cbb_history_tables()}
    assert set(tables) == {f"cbb_games_{s}", f"cbb_lines_{s}", f"cbb_roster_{s}"}
    now = pd.Timestamp("2026-11-24T12:00:00", tz="UTC")
    games = pd.DataFrame({"startDate": [f"{s}-01-14T21:00:00.000Z"], "status": ["final"]})
    assert list(tables[f"cbb_games_{s}"].available_at(games, now)) == [pd.Timestamp(f"{s}-01-15")], "a result the day after"
    assert list(tables[f"cbb_lines_{s}"].available_at(games, now)) == [pd.Timestamp(f"{s}-01-14T21:00:00")], "a line at tip-off"
    first = pd.Timestamp(cbb().start) - pd.DateOffset(years=cbb().season - s)
    assert list(tables[f"cbb_roster_{s}"].available_at(games, now)) == [first], "a roster from the season's first day (approximate)"
    for k in ("games", "lines", "roster"):
        assert list(tables[f"cbb_{k}_{s}"].key) == CONTRACT[k]["key"]


def test_portal_entries_without_a_destination_still_have_the_contract_columns():
    entry = {"id": 1, "sourceId": "9", "year": 2026, "firstName": "A", "lastName": "B", "position": "G", "stars": None, "rating": None,
             "eligibility": "immediate", "yearsRemaining": None, "origin": {"id": 3, "name": "X", "conference": "C"}, "destination": None}
    df = pull.FRAMES["portal"]([entry, {**entry, "id": 2}])
    assert pull.mismatch(pull._jsonable(df), CONTRACT["portal"]) == []
    out = pull.conform(df, "portal", CONTRACT["portal"])
    assert list(out.columns) == list(CONTRACT["portal"]["columns"]) and out["destination_id"].isna().all() and len(out) == 2


def test_known_gaps_hold_a_past_season_to_what_was_measured_and_leave_the_contract_alone():
    limit = CONTRACT["team_games"]["max_null"]["teamStats_possessions"]
    assert limit == 0.02, "the current season's limit is the contract's"
    assert all(v > limit for gaps in checks.KNOWN_GAPS.values() for v in gaps.values())
    assert set(checks.KNOWN_GAPS) <= set(cbb().history_seasons), "only past seasons have recorded gaps"
