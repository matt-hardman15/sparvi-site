"""Incremental Statcast pulls (scripts/pull_data.py --incremental), with Savant faked out."""
import importlib.util
from datetime import date, timedelta

import pandas as pd
import pytest

from sparvi.config import ROOT

spec = importlib.util.spec_from_file_location("pull_data", ROOT / "scripts" / "pull_data.py")
pdm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pdm)

SEASON = 2026
WINDOW = ("2026-09-01", "2026-09-27")


def games(days, velo=95.0):
    """Two pitches per day, one game a day."""
    rows = []
    for d in days:
        for pn in (1, 2):
            rows.append({"game_pk": int(d.strftime("%m%d")), "game_date": d.isoformat(), "at_bat_number": 1,
                         "pitch_number": pn, "pitcher": 1, "release_speed": velo})
    return pd.DataFrame(rows)


def days(a, b):
    return [date(2026, 9, a) + timedelta(n) for n in range(b - a + 1)]


@pytest.fixture
def raw(tmp_path, monkeypatch):
    monkeypatch.setattr(pdm, "RAW", tmp_path)
    monkeypatch.setitem(pdm.SEASON_WINDOWS, SEASON, WINDOW)
    monkeypatch.setattr(pdm.time, "sleep", lambda s: None)
    return tmp_path


def fake_savant(monkeypatch, frame_for):
    calls = []

    def fetch_range(level, start, end):
        calls.append((start, end))
        return frame_for(start, end)
    monkeypatch.setattr(pdm, "fetch_range", fetch_range)
    return calls


def test_appends_new_games_and_replaces_the_overlap(raw, monkeypatch):
    pdm.tidy(games(days(1, 10))).to_parquet(raw / "statcast_mlb_2026.parquet", index=False)
    # Savant now has games through the 12th, and revised the 8th.
    calls = fake_savant(monkeypatch, lambda s, e: pd.concat([games(days(7, 7)), games(days(8, 8), velo=96.5),
                                                              games(days(9, 12))]))
    pdm.update_season(SEASON, "MLB", today=date(2026, 9, 12))
    assert calls == [(date(2026, 9, 7), date(2026, 9, 12))]  # last game (10th) minus 3 days, to today
    out = pd.read_parquet(raw / "statcast_mlb_2026.parquet")
    assert sorted(out["game_date"].unique()) == [d.isoformat() for d in days(1, 12)]
    assert len(out) == 24 and not out.duplicated(pdm.KEY).any()
    assert set(out.loc[out["game_date"] == "2026-09-08", "release_speed"]) == {96.5}  # the revision won


def test_empty_response_keeps_the_file(raw, monkeypatch):
    path = raw / "statcast_mlb_2026.parquet"
    pdm.tidy(games(days(1, 10))).to_parquet(path, index=False)
    before = pd.read_parquet(path)
    fake_savant(monkeypatch, lambda s, e: pd.DataFrame())
    pdm.update_season(SEASON, "MLB", today=date(2026, 9, 12))
    pd.testing.assert_frame_equal(pd.read_parquet(path), before)


def test_no_file_means_a_full_pull(raw, monkeypatch):
    calls = fake_savant(monkeypatch, lambda s, e: games(days(1, 27)))
    pdm.update_season(SEASON, "MLB", today=date(2026, 9, 12))
    assert calls == [(date(2026, 9, 1), date(2026, 9, 27))]
    assert (raw / "statcast_mlb_2026.parquet").exists()


def test_incremental_matches_a_full_pull(raw, monkeypatch):
    path = raw / "statcast_mlb_2026.parquet"
    pdm.tidy(games(days(1, 20))).to_parquet(path, index=False)
    fake_savant(monkeypatch, lambda s, e: games([d for d in days(1, 27) if s <= d <= e]))
    pdm.update_season(SEASON, "MLB", today=date(2026, 9, 27))
    incremental = pd.read_parquet(path)
    path.unlink()
    pdm.pull_season(SEASON, "MLB")
    pd.testing.assert_frame_equal(incremental, pd.read_parquet(path))
