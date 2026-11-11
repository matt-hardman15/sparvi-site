from sparvi.config import Baseball, _cli, baseball


def test_seasons_derive_from_one_number():
    b = Baseball(season=2027, n_mlb=3, n_aaa=2, season_windows={})
    assert b.mlb_seasons == [2025, 2026, 2027]
    assert b.aaa_seasons == [2026, 2027]
    assert b.train_seasons == [2025, 2026]
    assert b.past_seasons == [2025, 2026]
    assert b.prev == 2026


def test_real_config_has_windows_for_every_season():
    b = baseball()
    assert all(s in b.season_windows for s in b.mlb_seasons)
    assert _cli("baseball.season") == str(b.season)
    assert _cli("baseball.mlb_seasons") == " ".join(map(str, b.mlb_seasons))
