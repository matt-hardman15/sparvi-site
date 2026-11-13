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


def test_zone_shares_agree_between_dbt_and_location_plus():
    # in_zone (dbt) and Location+ (pitch_quality/features.py) must use the same strike zone.
    import sys

    import yaml

    from sparvi.config import ROOT
    sys.path.insert(0, str(ROOT / "baseball" / "pitch_quality"))
    from features import ZONE_BOT, ZONE_TOP
    v = yaml.safe_load((ROOT / "baseball/pd_pipeline/dbt_project.yml").read_text(encoding="utf-8"))["vars"]
    assert (v["zone_top_share"], v["zone_bot_share"]) == (ZONE_TOP, ZONE_BOT)


def test_nfl_seasons_from_config():
    from sparvi.config import nfl
    n = nfl()
    assert n.seasons[0] == n.first_season and n.seasons[-1] == n.season
    assert n.past_seasons == n.seasons[:-1]
