"""Play-text parsing (ingest/cfb/play_players.py): the receiver is found in each play-text style CFBD uses."""
import pytest

from ingest.cfb.play_players import parse

CASES = [
    # 2025 jersey style: the receiver's name ends at "caught at" (CFB-RECV-FIX)
    ("No Huddle-Shotgun #10 K.Cooper pass complete short middle to #81 X.Phipps caught at ASU49, for 13 yards to the ASU49",
     "Pass Reception", ("K.Cooper", "X.Phipps")),
    ("(14:57) No Huddle-Shotgun #3 G.James pass complete short right to #11 L.Bond caught at LOU00, for 6 yards to the LOU00 "
     "TOUCHDOWN", "Passing Touchdown", ("G.James", "L.Bond")),
    ("Shotgun #15 E.Vasko pass incomplete short right to #5 A.Little", "Pass Incompletion", ("E.Vasko", "A.Little")),
    ("No Huddle-Shotgun #15 E.Vasko pass complete short right to #5 A.Little for 9 yards", "Pass Reception", ("E.Vasko", "A.Little")),
    ("Brad Kaaya pass complete to Stacy Coley for 12 yds", "Pass Reception", ("Brad Kaaya", "Stacy Coley")),
    ("J.Layne pass complete. Catch made by D.Bankston for 5 yards.", "Pass Reception", ("J.Layne", "D.Bankston")),
    ("Z. Tanner pass to R. Ransom for 18 yds, for a TD (J. Jenkins KICK)", "Passing Touchdown", ("Z. Tanner", "R. Ransom")),
    ("N.Vattiato pass complete to FIU 40. Catch made by J.Marshall at FIU 40.", "Pass Reception", ("N.Vattiato", "J.Marshall")),
    ("Miller Moss pass incomplete", "Pass Incompletion", ("Miller Moss", None)),   # the source names no receiver
]


@pytest.mark.parametrize("text,kind,want", CASES)
def test_passer_and_receiver(text, kind, want):
    roles = {r: n for r, n, _ in parse(text, kind)}
    assert (roles.get("passer"), roles.get("target")) == want


def test_parser_change_reparses_cached_seasons(tmp_path, monkeypatch):
    """A cached past season parsed by an older PARSER_VERSION is parsed again on the next pull (CFB-RECV-FIX)."""
    import json
    from types import SimpleNamespace

    import ingest.cfb.play_players as PP
    monkeypatch.setattr(PP, "OUT", tmp_path)
    monkeypatch.setattr(PP, "VERSIONS", tmp_path / "play_players_versions.json")
    monkeypatch.setattr(PP, "cfb", lambda: SimpleNamespace(seasons=[2024, 2025, 2026], season=2026))
    for s in (2024, 2025):
        (tmp_path / f"play_players_{s}.parquet").write_bytes(b"x")
    PP.VERSIONS.write_text(json.dumps({"2024": PP.PARSER_VERSION, "2025": PP.PARSER_VERSION - 1}))
    assert PP.missing_or_current() == [2025, 2026]


def _roster():
    from ingest.cfb.play_players import Roster
    rows = [{"id": "-1014218", "firstName": "Nate", "lastName": "Stanley", "position": "QB"},
            {"id": "4036149", "firstName": "Nate", "lastName": "Stanley", "position": "QB"},
            {"id": "4243485", "firstName": "Steven", "lastName": "Williams", "position": "LB"},
            {"id": "4273883", "firstName": "Steven", "lastName": "Williams", "position": "WR"},
            {"id": "4870518", "firstName": "Zeon", "lastName": "Chriss-Gremillion", "position": "QB"},
            {"id": "5079620", "firstName": "Michael", "lastName": "Van Buren Jr.", "position": "QB"},
            {"id": "4429501", "firstName": "Ray", "lastName": "Davis", "position": "RB"},
            {"id": "5100357", "firstName": "Ray", "lastName": "Davis", "position": "RB"},
            {"id": "1", "firstName": "Sam", "lastName": "Lee", "position": "WR"},
            {"id": "2", "firstName": "Sam", "lastName": "Lee", "position": "WR"}]
    stats = {"4429501": {"rushing": 12.0}, "5100357": {"rushing": 180.0}, "1": {"receiving": 10.0}, "2": {"receiving": 9.0}}
    return Roster(rows, stats)


def test_duplicate_names_resolve_by_id_role_and_volume():
    """CFB-MATCH-FIX: placeholder ids, two players sharing a name, shortened and multi-word last names."""
    r = _roster()
    assert r.match("Nate Stanley", role="passer") == "4036149"          # the real id over the placeholder
    assert r.match("Steven Williams", role="passer") == "4273883"       # offense over defense
    assert r.match("Zeon Chriss", role="passer") == "4870518"           # the one roster name it starts
    assert r.match("M.Van Buren Jr.", role="passer") == "5079620"       # initial + last word of the last name
    assert r.match("Ray Davis", role="rusher") == "5100357"             # the id with the season's carries
    assert r.match("Sam Lee", role="target") is None                    # too close to call: left unmatched
