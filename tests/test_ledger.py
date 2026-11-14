"""The forecast ledger (L1a): the writer refuses to run outside Actions, the format holds, voids need a
reason, and the append-only check fails on any edit, deletion, or (in a pull request) any new line."""
import subprocess

import pytest

from sparvi import ledger
from sparvi.ledger import LedgerError, NotActions

GATE = {"m": {"model": "m", "version": "1.2.0", "state": "experimental", "reason": "test"}}


def gate(name):
    return GATE[name]


def fc(i=1, **kw):
    return {"forecast_id": f"cfb-2027-test-{i}", "model": "m", "model_version": "1.2.0", "entity_type": "player",
            "entity_id": str(i), "target": "ppa_avg_all", "horizon": "season-2027", "resolves_after": "2099-01-01",
            "point": 0.2, "lo80": 0.1, "hi80": 0.3, **kw}


def git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A scratch git repo with one committed ledger file of two forecasts."""
    git(tmp_path, "init", "-q")
    for k, v in (("user.email", "t@localhost"), ("user.name", "t"), ("core.autocrlf", "false")):
        git(tmp_path, "config", k, v)
    root = tmp_path / "forecasts"
    ledger.append_forecasts("cfb", 2027, [fc(1), fc(2)], root=root, gate=gate)
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path


def test_writer_refuses_the_real_ledger_outside_actions(monkeypatch):
    for k in ("GITHUB_ACTIONS", "GITHUB_REF", "GITHUB_EVENT_NAME"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(NotActions):
        ledger.append_forecasts("cfb", 2027, [fc()], gate=gate)
    # An Actions run on a pull request branch is refused too.
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_REF", "refs/pull/9/merge")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    with pytest.raises(NotActions):
        ledger.append_resolutions("cfb", 2027, [{"forecast_id": "x", "status": "void", "void_reason": "r"}])
    assert not (ledger.LEDGER / "cfb").exists()


def test_writer_stamps_time_sha_and_gate_state(tmp_path):
    ledger.append_forecasts("cfb", 2027, [fc()], root=tmp_path, gate=gate)
    (rec,), res = ledger.read("cfb", 2027, tmp_path)
    assert res == [] and rec["model_state"] == "experimental" and len(rec["code_sha"]) == 40
    assert rec["created_at"].endswith("Z")
    line = (tmp_path / "cfb" / "2027.jsonl").read_text(encoding="utf-8")
    assert line.startswith('{"forecast_id":"cfb-2027-test-1","created_at":') and line.endswith("}\n")


@pytest.mark.parametrize("bad, msg", [
    ({"created_at": "2020-01-01T00:00:00Z"}, "sets"),              # can't backdate: the writer stamps the time
    ({"model_version": "1.1.0"}, "current version"),
    ({"resolves_after": "2020-01-01"}, "must be after"),            # forecasting something already over
    ({"lo80": None}, "interval"),
    ({"point": 0.5}, "lo80 <= point <= hi80"),
    ({"prob": 0.6}, "not both"),
    ({"forecast_id": "Has Spaces"}, "id must be"),
    ({"rank": 3}, "detail"),
])
def test_writer_rejects_bad_forecasts_and_writes_nothing(tmp_path, bad, msg):
    with pytest.raises(LedgerError, match=msg):
        ledger.append_forecasts("cfb", 2027, [fc(1), fc(2, **bad)], root=tmp_path, gate=gate)
    assert not (tmp_path / "cfb" / "2027.jsonl").exists()


def test_probability_forecasts_and_detail(tmp_path):
    p = fc(lo80=None, hi80=None, point=None, prob=0.62, detail={"template": "qb", "rank": 3})
    assert ledger.append_forecasts("cfb", 2027, [p], root=tmp_path, gate=gate) == 1


def test_forecasts_are_never_replaced_and_resolve_once(tmp_path):
    ledger.append_forecasts("cfb", 2027, [fc()], root=tmp_path, gate=gate)
    with pytest.raises(LedgerError, match="never replaced"):
        ledger.append_forecasts("cfb", 2027, [fc()], root=tmp_path, gate=gate)
    with pytest.raises(LedgerError, match="needs a reason"):
        ledger.append_resolutions("cfb", 2027, [{"forecast_id": fc()["forecast_id"], "status": "void"}], root=tmp_path)
    with pytest.raises(LedgerError, match="scored before resolves_after"):
        ledger.append_resolutions("cfb", 2027, [{"forecast_id": fc()["forecast_id"], "status": "scored", "actual": 0.2}],
                                  root=tmp_path)
    with pytest.raises(LedgerError, match="no such forecast"):
        ledger.append_resolutions("cfb", 2027, [{"forecast_id": "cfb-2027-nope", "status": "void", "void_reason": "r"}],
                                  root=tmp_path)
    void = {"forecast_id": fc()["forecast_id"], "status": "void", "void_reason": "transferred to an FCS school"}
    assert ledger.append_resolutions("cfb", 2027, [void], root=tmp_path) == 1
    with pytest.raises(LedgerError, match="already resolved"):
        ledger.append_resolutions("cfb", 2027, [void], root=tmp_path)
    _, (res,) = ledger.read("cfb", 2027, tmp_path)
    assert res["status"] == "void" and "actual" not in res


def test_appending_passes_the_check(repo):
    ledger.append_forecasts("cfb", 2027, [fc(3)], root=repo / "forecasts", gate=gate)
    ledger.append_resolutions("cfb", 2027, [{"forecast_id": fc(1)["forecast_id"], "status": "void", "void_reason": "r"}],
                              root=repo / "forecasts")
    assert ledger.check("HEAD", repo) == ["forecasts/cfb/2027.jsonl: +1 lines", "forecasts/cfb/2027.resolved.jsonl: +1 lines"]


def test_check_fails_on_an_edited_line(repo):
    f = repo / "forecasts" / "cfb" / "2027.jsonl"
    f.write_bytes(f.read_bytes().replace(b'"point":0.2', b'"point":0.25', 1))
    with pytest.raises(LedgerError, match="line 1 was changed"):
        ledger.check("HEAD", repo)


def test_check_fails_on_a_removed_line_or_file(repo):
    f = repo / "forecasts" / "cfb" / "2027.jsonl"
    f.write_bytes(f.read_bytes().split(b"\n", 1)[1])
    with pytest.raises(LedgerError, match="changed or removed"):
        ledger.check("HEAD", repo)
    f.unlink()
    with pytest.raises(LedgerError, match="deleted"):
        ledger.check("HEAD", repo)


def test_check_fails_on_a_reordered_or_whitespace_change(repo):
    f = repo / "forecasts" / "cfb" / "2027.jsonl"
    a, b, _ = f.read_bytes().split(b"\n")
    f.write_bytes(b + b"\n" + a + b"\n")
    with pytest.raises(LedgerError):
        ledger.check("HEAD", repo)
    f.write_bytes((a + b"\n" + b + b"\n").replace(b"\n", b"\r\n"))
    with pytest.raises(LedgerError):
        ledger.check("HEAD", repo)


def test_check_fails_on_hand_added_invalid_lines(repo):
    f = repo / "forecasts" / "cfb" / "2027.jsonl"
    f.write_bytes(f.read_bytes() + b'{"forecast_id":"cfb-2027-test-9"}\n')
    with pytest.raises(LedgerError, match="required"):
        ledger.check("HEAD", repo)
    f.write_bytes(f.read_bytes().rsplit(b"{", 1)[0] + b"not json\n")
    with pytest.raises(LedgerError, match="not JSON"):
        ledger.check("HEAD", repo)


def test_pull_requests_cant_add_lines(repo):
    assert ledger.check("HEAD", repo, pr=True) == []
    ledger.append_forecasts("cfb", 2027, [fc(3)], root=repo / "forecasts", gate=gate)
    with pytest.raises(LedgerError, match="only by an Actions push"):
        ledger.check("HEAD", repo, pr=True)


def test_real_ledger_is_valid_and_the_smoke_path_works(capsys):
    ledger.validate_files(ledger._disk(ledger.LEDGER))
    ledger.smoke()
    out = capsys.readouterr().out
    assert "wrote 1 forecast and 1 void" in out and "fails as it should" in out
    assert not (ledger.LEDGER / "smoke").exists()


def test_jsonl_is_never_line_ending_converted():
    attrs = subprocess.run(["git", "check-attr", "text", "--", "forecasts/cfb/2027.jsonl"], cwd=ledger.ROOT,
                           capture_output=True, text=True, check=True).stdout
    assert attrs.strip().endswith("unset")
