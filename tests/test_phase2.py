"""FIT-P2: the Phase 2 pre-registration is locked. Its hypotheses, populations, outcomes, and pass rules cannot change
after merge (PLAN-fit); an amendment goes in review/model-changes.md before the vault opens, never by editing the file.
"""
import hashlib
from collections import Counter

import yaml

from sparvi.config import ROOT

SPEC = ROOT / "cfb" / "fit" / "phase2.yml"
LOCKED_SHA256 = "7378f8710009d2ab946eb2b156aa881c2ad0738277c090b21fb9810b3c8ff1a7"   # set when FIT-P2 was written (Nov 17, 2026); never update it


def test_phase2_spec_is_unchanged():
    digest = hashlib.sha256(SPEC.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert digest == LOCKED_SHA256, "cfb/fit/phase2.yml is pre-registered and locked; record amendments in review/model-changes.md"


def test_phase2_spec_shape():
    s = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    assert s["status"] == "pre-registered" and s["vault"]["season"] == 2026
    hyps = s["hypotheses"]
    assert all(c <= 2 for c in Counter(h["model"].split(" (")[0] for h in hyps).values()), "at most two hypotheses per model"
    for h in hyps:
        assert {"id", "model", "question", "population", "outcome"} <= set(h), h["id"]
    assert "below zero" in s["common"]["passes_if"] and "seed 7" in s["common"]["interval"]
