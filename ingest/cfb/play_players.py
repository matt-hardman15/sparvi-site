"""Which players were on each play (PLAYER-ADJ): passer, rusher, target, and the defender on sacks and
interceptions, parsed from CFBD's play text and matched to that season's rosters. No API calls: it reads the
cached raw responses (ingest/cfb/client.py cache) for plays and rosters.

    python -m ingest.cfb.play_players            # every season in sparvi.toml [cfb]
    python -m ingest.cfb.play_players --missing  # the current season and any past season without a file (pull.py)

Writes data/raw/cfb/players/play_players_{season}.parquet (never committed; raw-derived): one row per play x role,
with the matched CFBD athlete id (None when no roster match). Roles: passer, rusher, target, and the defender on
sacks (sacked_by), interceptions (int_by), and, where the text names him, pass breakups (pbu_by). The data check requires >= MIN_MATCH of passer/rusher
roles to match in every season.

Play text comes in two families:
  full names    "Brad Kaaya pass complete to Stacy Coley for 12 yds ...", "A run for 5 yds", "A sacked by D for ...",
                "B 29 Yd pass from A (K Kick)", "A pass intercepted D return for 34 yds"
  jersey style  "No Huddle-Shotgun #15 E.Vasko pass complete short right to #5 A.Little for ...",
                "... pass complete short middle to #81 X.Phipps caught at ASU49, for 13 yards ..." (2025: a name
                can end at "caught"; before CFB-RECV-FIX, 42% of 2025 completions lost their receiver)
                "Shotgun #9 J.Thomas rush middle for 4 yards ...", "#15 E.Vasko pass intercepted by #21 Z.Rodriguez-Brown"
Matching, within the team's roster for that season: exact full name; then first initial + last name (unique);
then jersey number + last name (jersey style); then jersey alone if unique. Suffixes (Jr., III) and punctuation
are ignored.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
import unicodedata
from collections import defaultdict

import pandas as pd

from ingest.cfb.client import CACHE
from sparvi.config import RAW, cfb

OUT = RAW / "cfb" / "players"
MIN_MATCH = 0.95
MIN_TARGET = 0.95                 # completions naming a receiver (CFB-RECV-FIX)
PASS_TYPES = {"Pass Reception", "Pass Incompletion", "Passing Touchdown", "Sack", "Interception", "Pass Interception Return",
              "Pass Interception"}
RUSH_TYPES = {"Rush", "Rushing Touchdown"}
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.I)

NAME = r"[A-Z][\w'.\-]*(?:\s+[A-Za-z][\w'.\-]*)*?"          # one to several capitalized-ish tokens
J = r"#(\d+)\s+([A-Z][\w'.\-]*(?:\s+[A-Z][\w'.\-]*)*?)"     # "#15 E.Vasko", "#21 Z.Rodriguez-Brown"
END = r"(?=\s+(?:for|at|return|rush|rushed|pass|sacked|run|thrown|\d+\s+Yd|left|right|middle|up|short|deep|caught)\b|\s*[,(;]|\.\s*(?:$|(?:Catch|Tackled|Gain|End|Fumble|PENALTY|Penalty|TOUCHDOWN|Touchdown|Out|Ball)\b)|$)"
FULL = [  # (pattern, roles in group order)
    # "B.Shapen pass complete. Catch made by A.Evans for 21 yards." (CFB-RECV-FIX)
    (re.compile(rf"^(?P<a>{NAME})\s+pass\s+complete\.\s+Catch\s+made\s+by\s+(?P<b>{NAME}){END}"), ("passer", "target")),
    # "Z. Tanner pass to R. Ransom for 18 yds, for a TD" (CFB-RECV-FIX: no complete/incomplete word)
    (re.compile(rf"^(?P<a>{NAME})\s+pass\s+(?:(?:complete|incomplete)\s+)?to\s+(?P<b>{NAME}){END}"), ("passer", "target")),
    (re.compile(rf"^(?P<a>{NAME})\s+pass\s+(?:complete|incomplete)(?:\s+to\s+(?P<b>{NAME}))?{END}"), ("passer", "target")),
    (re.compile(rf"^(?P<b>{NAME})\s+\d+\s+Yd\s+pass\s+from\s+(?P<a>{NAME}){END}"), ("passer", "target")),
    (re.compile(rf"^(?P<a>{NAME})\s+sacked\s+by\s+(?P<b>{NAME}){END}"), ("passer", "sacked_by")),
    (re.compile(rf"^(?P<a>{NAME})\s+sacked{END}"), ("passer", None)),
    (re.compile(rf"^(?P<a>{NAME})\s+pass\s+intercepted\s+(?:by\s+)?(?P<b>{NAME}){END}"), ("passer", "int_by")),
    (re.compile(rf"^(?P<a>{NAME})\s+pass\s+intercepted{END}"), ("passer", None)),
    # "N.Vattiato pass complete to FIU 40. Catch made by J.Marshall at FIU 40."
    (re.compile(rf"^(?P<a>{NAME})\s+pass\s+complete\s+to\s+.+?Catch\s+made\s+by\s+(?P<b>{NAME}){END}"), ("passer", "target")),
    (re.compile(rf"^(?P<a>{NAME})\s+rushed\b"), ("rusher", None)),
    (re.compile(rf"^(?P<a>{NAME})\s+(?:run|rush)\s+"), ("rusher", None)),
    (re.compile(rf"^(?P<a>{NAME})\s+\d+\s+Yd\s+Run\b"), ("rusher", None)),
]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = SUFFIX.sub(" ", s)
    return " ".join(re.sub(r"[^a-z ]+", " ", s).split())


class Roster:
    """One team-season's players, indexed for the matching rules."""

    def __init__(self, rows: list[dict]):
        self.full, self.init_last, self.jersey = defaultdict(set), defaultdict(set), defaultdict(set)
        for r in rows:
            first, last = norm(r.get("firstName") or ""), norm(r.get("lastName") or "")
            if not last:
                continue
            pid = str(r["id"])
            self.full[f"{first} {last}".strip()].add(pid)
            if first:
                self.init_last[(first[0], last)].add(pid)
            if r.get("jersey") is not None:
                self.jersey[int(r["jersey"])].add((pid, last))

    def match(self, name: str, jersey: int | None = None) -> str | None:
        n = norm(name)
        if not n:
            return None
        hit = self.full.get(n)
        if hit and len(hit) == 1:
            return next(iter(hit))
        parts = n.split()
        # "E.Vasko" normalizes to "e vasko"; "A.J. Little" to "a j little": initial = first letter, last = last token
        key = (parts[0][0], parts[-1])
        hit = self.init_last.get(key)
        if hit and len(hit) == 1:
            return next(iter(hit))
        if jersey is not None:
            cands = self.jersey.get(jersey, set())
            by_last = {p for p, last in cands if last == parts[-1] or last.endswith(parts[-1]) or parts[-1].endswith(last)}
            if len(by_last) == 1:
                return next(iter(by_last))
            if len(cands) == 1:
                return next(iter(cands))[0]
        return None


def parse(text: str, play_type: str) -> list[tuple[str, str, int | None]]:
    """[(role, name, jersey)] for the players the text names, in the roles PLAYER-ADJ uses."""
    t = re.sub(r"^\(\d+:\d+\)\s*", "", text or "").strip()
    if "#" in t:                                     # jersey style: formation words come before the first "#"
        t = t[t.index("#"):]
        first = re.match(J + END, t)
        if not first:
            return []
        lead = (int(first.group(1)), first.group(2))
        if play_type in RUSH_TYPES or re.search(r"\b(rush|run)\b", t.split(" for ")[0]):
            return [("rusher", lead[1], lead[0])]
        out = [("passer", lead[1], lead[0])]
        # Each other player by the phrase in front of him (two players can share a number: receiver and defender).
        for role, phrase in (("target", r"\bto"), ("int_by", r"\bintercepted\s+by"), ("sacked_by", r"\bsacked\s+by"),
                             ("pbu_by", r"\bbroken\s+up\s+by")):
            m = re.search(phrase + r"\s+" + J + END, t)
            if m:
                out.append((role, m.group(2), int(m.group(1))))
        return out
    for pat, roles in FULL:
        m = pat.search(t)
        if m:
            out = [(roles[0], m.group("a"), None)]
            if roles[1] and m.groupdict().get("b"):
                out.append((roles[1], m.group("b"), None))
            return out
    return []


def _load(pattern: str) -> list[dict]:
    rows = []
    for f in sorted(CACHE.glob(pattern)):
        rows += json.load(gzip.open(f))
    return rows


def season(s: int) -> tuple[pd.DataFrame, dict]:
    rosters: dict[str, Roster] = {}
    by_team = defaultdict(list)
    for r in _load(f"roster__year-{s}*.json.gz"):
        by_team[r["team"]].append(r)
    rosters = {t: Roster(rows) for t, rows in by_team.items()}
    plays = _load(f"plays__seasonType-*_year-{s}.json.gz")
    out, tried, matched = [], defaultdict(int), defaultdict(int)
    for p in plays:
        pt = p.get("playType")
        if pt not in PASS_TYPES and pt not in RUSH_TYPES:
            continue
        for role, name, jersey in parse(p.get("playText"), pt):
            team = p["defense"] if role in ("sacked_by", "int_by", "pbu_by") else p["offense"]
            ro = rosters.get(team)
            pid = ro.match(name, jersey) if ro else None
            tried[role] += 1
            matched[role] += pid is not None
            out.append({"season": s, "game_id": int(p["gameId"]), "play_id": str(p["id"]), "role": role, "team": team,
                        "player_id": pid, "name": name.strip()})
    n_plays = sum(1 for p in plays if p.get("playType") in PASS_TYPES | RUSH_TYPES)
    n_parsed = len({(r["play_id"]) for r in out if r["role"] in ("passer", "rusher")})
    rates = {"plays": n_plays, "parsed": n_parsed / max(n_plays, 1),
             **{f"match_{r}": matched[r] / tried[r] for r in tried}}
    return pd.DataFrame(out), rates


def missing_or_current() -> list[int]:
    """Seasons to (re)parse: the current one (its plays grow every week) and any past one without a file."""
    cfg = cfb()
    return [s for s in cfg.seasons if s == cfg.season or not (OUT / f"play_players_{s}.parquet").exists()]


def main(seasons: list[int] | None = None) -> dict:
    seasons = seasons or cfb().seasons
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for s in seasons:
        df, rates = season(s)
        if df.empty:
            print(f"{s}: no cached plays")
            continue
        df.to_parquet(OUT / f"play_players_{s}.parquet", index=False)
        report[s] = rates
        print(f"{s}: {rates['plays']:,} pass/rush plays, parsed {rates['parsed']:.1%}; "
              + ", ".join(f"{k[6:]} {v:.1%}" for k, v in rates.items() if k.startswith("match_")))
    return report


if __name__ == "__main__":
    if sys.argv[1:] == ["--missing"]:
        main(missing_or_current())
    else:
        main([int(a) for a in sys.argv[1:]] or None)
