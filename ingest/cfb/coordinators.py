"""Offensive coordinators per FBS team-season (FIT-OC, exploratory) from Wikipedia's team-season infoboxes
("off_coach"). Wikipedia text is CC BY-SA 4.0 (Matt approved the terms, Nov 15 2026): credit "Coordinator names from
Wikipedia (CC BY-SA 4.0)" wherever a result built on it appears. The page is the current revision, written after the
season, so it is not point-in-time; this table is not in the snapshot or the loader until it earns a place in a model.

    python -m ingest.cfb.coordinators            # seasons 2016-2025 -> data/raw/cfb/coordinators.parquet
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

from sparvi.config import RAW

API = "https://en.wikipedia.org/w/api.php"
UA = "sparvi-research/1.0 (https://sparvi.io)"
SEASONS = range(2016, 2026)
OUT = RAW / "cfb" / "coordinators.parquet"
CACHE = RAW / "cfb" / "cache" / "wikipedia"


def _get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    key = CACHE / (re.sub(r"[^A-Za-z0-9]+", "_", urllib.parse.urlencode(sorted(params.items())))[:180] + ".json")
    if key.exists():
        return json.loads(key.read_text(encoding="utf-8"))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))
    CACHE.mkdir(parents=True, exist_ok=True)
    key.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(1.0)
    return data


def _clean(v: str) -> str:
    v = re.sub(r"<ref[^>]*/>|<ref.*?</ref>", "", v, flags=re.S)
    v = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", v)
    v = re.sub(r"\{\{[^{}]*\}\}", "", v)
    v = re.sub(r"\([^)]*\)?|\[[^\]]*\]?", "", v)                 # notes: "(1st season; fired Dec. 3)", "(interim)"
    return re.sub(r"\s+", " ", v).strip(" ,;")


def _field(text: str, name: str) -> list[str]:
    """Values of an infobox field (and its numbered variants), split on line breaks and 'and' / '/' (co-coordinators)."""
    out = []
    for m in re.finditer(rf"^\s*\|\s*{name}\d*\s*=\s*(.*)$", text, flags=re.M):
        for part in re.split(r"<br\s*/?>|\s+and\s+|\s*/\s*|\s*&\s*", m.group(1)):
            c = _clean(part)
            if c and not c.startswith("(") and len(c) < 60:
                out.append(c)
    return out


def _pages(titles: list[str]) -> dict[str, tuple[str, str]]:
    """title asked -> (resolved title, wikitext) for pages that exist."""
    out = {}
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        d = _get({"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main", "redirects": 1,
                  "titles": "|".join(chunk)})
        q = d.get("query", {})
        back = {t: t for t in chunk}
        for k in ("normalized", "redirects"):
            for r in q.get(k, []):
                for a, b in list(back.items()):
                    if b == r["from"]:
                        back[a] = r["to"]
        text = {p["title"]: p["revisions"][0]["slots"]["main"]["content"] for p in q.get("pages", []) if "revisions" in p}
        for a, b in back.items():
            if b in text:
                out[a] = (b, text[b])
    return out


def _search(season: int, school: str) -> str | None:
    d = _get({"action": "query", "list": "search", "srlimit": 5, "srsearch": f'intitle:"{season}" "{school}" football team'})
    for r in d.get("query", {}).get("search", []):
        t = r["title"]
        if t.startswith(f"{season} ") and t.endswith("football team") and school.split()[0] in t:
            return t
    return None


def pull() -> pd.DataFrame:
    rows = []
    for s in SEASONS:
        teams = pd.read_parquet(RAW / "cfb" / f"teams_{s}.parquet", columns=["school", "mascot"])
        want = {f"{s} {r.school} {r.mascot} football team": r.school for r in teams.itertuples()}
        got = _pages(list(want))
        miss = [t for t in want if t not in got]
        alt = {t: _search(s, want[t]) for t in miss}
        got.update({t: v for t, v in _pages([a for a in alt.values() if a]).items()})
        for t, school in want.items():
            page = got.get(t) or got.get(alt.get(t) or "")
            title, text = page if page else (None, "")
            oc = _field(text, "off_coach") if text else []
            hc = _field(text, "head_coach") if text else []
            rows.append({"season": s, "school": school, "page": title, "oc": oc[0] if oc else None, "oc_all": "; ".join(oc) or None,
                         "head_coach_wiki": hc[0] if hc else None})
        found = sum(r["season"] == s and r["oc"] is not None for r in rows)
        print(f"{s}: {len(want)} teams, {sum(r['season'] == s and r['page'] is not None for r in rows)} pages, {found} with an OC", flush=True)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    return df


if __name__ == "__main__":
    pull()
