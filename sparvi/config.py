"""Project config from sparvi.toml, so seasons live in exactly one place.

Python:  from sparvi.config import baseball; baseball().season
Shell:   python -m sparvi.config baseball.season        -> 2026
         python -m sparvi.config baseball.mlb_seasons   -> 2024 2025 2026
         python -m sparvi.config nfl.season             -> 2026
"""
from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Tests point this at a synthetic fixture; real runs use data/raw (never committed).
RAW = Path(os.environ.get("SPARVI_RAW_DIR", ROOT / "data" / "raw"))


@cache
def _load() -> dict:
    with open(ROOT / "sparvi.toml", "rb") as f:
        return tomllib.load(f)


@dataclass(frozen=True)
class Baseball:
    season: int
    n_mlb: int
    n_aaa: int
    season_windows: dict[int, tuple[str, str]]

    @property
    def prev(self) -> int:
        return self.season - 1

    @property
    def mlb_seasons(self) -> list[int]:
        return list(range(self.season - self.n_mlb + 1, self.season + 1))

    @property
    def aaa_seasons(self) -> list[int]:
        return list(range(self.season - self.n_aaa + 1, self.season + 1))

    @property
    def train_seasons(self) -> list[int]:
        """Stuff+ trains on every MLB season except the current one (the holdout)."""
        return self.mlb_seasons[:-1]

    @property
    def past_seasons(self) -> list[int]:
        """Seasons whose raw data no longer changes (safe to cache between runs)."""
        return [s for s in self.mlb_seasons if s != self.season]


@cache
def baseball() -> Baseball:
    c = _load()["baseball"]
    b = Baseball(
        season=int(c["season"]),
        n_mlb=int(c["mlb_seasons"]),
        n_aaa=int(c["aaa_seasons"]),
        season_windows={int(k): tuple(v) for k, v in c["season_windows"].items()},
    )
    missing = [s for s in b.mlb_seasons if s not in b.season_windows]
    if missing:
        raise ValueError(f"sparvi.toml: add a [baseball.season_windows] entry for {missing}")
    return b


@dataclass(frozen=True)
class NFL:
    season: int
    first_season: int

    @property
    def seasons(self) -> list[int]:
        return list(range(self.first_season, self.season + 1))

    @property
    def past_seasons(self) -> list[int]:
        """Seasons whose play-by-play no longer changes (safe to cache between runs)."""
        return [s for s in self.seasons if s != self.season]


@cache
def nfl() -> NFL:
    c = _load()["nfl"]
    n = NFL(season=int(c["season"]), first_season=int(c["first_season"]))
    if n.first_season > n.season:
        raise ValueError("sparvi.toml: [nfl] first_season is after season")
    return n


@dataclass(frozen=True)
class CFB:
    season: int
    first_season: int
    call_reserve: int        # stop CFBD pulls when the monthly quota left falls to this
    approve_over: int        # a backfill needing more calls than this asks first (--yes)

    @property
    def seasons(self) -> list[int]:
        return list(range(self.first_season, self.season + 1))

    @property
    def past_seasons(self) -> list[int]:
        return [s for s in self.seasons if s != self.season]


@cache
def cfb() -> CFB:
    c = _load()["cfb"]
    x = CFB(season=int(c["season"]), first_season=int(c["first_season"]),
            call_reserve=int(c.get("call_reserve", 100)), approve_over=int(c.get("approve_over", 300)))
    if x.first_season > x.season:
        raise ValueError("sparvi.toml: [cfb] first_season is after season")
    return x


def section(*keys: str) -> dict:
    """A raw table from sparvi.toml, e.g. section("baseball", "rule5")."""
    d = _load()
    for k in keys:
        d = d[k]
    return dict(d)


def _cli(key: str) -> str:
    sport, _, attr = key.partition(".")
    obj = {"baseball": baseball, "nfl": nfl, "cfb": cfb}[sport]()
    val = getattr(obj, attr)
    return " ".join(map(str, val)) if isinstance(val, (list, tuple)) else str(val)


if __name__ == "__main__":
    print(_cli(sys.argv[1]))
