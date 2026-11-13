"""CollegeFootballData.com (CFBD) API client with a cache and a call budget (PLAN-football §0).

- The key comes from CFBD_API_KEY (environment, or the local .env). It is sent as a bearer token and
  never printed, logged, or written to disk outside .env.
- Every response is cached as JSON under data/raw/cfb/cache/. A request for a *completed* period
  (a past season, or a finished week) is never repeated once cached.
- Every real call is logged to data/raw/cfb/_calls.csv (time, endpoint, parameters, status, calls
  left), and the run stops cleanly when the quota left reaches the reserve in sparvi.toml [cfb].

Terms (effective Aug 12 2026): commercial use allowed; no bulk redistribution of raw data; never
expose the key client-side. Credit "Data provided by CollegeFootballData.com".
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import requests

from sparvi.config import RAW, ROOT, cfb

BASE = "https://api.collegefootballdata.com"
CACHE = RAW / "cfb" / "cache"
LOG = RAW / "cfb" / "_calls.csv"


class BudgetExhausted(RuntimeError):
    pass


def _key() -> str:
    k = os.environ.get("CFBD_API_KEY")
    if not k and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("CFBD_API_KEY="):
                k = line.split("=", 1)[1].strip()
    if not k:
        raise RuntimeError("CFBD_API_KEY is not set (.env locally, an Actions secret in CI)")
    return k


def cache_path(endpoint: str, params: dict) -> Path:
    name = endpoint.strip("/").replace("/", "_") + ("__" + urlencode(sorted(params.items())).replace("&", "_").replace("=", "-") if params else "")
    return CACHE / f"{name}.json.gz"


class Client:
    def __init__(self, reserve: int | None = None):
        self.reserve = cfb().call_reserve if reserve is None else reserve
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {_key()}"
        self.session.headers["Accept"] = "application/json"
        self.remaining: int | None = None
        self.calls = 0

    def info(self) -> dict:
        """Quota for this key (one call)."""
        return self._call("/info", {})

    def cached(self, endpoint: str, params: dict) -> bool:
        return cache_path(endpoint, params).exists()

    def get(self, endpoint: str, params: dict, completed: bool):
        """The response for endpoint+params: from the cache if it's there and the period is over."""
        p = cache_path(endpoint, params)
        if completed and p.exists():
            return json.loads(gzip.decompress(p.read_bytes()))
        data = self._call(endpoint, params)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(gzip.compress(json.dumps(data, separators=(",", ":")).encode("utf-8"), mtime=0))
        return data

    def _call(self, endpoint: str, params: dict):
        if self.remaining is not None and self.remaining <= self.reserve:
            raise BudgetExhausted(f"CFBD calls left this month: {self.remaining} (reserve {self.reserve}); stopping")
        for attempt in range(4):
            r = self.session.get(BASE + endpoint, params=params, timeout=120)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            break
        self.calls += 1
        left = r.headers.get("X-CallLimit-Remaining")
        if left is not None:
            self.remaining = int(left)
        self._log(endpoint, params, r.status_code)
        if r.status_code == 401:
            raise PermissionError(f"CFBD {endpoint}: {r.json().get('message', 'unauthorized')}")
        r.raise_for_status()
        return r.json()

    def _log(self, endpoint: str, params: dict, status: int) -> None:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        new = not LOG.exists()
        with LOG.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time_utc", "endpoint", "params", "status", "calls_left"])
            w.writerow([datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), endpoint,
                        urlencode(sorted(params.items())), status, self.remaining])
