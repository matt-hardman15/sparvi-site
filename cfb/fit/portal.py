"""Transfer-portal entries linked to CFBD athlete ids (SYSTEM-FIT F0).

CFBD's portal records carry a name, position, origin, destination, date, and rating, but no athlete id. An entry for
season t (the season the player transfers into) is linked to a roster player, in order:
  1. same normalized full name on the origin team's season t-1 roster (unique)
  2. same full name on the origin team's season t-2 roster (players who sat a season out)
  3. same full name, unique among every roster of season t-1 (origin spelled differently, e.g. "Miami (OH)")
  4. same normalized last name and first initial on the origin team's season t-1 roster (unique)
Unlinked entries are kept with player_id None. Name normalization is the play-player parser's (accents, suffixes,
punctuation ignored).
"""
from __future__ import annotations

import pandas as pd

from ingest.cfb.play_players import norm


def _index(roster: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    r = roster.assign(id=roster["id"].astype(str), full=(roster["firstName"].fillna("") + " " + roster["lastName"].fillna("")).map(norm),
                      il=roster["firstName"].fillna("").map(lambda s: norm(s)[:1]) + "|" + roster["lastName"].fillna("").map(norm))
    by_team_full = r.groupby(["team", "full"])["id"].agg(lambda s: sorted(set(s)))
    by_full = r.groupby("full")["id"].agg(lambda s: sorted(set(s)))
    by_team_il = r.groupby(["team", "il"])["id"].agg(lambda s: sorted(set(s)))
    return by_team_full, by_full, by_team_il


def link(portal: pd.DataFrame, rosters: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """portal (one season key or many) + rosters by season -> portal with player_id and link_method."""
    out = []
    cache = {}
    for t, g in portal.groupby("season"):
        for s in (t - 1, t - 2):
            if s in rosters and s not in cache:
                cache[s] = _index(rosters[s])
        prev, prev2 = cache.get(t - 1), cache.get(t - 2)
        for r in g.itertuples():
            full = norm(f"{r.firstName or ''} {r.lastName or ''}")
            il = f"{norm(r.firstName or '')[:1]}|{norm(r.lastName or '')}"
            pid, how = None, None
            for method, idx, key in (("origin_t1", prev and prev[0], (r.origin, full)), ("origin_t2", prev2 and prev2[0], (r.origin, full)),
                                     ("any_t1", prev and prev[1], full), ("initial_last", prev and prev[2], (r.origin, il))):
                if idx is None:
                    continue
                c = idx.get(key, [])
                if len(c) == 1:
                    pid, how = c[0], method
                    break
            out.append({**r._asdict(), "player_id": pid, "link_method": how})
    if not out:                       # no entries yet (a window shown before it opens)
        return portal.assign(player_id=pd.Series(dtype=object), link_method=pd.Series(dtype=object))
    return pd.DataFrame(out).drop(columns=["Index"])
