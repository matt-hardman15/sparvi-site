"""CFB player-level data from CFBD (CFB-PLAYERS): the inputs for the player-level opponent adjustment
(PLAYER-ADJ), the public portal board (SC-3a), and the Portal Fit Finder (SC-3, Layer A).

Per season, into data/raw/cfb/players/ (never committed; raw CFBD data is not redistributed):
    roster_{s}         one row per player x team: position, height (in), weight (lb), class year
    stats_{s}          season box-score stats, long: player x team x category x statType -> stat
    usage_{s}          share of the team's plays a player was involved in (overall, pass, rush, by down)
    ppa_{s}            CFBD's EPA ("PPA") per play and total, overall / pass / rush / by down
    wepa_{s}           CFBD's opponent-adjusted player metrics (Tier 1): passing, rushing, kicking (PAAR)
    portal_{s}         transfer portal entries for the season the player transfers *into*; the January
                       2027 window is portal_2027. Changes daily in a window, so it's never cached as final
    recruits_{s}       high-school recruiting class of year s (signed the winter before season s)

Calls: nine per season (portal from 2021, CFBD's first portal year), plus next season's portal and
recruiting class. A finished season is never requested twice (client cache).
"""
from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from ingest.cfb.client import Client
from sparvi.config import RAW, cfb

OUT = RAW / "cfb" / "players"
FIRST_PORTAL = 2021
WEPA_KINDS = ("passing", "rushing", "kicking")


def _now() -> datetime:
    return datetime.now(UTC)


def _season_done(season: int) -> bool:
    return season < cfb().season or _now() > datetime(season + 1, 2, 15, tzinfo=UTC)


def portal_done(season: int) -> bool:
    """Entries for season s come in the windows before it (December/January, and spring until 2025)."""
    return _now() > datetime(season, 6, 1, tzinfo=UTC)


def recruits_done(season: int) -> bool:
    return _now() > datetime(season, 7, 1, tzinfo=UTC)


def seasons(cfg=None) -> dict[str, list[int]]:
    cfg = cfg or cfb()
    nxt = cfg.season + 1
    return {"season": cfg.seasons,
            "portal": [s for s in range(max(FIRST_PORTAL, cfg.first_season), nxt + 1)],
            "recruits": cfg.seasons + [nxt]}


def requests() -> list[tuple[str, str, dict, bool, int]]:
    """(table, endpoint, params, completed, season) for every player request."""
    r = []
    for s in seasons()["season"]:
        done = _season_done(s)
        r += [("roster", "/roster", {"year": s}, done, s),
              ("stats", "/stats/player/season", {"year": s}, done, s),
              ("usage", "/player/usage", {"year": s}, done, s),
              ("ppa", "/ppa/players/season", {"year": s}, done, s)]
        r += [("wepa", f"/wepa/players/{k}", {"year": s}, done, s) for k in WEPA_KINDS]
    r += [("portal", "/player/portal", {"year": s}, portal_done(s), s) for s in seasons()["portal"]]
    r += [("recruits", "/recruiting/players", {"year": s}, recruits_done(s), s) for s in seasons()["recruits"]]
    return r


# --- frames: fixed columns and types (the contract checks them) --------------------------------------

def _num(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def roster_frame(raw: list[dict], season: int) -> pd.DataFrame:
    df = pd.DataFrame(raw).reindex(columns=["id", "firstName", "lastName", "team", "position", "jersey", "year",
                                            "height", "weight", "homeState", "recruitIds"])
    df["recruitIds"] = df["recruitIds"].map(lambda v: ",".join(map(str, v)) if isinstance(v, list) and v else None)
    df.insert(0, "season", season)
    df["id"] = df["id"].astype(str)
    return _num(df, ["jersey", "year", "height", "weight"]).drop_duplicates(["id", "team"])


def stats_frame(raw: list[dict], season: int) -> pd.DataFrame:
    df = pd.DataFrame(raw).reindex(columns=["playerId", "player", "position", "team", "conference", "category", "statType", "stat"])
    df.insert(0, "season", season)
    df["playerId"] = df["playerId"].astype(str)
    return _num(df, ["stat"]).drop_duplicates(["playerId", "team", "category", "statType"])


SPLITS = ("all", "pass", "rush", "firstDown", "secondDown", "thirdDown", "standardDowns", "passingDowns")


def usage_frame(raw: list[dict], season: int) -> pd.DataFrame:
    rows = [{"season": season, "id": str(x["id"]), "name": x.get("name"), "position": x.get("position"), "team": x.get("team"),
             "conference": x.get("conference"),
             **{f"usage_{k}": (x.get("usage") or {}).get("overall" if k == "all" else k) for k in SPLITS}} for x in raw]
    df = pd.DataFrame(rows, columns=["season", "id", "name", "position", "team", "conference", *[f"usage_{k}" for k in SPLITS]])
    return _num(df, [f"usage_{k}" for k in SPLITS]).drop_duplicates(["id", "team"])


def ppa_frame(raw: list[dict], season: int) -> pd.DataFrame:
    cols = [f"ppa_{a}_{k}" for a in ("avg", "total") for k in SPLITS]
    rows = [{"season": season, "id": str(x["id"]), "name": x.get("name"), "position": x.get("position"), "team": x.get("team"),
             "conference": x.get("conference"),
             **{f"ppa_avg_{k}": (x.get("averagePPA") or {}).get(k) for k in SPLITS},
             **{f"ppa_total_{k}": (x.get("totalPPA") or {}).get(k) for k in SPLITS}} for x in raw]
    df = pd.DataFrame(rows, columns=["season", "id", "name", "position", "team", "conference", *cols])
    return _num(df, cols).drop_duplicates(["id", "team"])


def wepa_frame(parts: dict[str, list[dict]], season: int) -> pd.DataFrame:
    rows = [{"season": season, "kind": kind, "athleteId": str(x.get("athleteId")), "athleteName": x.get("athleteName"),
             "position": x.get("position"), "team": x.get("team"), "conference": x.get("conference"),
             "value": x.get("paar") if kind == "kicking" else x.get("wepa"),
             "n": x.get("attempts") if kind == "kicking" else x.get("plays")} for kind, raw in parts.items() for x in raw]
    df = pd.DataFrame(rows, columns=["season", "kind", "athleteId", "athleteName", "position", "team", "conference", "value", "n"])
    return _num(df, ["value", "n"]).drop_duplicates(["kind", "athleteId", "team"])


def portal_frame(raw: list[dict], season: int) -> pd.DataFrame:
    df = pd.DataFrame(raw).reindex(columns=["firstName", "lastName", "position", "origin", "destination", "transferDate",
                                            "rating", "stars", "eligibility"])
    df.insert(0, "season", season)
    df = _num(df, ["rating", "stars"])
    # No player id in the source: a portal entry is a name + origin + entry date. Exact repeats are dropped.
    return df.drop_duplicates(["firstName", "lastName", "origin", "transferDate"])


def recruits_frame(raw: list[dict], season: int) -> pd.DataFrame:
    df = pd.DataFrame(raw).reindex(columns=["id", "athleteId", "recruitType", "year", "ranking", "name", "school", "committedTo",
                                            "position", "height", "weight", "stars", "rating", "stateProvince"])
    df["id"] = df["id"].astype(str)
    df["athleteId"] = df["athleteId"].map(lambda v: None if v is None or pd.isna(v) else str(v))
    df.insert(0, "season", season)
    return _num(df, ["year", "ranking", "height", "weight", "stars", "rating"]).drop_duplicates(["id"])


def _text_when_empty(df: pd.DataFrame) -> pd.DataFrame:
    """An empty response (next season's portal before the window opens) has no values to type its text
    columns by; write them as text so the file still matches the contract."""
    if len(df):
        return df
    num = {"season"} | {c for c in df.columns if c in ("jersey", "year", "height", "weight", "stat", "rating", "stars",
                                                         "ranking", "value", "n") or c.startswith(("usage_", "ppa_"))}
    return df.astype({c: "string" for c in df.columns if c not in num}).astype({"season": "int64"})


SORT = {"roster": ["team", "id"], "stats": ["team", "playerId", "category", "statType"], "usage": ["team", "id"],
        "ppa": ["team", "id"], "wepa": ["kind", "team", "athleteId"],
        "portal": ["transferDate", "lastName", "firstName", "origin"], "recruits": ["id"]}
FRAMES = {"roster": roster_frame, "stats": stats_frame, "usage": usage_frame, "ppa": ppa_frame,
          "portal": portal_frame, "recruits": recruits_frame}


def pull(c: Client) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reqs = requests()
    wepa: dict[int, dict[str, list[dict]]] = {}
    n = {}
    for table, ep, prm, done, s in reqs:
        raw = c.get(ep, prm, done)
        if table == "wepa":
            wepa.setdefault(s, {})[ep.rsplit("/", 1)[1]] = raw or []
            continue
        df = _text_when_empty(FRAMES[table](raw or [], s))
        df.sort_values(SORT[table]).to_parquet(OUT / f"{table}_{s}.parquet", index=False)
        n[table] = n.get(table, 0) + len(df)
    for s, parts in wepa.items():
        wepa_frame(parts, s).sort_values(SORT["wepa"]).to_parquet(OUT / f"wepa_{s}.parquet", index=False)
    print("players: " + ", ".join(f"{k} {v:,}" for k, v in n.items()) + f" (calls so far {c.calls}, left {c.remaining})")
