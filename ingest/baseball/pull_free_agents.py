"""Pull this off-season's free-agent list from the MLB Stats API into data/raw/ (gitignored).

    uv run python ingest/baseball/pull_free_agents.py

Writes data/raw/free_agents_SEASON.parquet: one row per player on MLB's free-agent list for the season
just ended (SEASON = sparvi.toml baseball.season), with the club he left, the club he signed with (if
any), and the dates. MLB keeps updating the list as players sign, so the file is a snapshot as of the
day it's pulled (`pulled_on`). Run it by hand, then `python -m ingest.snapshot baseball`.
"""
from __future__ import annotations

import time
from datetime import date

import pandas as pd
import requests

from sparvi.config import RAW, baseball

API = "https://statsapi.mlb.com/api/v1/people/freeAgents"
CFG = baseball()


def pull(season: int) -> pd.DataFrame:
    for attempt in range(4):
        try:
            r = requests.get(API, params={"season": season}, timeout=60)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"  retry {attempt + 1}: {e}")
            time.sleep(3 * (attempt + 1))
    else:
        raise RuntimeError("free-agent list: request failed")
    rows = []
    for x in r.json().get("freeAgents", []):
        new = x.get("newTeam") or {}
        rows.append({
            "season": season, "player_id": x["player"]["id"], "name": x["player"]["fullName"],
            "position": x["position"]["abbreviation"], "position_type": x["position"]["type"],
            "from_team_id": (x.get("originalTeam") or {}).get("id"), "from_team": (x.get("originalTeam") or {}).get("name"),
            "to_team_id": new.get("id"), "to_team": new.get("name"),
            "notes": x.get("notes"), "date_declared": x.get("dateDeclared"), "date_signed": x.get("dateSigned"),
            "pulled_on": date.today().isoformat(),
        })
    df = pd.DataFrame(rows).astype({"from_team_id": "Int64", "to_team_id": "Int64"})
    return df.sort_values("player_id").reset_index(drop=True)


def main() -> None:
    df = pull(CFG.season)
    out = RAW / f"free_agents_{CFG.season}.parquet"
    df.to_parquet(out, index=False)
    print(f"{out.name}: {len(df)} players ({(df['position_type'] == 'Pitcher').sum()} pitchers, "
          f"{df['to_team'].notna().sum()} with a new club) as of {df['pulled_on'].iloc[0]}")


if __name__ == "__main__":
    main()
