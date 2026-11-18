"""FBS head coaches by season from CFBD (SYSTEM-FIT F1: a team's system should persist unless its staff changes).

One call per season (`/coaches?year=`), cached like every CFBD response; a finished season is never requested twice.
Writes data/raw/cfb/coaches.parquet: one row per coach x school x season (a mid-season change gives two rows).

    python -m ingest.cfb.coaches            # every season in sparvi.toml [cfb]
"""
from __future__ import annotations

import pandas as pd

from ingest.cfb.client import Client
from sparvi.config import RAW, cfb

OUT = RAW / "cfb" / "coaches.parquet"


def pull(c: Client | None = None) -> pd.DataFrame:
    c = c or Client()
    cfg = cfb()
    rows = []
    for s in cfg.seasons:
        for coach in c.get("/coaches", {"year": s}, completed=s < cfg.season):
            for x in coach.get("seasons", []):
                if x.get("year") != s:
                    continue
                rows.append({"season": s, "coach_id": int(coach["id"]), "coach": f"{coach.get('firstName', '')} {coach.get('lastName', '')}".strip(),
                             "school": x.get("school"), "games": x.get("games"), "hire_date": coach.get("hireDate")})
    df = pd.DataFrame(rows).astype({"games": "Int64"}).sort_values(["season", "school", "coach_id"]).reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    print(f"coaches.parquet: {len(df)} coach-seasons, {df['season'].min()}-{df['season'].max()}")
    return df


if __name__ == "__main__":
    pull()
