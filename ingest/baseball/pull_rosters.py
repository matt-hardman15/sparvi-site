"""Pull the inputs for the Rule 5 board from the MLB Stats API into data/raw/ (gitignored).

    uv run python ingest/baseball/pull_rosters.py                 # snapshot dated today
    uv run python ingest/baseball/pull_rosters.py --ids 123 456   # also fetch these people (validation list)

Writes, for snapshot date D:
  data/raw/rosters_40man_D.parquet       every club's 40-man roster as of D
  data/raw/aaa_affiliates_SEASON.parquet Triple-A club -> parent MLB org (reused by farm rankings)
  data/raw/people_rule5_D.json           bio, draft history, transactions, current team for every
                                         Triple-A player this season plus the validation list
  data/raw/milb_season_end.json          minor league regular-season end date by year

Take the board's final snapshot the day after the 40-man protection deadline, when the eligible
pool is set. The analysis uses the newest snapshot it finds.
"""
import argparse
import json
import time
from datetime import date

import duckdb
import pandas as pd
import requests

from sparvi.config import RAW, ROOT, baseball

API = "https://statsapi.mlb.com/api/v1"
CFG = baseball()
SESSION = requests.Session()
VALIDATION = ROOT / "baseball" / "rule5" / "validation_list.csv"


def get(path: str, **params) -> dict:
    for attempt in range(4):
        try:
            r = SESSION.get(f"{API}{path}", params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            wait = 3 * (attempt + 1)
            print(f"  retry {attempt + 1} {path}: {e} (sleeping {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"failed: {path}")


def rosters(snapshot: str) -> pd.DataFrame:
    rows = []
    for t in get("/teams", sportId=1, season=CFG.season)["teams"]:
        for p in get(f"/teams/{t['id']}/roster", rosterType="40Man").get("roster", []):
            rows.append({"snapshot_date": snapshot, "org_id": t["id"], "org": t["abbreviation"],
                         "player_id": p["person"]["id"], "name": p["person"]["fullName"],
                         "position": p["position"]["abbreviation"], "status": p["status"]["description"]})
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    by_org = df.groupby("org").size()
    print(f"40-man rosters: {len(df)} players across {len(by_org)} clubs (sizes {by_org.min()}-{by_org.max()})")
    return df


def affiliates() -> pd.DataFrame:
    orgs = {t["id"]: t["abbreviation"] for t in get("/teams", sportId=1, season=CFG.season)["teams"]}
    rows = [{"season": CFG.season, "team_id": t["id"], "team": t["abbreviation"], "team_name": t["name"],
             "org_id": t.get("parentOrgId"), "org": orgs.get(t.get("parentOrgId"))}
            for t in get("/teams", sportId=11, season=CFG.season)["teams"]]
    return pd.DataFrame(rows)


def season_ends(first: int) -> dict[str, str]:
    """Rookie-level regular-season end by year: the cutoff for whether a signing year counts.

    A season counts toward Rule 5 eligibility if the player signed before the season of the club he's
    first assigned to ended. New signees start in rookie or complex leagues, which end in late August,
    weeks before Triple-A, so their end date is the right cutoff. Falls back to Triple-A if missing.
    """
    out = {}
    for y in range(first, CFG.season + 1):
        for sport in (16, 11):
            s = get(f"/seasons/{y}", sportId=sport).get("seasons", [])
            if s and s[0].get("regularSeasonEndDate"):
                out[str(y)] = s[0]["regularSeasonEndDate"]
                break
    return out


def aaa_player_ids() -> list[int]:
    f = (RAW / f"statcast_aaa_{CFG.season}.parquet").as_posix()
    return [r[0] for r in duckdb.sql(
        f"select distinct id from (select pitcher as id from '{f}' union select batter from '{f}') order by 1").fetchall()]


def people(ids: list[int]) -> list[dict]:
    out = []
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        out += get("/people", personIds=",".join(map(str, chunk)), hydrate="draft,transactions,currentTeam")["people"]
        print(f"  people {i + len(chunk):,}/{len(ids):,}")
        time.sleep(0.3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", type=int, default=[], help="extra player ids to fetch")
    ap.add_argument("--date", default=date.today().isoformat(), help="snapshot date label (default today)")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    rosters(args.date).to_parquet(RAW / f"rosters_40man_{args.date}.parquet", index=False)
    aff = affiliates()
    aff.to_parquet(RAW / f"aaa_affiliates_{CFG.season}.parquet", index=False)
    print(f"Triple-A affiliates: {len(aff)}, unmapped: {int(aff['org'].isna().sum())}")

    extra = list(args.ids)
    if VALIDATION.exists():
        extra += pd.read_csv(VALIDATION)["player_id"].dropna().astype(int).tolist()
    ids = sorted(set(aaa_player_ids()) | set(extra))
    (RAW / f"people_rule5_{args.date}.json").write_text(json.dumps(people(ids)), encoding="utf-8")
    (RAW / "milb_season_end.json").write_text(json.dumps(season_ends(2010), indent=1), encoding="utf-8")
    print(f"snapshot {args.date} written to {RAW}")


if __name__ == "__main__":
    main()
