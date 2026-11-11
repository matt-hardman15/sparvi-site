"""Estimate Rule 5 eligibility for every Triple-A player, and check it against published lists.

The rule: a player who was 18 or younger on the June 5 immediately preceding his first pro signing
must be added to the 40-man roster within 5 seasons, or he's eligible for the Rule 5 Draft; 19 or
older, within 4. The signing year counts as a season if he signed before that season ended for the
club he was first assigned to (rookie/complex leagues, which end in late August). Eligible means:
not on a 40-man roster, under contract (minor league free agents can't be drafted), and past that
window. This is an estimate: transactions can be missing, and edge cases (voided contracts,
players signed out of foreign leagues) aren't modeled.

Inputs (data/raw, from scripts/pull_rosters.py; newest snapshot wins):
  rosters_40man_D.parquet, people_rule5_D.json, milb_season_end.json, aaa_affiliates_SEASON.parquet
Outputs (baseball/rule5/output, committed; no birth dates):
  eligibility.csv   one row per player: status, reason, org, estimate inputs
  validation.csv    the published-list check, row by row
  results.json      counts, validation agreement, snapshot date
"""
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from sparvi.config import RAW, baseball
from sparvi.provenance import library_versions

CFG = baseball()
SEASON = CFG.season
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

JOIN = {"SGN", "SFA", "TR", "CLW", "R5", "R5M", "PUR", "OBT", "ACQ", "SE"}   # under contract after this
EXIT = {"DFA": "Free agent", "REL": "Released", "RET": "Retired"}             # DFA here = "Declared Free Agency"


def newest(pattern: str) -> Path:
    files = sorted(RAW.glob(pattern))
    if not files:
        sys.exit(f"no {pattern} in {RAW}; run scripts/pull_rosters.py")
    return files[-1]


def june5_before(d: date) -> date:
    j = date(d.year, 6, 5)
    return j if d >= j else date(d.year - 1, 6, 5)


def age_on(birth: date, day: date) -> int:
    return day.year - birth.year - ((day.month, day.day) < (birth.month, birth.day))


def first_signing(p: dict, mlb_ids: set[int]) -> tuple[date | None, str | None, str]:
    """Earliest signing with an MLB organization; else the draft year; else nothing."""
    sg = sorted((t for t in p.get("transactions", [])
                 if t.get("typeCode") in ("SGN", "SFA") and t.get("toTeam", {}).get("id") in mlb_ids and t.get("date")),
                key=lambda t: t["date"])
    if sg:
        return date.fromisoformat(sg[0]["date"]), sg[0]["toTeam"]["name"], "signing"
    drafted = [d for d in p.get("drafts", []) if d.get("isDrafted") and d.get("year")]
    if drafted:
        d = max(drafted, key=lambda x: int(x["year"]))  # the pick he signed is normally the last one
        return date(int(d["year"]), 7, 15), d.get("team", {}).get("name"), "draft"
    return None, None, "unknown"


def contract_status(p: dict) -> str | None:
    """'Free agent' / 'Released' / 'Retired' if his last exit comes after his last join; else None."""
    events = sorted((t["date"], 1 if t["typeCode"] in JOIN else 0, t["typeCode"]) for t in p.get("transactions", [])
                     if t.get("date") and t.get("typeCode") in JOIN | set(EXIT))
    # Same-day ties sort exits before joins, so a release and re-signing on one day counts as signed.
    return EXIT.get(events[-1][2]) if events else None


def org_of(p: dict, mlb: dict[int, str], affiliate_parent: dict[int, int]) -> str | None:
    t = p.get("currentTeam") or {}
    if t.get("id") in mlb:
        return mlb[t["id"]]
    parent = t.get("parentOrgId") or affiliate_parent.get(t.get("id"))
    return mlb.get(parent)


def estimate(p: dict, season_end: dict[str, str], mlb_ids: set[int]) -> dict:
    birth = date.fromisoformat(p["birthDate"]) if p.get("birthDate") else None
    signed, by, basis = first_signing(p, mlb_ids)
    out = {"signing_date": signed.isoformat() if signed else None, "signed_by": by, "basis": basis,
           "age_at_signing": None, "seasons_to_protect": None, "first_season": None, "eligible_from": None}
    if not signed or not birth:
        return out
    age = age_on(birth, june5_before(signed))
    end = date.fromisoformat(season_end.get(str(signed.year), f"{signed.year}-08-31"))
    first = signed.year if signed <= end else signed.year + 1
    n = 5 if age <= 18 else 4
    out |= {"age_at_signing": age, "seasons_to_protect": n, "first_season": first, "eligible_from": first + n - 1}
    return out


def reason(r: dict) -> str:
    if r["basis"] == "unknown":
        return "No signing record or draft year found."
    how = (f"Signed {r['signing_date']} by {r['signed_by']}" if r["basis"] == "signing"
           else f"Drafted {r['signing_date'][:4]} by {r['signed_by']} (signing date assumed mid-July)")
    if r["age_at_signing"] is None:
        return how + "; birth date unknown."
    last = r["first_season"] + r["seasons_to_protect"] - 1
    return (f"{how}, age {r['age_at_signing']} on the preceding June 5 → {r['seasons_to_protect']} seasons "
            f"({r['first_season']}–{last}) → eligible from the {r['eligible_from']} Rule 5 Draft.")


def status(r: dict) -> str:
    if r["on_40man"]:
        return "On 40-man"
    if r["contract"]:
        return r["contract"]
    if r["eligible_from"] is None:
        return "Unknown"
    if r["eligible_from"] > SEASON:
        return "Not yet eligible"
    return "Eligible" if r["basis"] == "signing" else "Likely eligible"


def main() -> None:
    roster_file = newest("rosters_40man_*.parquet")
    snap = roster_file.stem.removeprefix("rosters_40man_")
    people_file = RAW / f"people_rule5_{snap}.json"
    rosters = pd.read_parquet(roster_file)
    people = json.loads(people_file.read_text(encoding="utf-8"))
    season_end = json.loads((RAW / "milb_season_end.json").read_text(encoding="utf-8"))
    aff = pd.read_parquet(RAW / f"aaa_affiliates_{SEASON}.parquet")
    mlb = dict(zip(rosters["org_id"], rosters["org"], strict=True))
    mlb_ids = set(mlb)
    affiliate_parent = dict(zip(aff["team_id"], aff["org_id"], strict=True))
    on40 = set(rosters["player_id"])

    rows = []
    for p in people:
        r = {"player_id": p["id"], "name": p["fullName"], "org": org_of(p, mlb, affiliate_parent),
             "on_40man": p["id"] in on40, "contract": contract_status(p), **estimate(p, season_end, mlb_ids)}
        r["status"], r["reason"] = status(r), reason(r)
        rows.append(r)
    el = pd.DataFrame(rows).sort_values("player_id")
    for c in ("age_at_signing", "seasons_to_protect", "first_season", "eligible_from"):
        el[c] = el[c].astype("Int64")
    el.to_csv(OUT / "eligibility.csv", index=False)
    # Triple-A club -> MLB org, committed so CI (and the farm-system work) can use it without raw data.
    aff.sort_values("team")[["season", "team_id", "team", "team_name", "org_id", "org"]].to_csv(
        OUT / "aaa_affiliates.csv", index=False)

    # Validation against published 2026 eligibility lists (baseball/rule5/validation_list.csv).
    v = pd.read_csv(HERE / "validation_list.csv", dtype={"player_id": "Int64"}).dropna(subset=["player_id"])
    v = v.merge(el[["player_id", "eligible_from", "basis", "reason"]], on="player_id", how="left")
    v["published_eligible"] = v["first_eligible"] <= SEASON
    v["estimated_eligible"] = v["eligible_from"].fillna(9999) <= SEASON
    v["agree"] = v["published_eligible"] == v["estimated_eligible"]
    v.to_csv(OUT / "validation.csv", index=False)

    n_ok, n = int(v["agree"].sum()), len(v)
    pos, neg = v[v["published_eligible"]], v[~v["published_eligible"]]
    res = {
        "snapshot_date": snap,
        "season": SEASON,
        "players": len(el),
        "status_counts": el["status"].value_counts().to_dict(),
        "basis_counts": el["basis"].value_counts().to_dict(),
        "on_40man_snapshot": int(len(on40)),
        "validation": {
            "n": n, "agree": n_ok, "agreement": n_ok / n,
            "n_listed_eligible": len(pos), "agree_listed_eligible": int(pos["agree"].sum()),
            "n_listed_not_yet": len(neg), "agree_listed_not_yet": int(neg["agree"].sum()),
            "sources": sorted(v["source"].unique().tolist()),
        },
        "libraries": library_versions(),
    }
    (OUT / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "libraries"}, indent=2))
    bad = v[~v["agree"]]
    if len(bad):
        print("\nDisagreements:")
        for _, r in bad.iterrows():
            print(f"  {r['name']} ({r['org']}): published {r['first_eligible']}, estimated {r['eligible_from']}. {r['reason']}")


if __name__ == "__main__":
    main()
