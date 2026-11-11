"""Spot-check pitcher games (G) and starts (GS) from dim_pitcher_role against MLB's official stats.

    uv run python scripts/check_roles.py                    # the default check list, current season
    uv run python scripts/check_roles.py 669373 694973      # specific MLB player ids

Reads baseball/pd_pipeline/pd.duckdb (run build.sh or `dbt build` first) and the MLB Stats API
season pitching line (the source Baseball-Reference's G/GS also come from). Exits non-zero on any
mismatch, so it can be rerun after pipeline changes.
"""
import sys

import duckdb
import requests

from sparvi.config import ROOT, baseball

SEASON = baseball().season
DB = ROOT / "baseball" / "pd_pipeline" / "pd.duckdb"
# 5 starters, 5 relievers. Ragans (missed time) and the relievers' 0 GS are the useful edge cases.
DEFAULT = {
    669373: "Tarik Skubal", 694973: "Paul Skenes", 657277: "Logan Webb", 554430: "Zack Wheeler", 666142: "Cole Ragans",
    662253: "Andres Munoz", 661395: "Jhoan Duran", 695243: "Mason Miller", 628452: "Raisel Iglesias", 623352: "Josh Hader",
}


def official(pid: int) -> tuple[int, int]:
    r = requests.get(f"https://statsapi.mlb.com/api/v1/people/{pid}/stats",
                     params={"stats": "season", "group": "pitching", "season": SEASON, "sportId": 1}, timeout=30)
    r.raise_for_status()
    splits = [s for st in r.json().get("stats", []) for s in st.get("splits", [])]
    if not splits:
        return 0, 0
    # A traded pitcher has one split per team plus a total row without a team; use the total.
    total = next((s for s in splits if "team" not in s), splits[0])["stat"]
    return int(total["gamesPlayed"]), int(total["gamesStarted"])


def main(ids: list[int]) -> int:
    con = duckdb.connect(str(DB), read_only=True)
    ours = {r[0]: r[1:] for r in con.execute(
        "select pitcher_id, games, starts, role from dim_pitcher_role where level = 'MLB' and season = ? "
        "and pitcher_id in (select unnest(?))", [SEASON, ids]).fetchall()}
    bad = 0
    print(f"{'pitcher':<18} {'id':>7}  role   G ours/MLB   GS ours/MLB")
    for pid in ids:
        g, gs, role = ours.get(pid, (0, 0, "-"))
        og, ogs = official(pid)
        ok = (g, gs) == (og, ogs)
        bad += not ok
        print(f"{DEFAULT.get(pid, ''):<18} {pid:>7}  {role:<5} {g:>4} / {og:<4}   {gs:>4} / {ogs:<4}  {'ok' if ok else 'MISMATCH'}")
    print(f"\n{len(ids) - bad} of {len(ids)} match exactly (season {SEASON}, regular season, MLB).")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main([int(a) for a in sys.argv[1:]] or list(DEFAULT)))
