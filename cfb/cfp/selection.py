"""Did the committee pick the best 12? (CFP-12 article, frozen at the season's selection day.)

1. The field and the power rating at selection. The committee's final ranking (its last release of the
   regular season) and the power rating after the last regular-season week (conference title games).
   The field: 2014-2023, the committee's top 4. From 2024, the five highest-ranked conference champions
   plus the seven highest-ranked other teams; the rule is checked against the teams that actually played
   in the playoff (games labeled "College Football Playoff"), and it must match for every season played.
   Before selection day the field is the committee's latest ranking read as if the season ended that
   week, with each conference's highest-ranked team standing in for its champion (`final: false`).
2. The fair comparison is the at-large picks: automatic bids follow a rule, not a judgment. The power
   rating's picks are the best-rated teams outside the automatic bids, as many as there are at-large
   spots. Overlap, who was left out that the rating prefers, and by how many points.
3. How sure. The season's games through selection are resampled with replacement (BOOT times, fixed
   seed), the efficiency half of the power rating is refit on the resampled plays, and Elo is held at
   its value (it isn't resampled; the intervals are a little narrow for that reason). From the resamples:
   each team's chance to rate among the best at-large candidates, and for each left-out team the chance
   it rates above the lowest-rated at-large pick.
4. Does it matter? Every playoff game 2014 to last season (both teams in the field): how often the team
   ranked higher by the committee won, and how often the team rated higher by the power rating did.
   The power rating here is as of selection, the same moment as the committee's ranking.

Reads the football dbt database (stg_cfb_plays), cfb/adjusted_ratings/output (power rating and its model),
and data/raw/cfb games, teams and rankings. Writes output/results.json (quoted via cfb/export_site_data.py).
"""
from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from football.common.adjust import ridge_ratings
from sparvi.config import RAW, ROOT, cfb
from sparvi.provenance import provenance

PUB_SEASON = 2026          # the article's season; its numbers stay as of that season's selection day
SELECTION_DAY = date(2026, 12, 6)
BOOT, SEED = 1000, 11
FIELD_12_FROM = 2024
AUTO_BIDS, FIELD_12 = 5, 12
TOP_N = 20                  # teams shown per season: the power rating's top 20, plus any of the field below it
CFG = cfb()
OUT = Path(__file__).parent / "output"
ADJ = ROOT / "cfb" / "adjusted_ratings" / "output"
DB = Path(os.environ.get("FOOTBALL_DB_PATH", ROOT / "football" / "dbt" / "football.duckdb"))
COMMITTEE = "Playoff Committee Rankings"


def _cat(pattern: str) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in sorted((RAW / "cfb").glob(pattern))], ignore_index=True)


def selection_week(g: pd.DataFrame, season: int) -> int:
    """The last regular-season week with a completed game other than Army-Navy, which some years is
    played the week after selection: the conference title games' week."""
    s = g[(g["season"] == season) & (g["seasonType"] == "regular") & g["completed"].fillna(False)]
    s = s[~(s["homeTeam"].isin(["Army", "Navy"]) & s["awayTeam"].isin(["Army", "Navy"]))]
    return int(s["week"].max())


def committee_final(r: pd.DataFrame, season: int, sel_week: int) -> tuple[pd.DataFrame, bool]:
    c = r[(r["season"] == season) & (r["poll"] == COMMITTEE) & (r["seasonType"] == "regular")]
    wk = int(c["week"].max())
    # The final release comes the day after the title games; CFBD labels it that week or the next.
    final = wk >= sel_week and (season != PUB_SEASON or datetime.now(UTC).date() > SELECTION_DAY)
    return c[c["week"] == wk][["rank", "school"]].sort_values("rank").reset_index(drop=True), final


def champions(g: pd.DataFrame, teams: pd.DataFrame, season: int, ranking: pd.DataFrame, final: bool) -> dict[str, str]:
    """Conference -> champion: title-game winners; before they're played, each conference's
    highest-ranked team (the committee's own "if the season ended today" reading)."""
    conf = teams[teams["season"] == season].set_index("school")["conference"]
    if final:
        t = g[(g["season"] == season) & (g["seasonType"] == "regular") & g["completed"].fillna(False)
              & g["notes"].fillna("").str.contains("Championship") & ~g["notes"].fillna("").str.contains("Playoff")]
        win = np.where(t["homePoints"] > t["awayPoints"], t["homeTeam"], t["awayTeam"])
        return {conf.get(w): w for w in win if conf.get(w) not in (None, "FBS Independents")}
    out = {}
    for s in ranking["school"]:
        c = conf.get(s)
        if c and c != "FBS Independents" and c not in out:
            out[c] = s
    return out


def field(season: int, ranking: pd.DataFrame, champs: dict[str, str]) -> tuple[list[str], list[str]]:
    """(field in committee order, automatic bids)."""
    order = list(ranking["school"])
    if season < FIELD_12_FROM:
        return order[:4], []
    champ_set = set(champs.values())
    auto = [s for s in order if s in champ_set][:AUTO_BIDS]
    if len(auto) < AUTO_BIDS:
        # A champion outside the committee's top 25 still takes a bid (none has yet).
        raise ValueError(f"{season}: fewer than {AUTO_BIDS} ranked conference champions; extend the rule")
    at_large = [s for s in order if s not in auto][:FIELD_12 - AUTO_BIDS]
    return [s for s in order if s in set(auto) | set(at_large)], auto


def played_field(g: pd.DataFrame, season: int) -> set[str]:
    p = g[(g["season"] == season) & (g["seasonType"] == "postseason") & g["notes"].fillna("").str.contains("College Football Playoff")]
    return set(p["homeTeam"]) | set(p["awayTeam"])


def power_at(power: pd.DataFrame, season: int, week: int) -> pd.Series:
    p = power[(power["season"] == season) & (power["week"] == week)]
    return p.set_index("team")["power"].sort_values(ascending=False)


# --- bootstrap ------------------------------------------------------------------------------------------

def load_plays(season: int, week: int) -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute("""select game_id, offense, defense, is_home::int as is_home, epa from stg_cfb_plays
                        where season = ? and week <= ? and not is_garbage
                        order by game_id, play_id""", [season, week]).df()
    con.close()
    return df


def power_from(plays: pd.DataFrame, elo: pd.Series, lam: float, b_net: float, b_elo: float) -> pd.Series:
    r = ridge_ratings(plays, lam).set_index("team")
    net = (r["off_ridge"] - r["def_ridge"]).reindex(elo.index)
    raw = b_net * net + b_elo * elo
    return (raw - raw.mean()).sort_values(ascending=False)


def bootstrap(season: int, week: int, base: pd.Series, elo: pd.Series, model: dict, lam: float) -> pd.DataFrame:
    """BOOT x team matrix of power ratings, refit on resampled games."""
    plays = load_plays(season, week)
    # Check the refit reproduces the published rating before trusting resamples of it.
    again = power_from(plays, elo, lam, model["points_per_net_epa"], model["points_per_elo"])
    gap = float((again - base.reindex(again.index)).abs().max())
    assert gap < 1e-3, f"refit differs from power_team_week by {gap:.4f} points"
    ids = plays["game_id"].unique()
    by_game = {gid: d for gid, d in plays.groupby("game_id", sort=True)}
    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(BOOT):
        draw = rng.choice(ids, size=len(ids), replace=True)
        rows.append(power_from(pd.concat([by_game[i] for i in draw], ignore_index=True), elo, lam,
                               model["points_per_net_epa"], model["points_per_elo"]))
    return pd.DataFrame(rows).reset_index(drop=True)


# --- one season ------------------------------------------------------------------------------------------

def season_view(season, g, r, teams, power) -> dict:
    sel = selection_week(g, season)
    ranking, final = committee_final(r, season, sel)
    if not final and season != PUB_SEASON:
        raise ValueError(f"{season}: no committee ranking after the last regular-season week")
    champs = champions(g, teams, season, ranking, final)
    fld, auto = field(season, ranking, champs)
    if final and season >= FIELD_12_FROM:
        played = played_field(g, season)
        if played:      # once the bracket is in the data, the rule must reproduce it exactly
            assert played == set(fld), f"{season}: field rule {sorted(set(fld) - played)} vs played {sorted(played - set(fld))}"
    pw = power_at(power, season, sel if final else int(power[power["season"] == season]["week"].max()))
    rank = pw.rank(ascending=False, method="min").astype(int)
    pool = [t for t in pw.index if t not in auto]
    k = len(fld) - len(auto)
    picks = [t for t in fld if t not in auto]
    model_picks = pool[:k]
    left_out = [t for t in model_picks if t not in picks]
    worst_pick = min(picks, key=lambda t: pw.get(t, -np.inf))
    crank = dict(zip(ranking["school"], ranking["rank"], strict=True))
    po = g[(g["season"] == season) & (g["seasonType"] == "postseason") & g["completed"].fillna(False)
           & g["homeTeam"].isin(fld) & g["awayTeam"].isin(fld)].sort_values("startDate")
    champ = None
    if len(po) == len(fld) - 1:          # the bracket is complete: the last game is the title game
        last = po.iloc[-1]
        champ = last["homeTeam"] if last["homePoints"] > last["awayPoints"] else last["awayTeam"]
    return {
        "champion": champ, "champion_committee_rank": int(crank[champ]) if champ else None,
        "champion_power_rank": int(rank[champ]) if champ else None,
        "season": season, "final": final, "selection_week": sel, "field_size": len(fld),
        "rows": [{"team": t, "committee_rank": int(crank[t]), "power": round(float(pw[t]), 2), "power_rank": int(rank[t]),
                  "bid": "auto" if t in auto else "at-large"} for t in fld],
        "top": [{"team": t, "power": round(float(pw[t]), 2), "power_rank": int(rank[t]),
                 "committee_rank": int(crank[t]) if t in crank else None, "in_field": t in fld}
                for t in list(pw.index[:TOP_N]) + [t for t in fld if t not in set(pw.index[:TOP_N])]],
        "at_large_spots": k, "overlap": k - len(left_out),
        "left_out": [{"team": t, "power": round(float(pw[t]), 2), "power_rank": int(rank[t]),
                      "committee_rank": int(crank[t]) if t in crank else None} for t in left_out],
        "chosen_instead": [{"team": t, "power": round(float(pw[t]), 2), "power_rank": int(rank[t]),
                            "committee_rank": int(crank[t])} for t in picks if t not in model_picks],
        "worst_pick": worst_pick, "worst_pick_power_rank": int(rank[worst_pick]),
        "best_left_out": left_out[0] if left_out else None,
        "_power": pw, "_auto": auto, "_picks": picks, "_week": sel if final else int(power[power["season"] == season]["week"].max()),
    }


def playoff_games(g: pd.DataFrame, views: dict[int, dict]) -> dict:
    rows = []
    for s, v in views.items():
        if s >= PUB_SEASON:
            continue
        fld = {x["team"]: x for x in v["rows"]}
        p = g[(g["season"] == s) & (g["seasonType"] == "postseason") & g["completed"].fillna(False)
              & g["homeTeam"].isin(fld) & g["awayTeam"].isin(fld)]
        for _, x in p.iterrows():
            h, a = fld[x["homeTeam"]], fld[x["awayTeam"]]
            home_won = x["homePoints"] > x["awayPoints"]
            rows.append({"season": s, "committee_right": (h["committee_rank"] < a["committee_rank"]) == home_won,
                         "power_right": (h["power"] > a["power"]) == home_won,
                         "disagree": (h["committee_rank"] < a["committee_rank"]) != (h["power"] > a["power"])})
    d = pd.DataFrame(rows)
    dis = d[d["disagree"]]
    return {"n": int(len(d)), "committee_right": int(d["committee_right"].sum()), "power_right": int(d["power_right"].sum()),
            "n_disagree": int(len(dis)), "power_right_when_disagree": int(dis["power_right"].sum()),
            "seasons": [int(d["season"].min()), int(d["season"].max())]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    g, r = _cat("games_*.parquet"), _cat("rankings_*.parquet")
    teams = pd.concat([pd.read_parquet(p, columns=["school", "conference"]).assign(season=int(p.stem.split("_")[1]))
                       for p in sorted((RAW / "cfb").glob("teams_*.parquet"))], ignore_index=True)
    power = pd.read_csv(ADJ / "power_team_week.csv")
    metrics = json.loads((ADJ / "metrics.json").read_text(encoding="utf-8"))
    views = {s: season_view(s, g, r, teams, power) for s in CFG.seasons if s <= PUB_SEASON}

    v = views[PUB_SEASON]
    pw = v["_power"]
    elo = power[(power["season"] == PUB_SEASON) & (power["week"] == v["_week"])].set_index("team")["elo"].reindex(pw.index)
    boot = bootstrap(PUB_SEASON, v["_week"], pw, elo, metrics["power_model"], metrics["lambda"])
    pool = [t for t in pw.index if t not in v["_auto"]]
    k = v["at_large_spots"]
    pool_boot = boot[pool]
    top_k = pool_boot.rank(axis=1, ascending=False, method="first") <= k
    worst = v["worst_pick"]
    cand = [t for t in pool[: k + 6]] + [t for t in v["_picks"] if t not in pool[: k + 6]]
    sure = [{"team": t, "in_field": t in v["_picks"], "power": round(float(pw[t]), 2),
             "p_top": round(float(top_k[t].mean()), 3),
             "p_above_worst_pick": None if t == worst else round(float((boot[t] > boot[worst]).mean()), 3),
             "lo90": round(float(boot[t].quantile(0.05)), 2), "hi90": round(float(boot[t].quantile(0.95)), 2)}
            for t in cand]
    exp_overlap = float(top_k[v["_picks"]].sum(axis=1).mean())

    hist = [{k2: x for k2, x in vv.items() if not k2.startswith("_")} for s, vv in views.items()]
    res = {"pub_season": PUB_SEASON, "final": v["final"], "boot": BOOT, "seed": SEED,
           "season": hist[-1], "bubble": sure, "expected_overlap": round(exp_overlap, 2),
           "history": hist[:-1], "games": playoff_games(g, views), "field_12_from": FIELD_12_FROM, "auto_bids": AUTO_BIDS, "field_12": FIELD_12,
           "provenance": provenance(sorted((RAW / "cfb").glob("games_*.parquet")), seeds={"bootstrap": SEED},
                                    date_expr="startDate::timestamp")}
    (OUT / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k2: x for k2, x in res.items() if k2 not in ("history", "provenance")}, indent=2))
    for h in res["history"]:
        print(h["season"], h["field_size"], f"overlap {h['overlap']}/{h['at_large_spots']}", "left out:",
              [(x["team"], x["power_rank"]) for x in h["left_out"]], "instead:", [(x["team"], x["power_rank"]) for x in h["chosen_instead"]],
              "champion:", h["champion"], h["champion_committee_rank"], h["champion_power_rank"])


if __name__ == "__main__":
    main()
