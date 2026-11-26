"""College basketball site data (CBB-PAGES-1): the ONLY thing that writes site/data/cbb/*.json and
site/_generated/cbb.yml.

    python cbb/export_site_data.py           # run each morning by .github/workflows/cbb.yml, after the pull

Writes:
    site/data/cbb/ratings.json   every Division I team's rating this morning: net, offense, defense, tempo, record
    site/data/cbb/meta.json      the season, the date the ratings run through, league terms, the model's version and state
    site/data/cbb/teams.json     every Division I team: id, page slug, name, conference (CBB-PAGES-1b)
    site/data/cbb/team_games.json  each team's games this season: date, opponent, site and, once final, the score
    site/_generated/cbb.yml      every number quoted in basketball prose
    site/data/status/cbb.json    the data checks' result for /status/ (check names, pass or fail, row counts: no data),
                                 copied from where ingest/cbb/checks.py writes it beside the raw files

The ratings are the published team rating's (cbb/ratings/daily.py `morning`): this season's counted games before today,
with last season's final ratings as the prior. Only our own ratings, team names, conferences and win-loss records are
written: no box scores, no lines, none of the source's ratings (no raw data in site/ or the repo).
"""
from __future__ import annotations

import json

import pandas as pd

from backtest.asof import Loader
from backtest.gate import require_passing
from cbb.ratings import daily as D
from cbb.ratings import model as M
from sparvi import registry
from sparvi.config import RAW, ROOT, cbb
from sparvi.io import SchemaError, StaleProse, write_json, write_numbers
from sparvi.urls import slug

MODEL = "cbb_team_rating"
OUT = ROOT / "site" / "data" / "cbb"
DEV = ROOT / "cbb" / "ratings" / "output" / "development.json"
MIN_TEAMS = 300


def records(loader, season: int, as_of, on: pd.Timestamp) -> pd.DataFrame:
    """Wins and losses in every final game before game day `on` (all opponents, not only the games the ratings count).
    A final game the source lists without a score (0-0, seen for a few games against teams outside Division I) is in
    neither column."""
    g = loader.load(f"cbb_games_{season}", as_of, columns=["id", "startDate", "status", "homeTeamId", "awayTeamId", "homePoints", "awayPoints"])
    g = g[g["status"].eq("final") & g["homePoints"].notna() & g["awayPoints"].notna() & g["homePoints"].ne(g["awayPoints"])
          & (M.game_day(g["startDate"]) < on)].drop_duplicates("id")
    home = pd.DataFrame({"team": g["homeTeamId"], "w": g["homePoints"] > g["awayPoints"]})
    away = pd.DataFrame({"team": g["awayTeamId"], "w": g["awayPoints"] > g["homePoints"]})
    x = pd.concat([home, away])
    return x.groupby("team")["w"].agg(w="sum", n="size").assign(l=lambda d: d["n"] - d["w"]).drop(columns="n").astype(int).reset_index()


def team_list(loader, season: int, as_of) -> pd.DataFrame:
    """Division I teams with the slug of their page (/cbb/teams/<slug>/). A slug is the school's name; two schools that
    would share one stop the export (a published URL never changes, so this has to be settled by hand)."""
    t = loader.load(f"cbb_teams_{season}", as_of, columns=["id", "school", "conference"])
    t = t[t["conference"].notna()].drop_duplicates("id").sort_values("school").reset_index(drop=True)
    t["slug"] = t["school"].map(slug)
    dup = t[t["slug"].duplicated(keep=False)]
    if len(dup) or (t["slug"].str.len() < 2).any():
        raise SchemaError(f"cbb teams: page slugs are not unique or are empty: {sorted(dup['school'])[:6]}")
    return t


def team_games(loader, season: int, as_of, teams: pd.DataFrame, on: pd.Timestamp) -> dict:
    """{team id: [[game day, opponent id or None, opponent name, 'H' | 'A' | 'N', status, points for, points against], ...]}
    for every game on the schedule (any opponent), in date order; status is final, scheduled, postponed or "no score"
    (played, but the source has no score). A score is shown once the game is final and was
    played before game day `on`, the same games the records and ratings run through."""
    cols = ["id", "startDate", "status", "neutralSite", "homeTeamId", "homeTeam", "awayTeamId", "awayTeam", "homePoints", "awayPoints"]
    g = loader.load(f"cbb_games_{season}", as_of, columns=cols).drop_duplicates("id")
    g = g[g["status"].isin(["final", "scheduled", "in_progress", "postponed"])].assign(day=M.game_day(g["startDate"]))
    g = g.sort_values(["day", "startDate", "id"])
    known = set(teams["id"])
    out: dict[int, list] = {int(t): [] for t in teams["id"]}
    for r in g.itertuples():
        played = r.status == "final" and r.day < on
        final = played and pd.notna(r.homePoints) and pd.notna(r.awayPoints) and r.homePoints != r.awayPoints
        neutral = bool(r.neutralSite) if pd.notna(r.neutralSite) else False
        for me, opp, opp_name, site, pf, pa in ((r.homeTeamId, r.awayTeamId, r.awayTeam, "N" if neutral else "H", r.homePoints, r.awayPoints),
                                                (r.awayTeamId, r.homeTeamId, r.homeTeam, "N" if neutral else "A", r.awayPoints, r.homePoints)):
            if me in out:
                out[int(me)].append([str(r.day.date()), int(opp) if opp in known else None, str(opp_name), site,
                                     "final" if final else "no score" if played else "postponed" if r.status == "postponed" else "scheduled",
                                     int(pf) if final else None, int(pa) if final else None])
    return out


def build(loader, on: pd.Timestamp, season: int, earlier: list[int]) -> tuple[dict, dict]:
    """(ratings.json, meta.json without the model stamp) for the morning of game day `on`."""
    as_of = on + pd.Timedelta(days=2)
    r, league = D.morning(loader, on, season, earlier, as_of)
    teams = team_list(loader, season, as_of)
    d = teams.rename(columns={"id": "team"}).merge(r, on="team", how="inner").merge(records(loader, season, as_of, on), on="team", how="left")
    d[["w", "l"]] = d[["w", "l"]].fillna(0).astype(int)
    if len(d) < MIN_TEAMS:
        raise SchemaError(f"cbb ratings: {len(d)} Division I teams with a rating, expected at least {MIN_TEAMS}")
    for col, key in (("net", "rank"), ("offense", "off_rank"), ("defense", "def_rank")):
        d[key] = d[col].rank(ascending=False, method="min").astype(int)
    d = d.sort_values(["rank", "school"]).reset_index(drop=True)
    rows = [{"id": int(x.team), "slug": x.slug, "team": x.school, "conf": x.conference, "w": int(x.w), "l": int(x.l), "games": int(x.games),
             "net": round(float(x.net), 1), "off": round(float(x.offense), 1), "def": round(float(x.defense), 1),
             "tempo": round(float(x.tempo), 1), "rank": int(x["rank"]), "off_rank": int(x.off_rank), "def_rank": int(x.def_rank)}
            for _, x in d.iterrows()]
    poss = league["mu_t"]
    meta = {"season": season, "season_label": f"{season - 1}–{str(season)[2:]}", "through": str(league["through"].date()),
            "teams": len(rows), "games": league["games"], "league_efficiency": round(league["mu"], 1), "league_tempo": round(poss, 1),
            # Home court in the ratings is per 100 possessions and applies to both offenses: in points of margin at league tempo.
            "home_court_points": round(2 * league["hca"] * poss / 100, 1),
            "conferences": sorted({x["conf"] for x in rows})}
    return {"season": season, "through": meta["through"], "teams": rows}, meta


def main() -> None:
    cfg = cbb()
    status = require_passing(MODEL)       # ships only with a passing backtest or the card's experimental flag (the page says which)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    on = (now - M.UTC_TO_GAME_DAY).normalize()
    ratings, meta = build(Loader(), on, cfg.season, cfg.history_seasons)
    meta["model"] = {**registry.stamp(MODEL), "state": status["state"]}
    write_json(OUT / "ratings.json", ratings)
    checks = RAW / "cbb" / "_status" / "cbb.json"
    if checks.exists():        # now that there are pages, the checks' result is shown on /status/ like the other sports'
        write_json(ROOT / "site" / "data" / "status" / "cbb.json", json.loads(checks.read_text(encoding="utf-8")), indent=1)
    write_json(OUT / "meta.json", meta, indent=1)
    loader, as_of = Loader(), on + pd.Timedelta(days=2)
    tl = team_list(loader, cfg.season, as_of)
    write_json(OUT / "teams.json", [{"id": int(r.id), "slug": r.slug, "team": r.school, "conf": r.conference} for r in tl.itertuples()])
    write_json(OUT / "team_games.json", {"season": cfg.season, "through": meta["through"],
                                         "fields": ["date", "opp", "opp_name", "site", "status", "pf", "pa"],
                                         "teams": team_games(loader, cfg.season, as_of, tl, on)})

    dev = json.loads(DEV.read_text(encoding="utf-8"))
    p = dev["development"]["pooled"]
    top, through = ratings["teams"][0], pd.Timestamp(meta["through"])
    n1 = lambda v: f"{v:.1f}"  # noqa: E731
    N = {
        "season": str(cfg.season), "season_label": meta["season_label"], "through": f"{through:%B} {through.day}, {through.year}",
        "n_teams": str(meta["teams"]), "n_games": f"{meta['games']:,}", "home_court": n1(meta["home_court_points"]),
        "league_efficiency": n1(meta["league_efficiency"]), "league_tempo": n1(meta["league_tempo"]),
        "top_team": top["team"], "top_net": f"{top['net']:+.1f}".replace("-", "−"),
        "prior_games": f"{M.PUBLISHED.prior_weight:.0f}",
        "dev_seasons": " and ".join(str(s) for s in dev["seasons"]["develop"]), "dev_games": f"{p['games']:,}",
        "dev_mae": n1(p["margin_mae"]["second_stage"]), "dev_mae_ratings": n1(p["margin_mae"]["ratings"]),
        "dev_mae_elo": n1(p["margin_mae"]["elo"]), "dev_mae_prior": n1(p["margin_mae"]["prior"]), "dev_mae_home": n1(p["margin_mae"]["home"]),
        "dev_total_mae": n1(p["total_mae"]["model"]), "dev_accuracy": f"{100 * p['win']['second_stage']['accuracy']:.0f}%",
    }
    # Claims the ratings page makes. If one stops holding, rewrite the sentence.
    m = p["margin_mae"]
    if not m["second_stage"] < m["ratings"] < m["elo"] < m["prior"] < m["home"]:
        raise StaleProse("Basketball ratings page: the order of the development errors (model, ratings alone, Elo, last season's "
                         "ratings, home court) no longer holds.")
    if not 1.0 < meta["home_court_points"] < 6.0:
        raise StaleProse("Basketball ratings page: 'home court is worth a few points' no longer holds.")
    write_numbers("cbb", N, "cbb/export_site_data.py")
    print(f"cbb: {meta['teams']} teams through {meta['through']} ({meta['games']:,} counted games); model {status['state']}")


if __name__ == "__main__":
    main()
