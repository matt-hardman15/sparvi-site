"""Which farm systems make players better during the season, and is it the same orgs every year?

For each Triple-A season in sparvi.toml, split the season at its median pitch date into two halves.
Players are attributed to the MLB org whose Triple-A club they played for (aaa_affiliates_SEASON).

  Pitchers  >= MIN_PITCHES Triple-A pitches in each half: change in Stuff+, Pitching+, and primary
            fastball velocity (second half minus first).
  Hitters   >= MIN_PA Triple-A PA in each half: change in chase rate (first minus second, so a drop is
            an improvement), zone contact, and EV90.

Every change is oriented so positive = better, and league-adjusted: the mean change of all included
players that season is subtracted (velocity rises into summer for almost everyone, for example).

Excluded, and counted:
  - Players who played for more than one Triple-A club in the season (each club is a different org).
  - Players with meaningful MLB time that season before their first Triple-A game (rehab assignments
    and optioned big leaguers): >= REHAB_MLB_PITCHES MLB pitches thrown or >= REHAB_MLB_PA MLB PA.

Org score = mean league-adjusted change of the org's players, shrunk toward the league mean by
empirical Bayes (method-of-moments between-org variance), with a 90% bootstrap interval (players
resampled within the org; shrinkage factor held fixed). The test of whether an org effect exists is
the correlation of org means between seasons, with a bootstrap interval over orgs.

Survivorship: pitchers who qualified in the first half but were promoted (fewer than MIN_PITCHES
Triple-A pitches and >= MIN_PITCHES MLB pitches in the second half) leave the sample. They are counted
per org, and a sensitivity version adds them back using their MLB second half (Stuff+, Pitching+, and
velocity are graded on the same scale at both levels). Hitters aren't added back: MLB swing and
contact rates aren't comparable to Triple-A ones without a translation.

Reads the stg_pitches view from pd_pipeline (run `dbt build` first) and pitch-level predictions from
pitch_quality/train.py.
"""
import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from sparvi.config import RAW, baseball
from sparvi.provenance import provenance

CFG = baseball()
HERE = Path(__file__).resolve().parent
PD = HERE.parent / "pd_pipeline"
PQ = HERE.parent / "pitch_quality" / "output"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
SEASONS = list(CFG.aaa_seasons)
MIN_PITCHES, MIN_PA = 100, 100
REHAB_MLB_PITCHES, REHAB_MLB_PA = 100, 50
BOOT, SEED = 2000, 23

# metric -> (side, label, sign): sign turns "second half minus first" into "positive = better"
METRICS = {
    "stuff_plus": ("pitchers", "Stuff+", 1),
    "pitching_plus": ("pitchers", "Pitching+", 1),
    "fb_velo": ("pitchers", "Fastball velocity (mph)", 1),
    "chase_pct": ("hitters", "Chase rate (pts)", -1),
    "zone_contact_pct": ("hitters", "Zone contact (pts)", 1),
    "ev90": ("hitters", "EV90 (mph)", 1),
}
PCT = {"chase_pct", "zone_contact_pct"}  # stored as shares; changes reported in percentage points

os.chdir(PD)  # stg_pitches resolves parquet paths relative to the dbt project
con = duckdb.connect("pd.duckdb", read_only=True, config={"threads": 1})
ref = pd.read_csv(PQ / "scale_reference.csv").query("grp == 'all'").set_index("kind")
aff = pd.concat([pd.read_parquet(RAW / f"aaa_affiliates_{y}.parquet") for y in SEASONS])
con.register("aff", aff[["season", "team", "org"]])
split = {int(s): d for s, d in con.sql(
    "select season, quantile_disc(game_date, 0.5) from stg_pitches where level = 'AAA' group by 1").fetchall()}
con.register("halves", pd.DataFrame({"season": list(split), "split_date": list(split.values())}))


def plus(kind: str, col: str) -> str:
    return f"100 + 10 * (100 * avg({col}) - {float(ref.loc[kind, 'mean'])!r}) / {float(ref.loc[kind, 'sd'])!r}"


def preds() -> str:
    files = {lv: [(PQ / f"pitch_predictions_{lv.lower()}_{y}.parquet").as_posix() for y in SEASONS] for lv in ("AAA", "MLB")}
    return " union all ".join(
        f"select '{lv}' as level, game_pk, at_bat_number, pitch_number, xrv_stuff, xrv_pitching from read_parquet({f})"
        for lv, f in files.items())


# Pitch-level rows (internal only; nothing below the player-half level is written out).
con.sql(f"""
    create temp table p as
    select s.*, case when s.game_date <= h.split_date then 1 else 2 end as half, x.xrv_stuff, x.xrv_pitching
    from stg_pitches s join halves h using (season)
    left join ({preds()}) x using (level, game_pk, at_bat_number, pitch_number)
    where s.season in ({','.join(map(str, SEASONS))})
""")

# Who's in: one Triple-A club all season, no meaningful MLB time before the first Triple-A game.
status = con.sql(f"""
    with aaa as (
        select season, 'pitchers' as side, pitcher_id as player_id, count(distinct pitching_team) as clubs,
               any_value(pitching_team) as team, min(game_date) as first_aaa
        from p where level = 'AAA' group by all
        union all
        select season, 'hitters', batter_id, count(distinct batting_team), any_value(batting_team), min(game_date)
        from p where level = 'AAA' group by all
    ),
    mlb_before as (
        select a.season, a.side, a.player_id,
               count(*) filter (where a.side = 'pitchers' and m.pitcher_id = a.player_id) as mlb_pitches,
               count(*) filter (where a.side = 'hitters' and m.batter_id = a.player_id and m.ends_pa) as mlb_pa
        from aaa a join p m on m.level = 'MLB' and m.season = a.season and m.game_date < a.first_aaa
             and (m.pitcher_id = a.player_id or m.batter_id = a.player_id)
        group by all
    )
    select a.*, f.org, coalesce(b.mlb_pitches, 0) as mlb_pitches_before, coalesce(b.mlb_pa, 0) as mlb_pa_before,
           a.clubs > 1 as multi_club,
           coalesce(b.mlb_pitches, 0) >= {REHAB_MLB_PITCHES} or coalesce(b.mlb_pa, 0) >= {REHAB_MLB_PA} as mlb_first
    from aaa a join aff f on f.season = a.season and f.team = a.team
    left join mlb_before b on b.season = a.season and b.side = a.side and b.player_id = a.player_id
""").df()

# Player-half lines, both levels (MLB is used only for the survivorship check).
pit = con.sql(f"""
    with fb as (  -- primary fastball per pitcher-season at Triple-A: the more-thrown of FF and SI
        select season, pitcher_id, arg_max(pitch_type, n) as fb_type from (
            select season, pitcher_id, pitch_type, count(*) n from p
            where level = 'AAA' and pitch_type in ('FF', 'SI') group by all) group by all
    )
    select p.level, p.season, p.pitcher_id as player_id, p.half, count(*) as n,
           {plus('stuff', 'xrv_stuff')} as stuff_plus,
           {plus('pitching', 'xrv_pitching')} as pitching_plus,
           avg(release_speed) filter (where p.pitch_type = fb.fb_type) as fb_velo
    from p left join fb using (season, pitcher_id)
    group by all
""").df()
hit = con.sql("""
    select level, season, batter_id as player_id, half, count(*) filter (where ends_pa) as n,
           avg(is_swing::int) filter (where not in_zone) as chase_pct,
           1 - avg(is_whiff::int) filter (where is_swing and in_zone) as zone_contact_pct,
           quantile_cont(launch_speed, 0.9) filter (where is_bip) as ev90
    from p group by all
""").df()
names = (pd.concat([pd.read_parquet(RAW / f"milb_players_{y}.parquet", columns=["player_id", "name"]) for y in SEASONS[::-1]])
         .drop_duplicates("player_id"))
roles = con.sql("select season, pitcher_id as player_id, role from dim_pitcher_role where level = 'AAA'").df()


def wide(df: pd.DataFrame, level: str, cols: list[str]) -> pd.DataFrame:
    w = df[df.level == level].pivot_table(index=["season", "player_id"], columns="half", values=["n"] + cols)
    w.columns = [f"{c}{h}" for c, h in w.columns]
    return w.reset_index()


def changes(side: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(included players with changes, promoted players with their MLB second halves)."""
    df, min_n = (pit, MIN_PITCHES) if side == "pitchers" else (hit, MIN_PA)
    cols = [m for m, v in METRICS.items() if v[0] == side]
    a = wide(df, "AAA", cols)
    st = status[status.side == side].drop(columns="side")
    a = a.merge(st, on=["season", "player_id"])
    q1 = a[a.n1.fillna(0) >= min_n]
    ok = q1[~q1.multi_club & ~q1.mlb_first]
    inc = ok[ok.n2.fillna(0) >= min_n].copy()
    m = wide(df, "MLB", cols)[["season", "player_id", "n2"] + [f"{c}2" for c in cols]]
    cand = ok[ok.n2.fillna(0) < min_n].drop(columns=["n2"] + [f"{c}2" for c in cols])
    promoted = cand.merge(m, on=["season", "player_id"])
    promoted = promoted[promoted.n2 >= min_n].copy()
    for d in (inc, promoted):
        for c in cols:
            if len(d):
                d[f"{c}_chg"] = METRICS[c][2] * (d[f"{c}2"] - d[f"{c}1"]) * (100 if c in PCT else 1)
    return inc, promoted


def adjust(d: pd.DataFrame, cols: list[str], means: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Subtract the league-wide mean change per season (from `means` if given)."""
    d = d.copy()
    means = means or {c: d.groupby("season")[f"{c}_chg"].mean().to_dict() for c in cols}
    for c in cols:
        d[f"{c}_adj"] = d[f"{c}_chg"] - d["season"].map(means[c])
    return d, means


def eb(d: pd.DataFrame, col: str, rng: np.random.Generator) -> pd.DataFrame:
    """Org scores for one season and metric: raw mean, shrunk mean, 90% bootstrap interval."""
    d = d.dropna(subset=[col])
    g = d.groupby("org")[col]
    o = pd.DataFrame({"n": g.size(), "raw": g.mean()})
    s2 = float((d[col] - d.groupby("org")[col].transform("mean")).pow(2).sum() / (len(d) - len(o)))
    se2 = s2 / o["n"]
    mu = float(d[col].mean())
    tau2 = max(0.0, float(o["raw"].var(ddof=1) - se2.mean()))
    o["shrink"] = tau2 / (tau2 + se2)
    o["score"] = mu + o["shrink"] * (o["raw"] - mu)
    lo, hi = [], []
    for org, grp in d.groupby("org"):
        v = grp[col].to_numpy()
        means = v[rng.integers(0, len(v), (BOOT, len(v)))].mean(axis=1)
        sh = mu + o.loc[org, "shrink"] * (means - mu)
        lo.append(float(np.percentile(sh, 5)))
        hi.append(float(np.percentile(sh, 95)))
    o["lo"], o["hi"] = lo, hi
    return o.assign(tau=np.sqrt(tau2), within_sd=np.sqrt(s2), reliability=tau2 / (tau2 + se2.mean()))


def consistency(o25: pd.Series, o26: pd.Series, rng: np.random.Generator) -> dict:
    j = pd.concat([o25.rename("a"), o26.rename("b")], axis=1).dropna()
    r = float(j.a.corr(j.b))
    idx = rng.integers(0, len(j), (BOOT, len(j)))
    a, b = j.a.to_numpy()[idx], j.b.to_numpy()[idx]
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    with np.errstate(invalid="ignore"):  # a draw of identical orgs has no correlation; skip it
        rs = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return {"r": r, "ci90": [float(np.nanpercentile(rs, 5)), float(np.nanpercentile(rs, 95))], "n_orgs": int(len(j))}


rng = np.random.default_rng(SEED)
y0, y1 = SEASONS[0], SEASONS[-1]
results = {"seasons": SEASONS, "split_dates": {str(k): str(v) for k, v in split.items()},
           "min_pitches": MIN_PITCHES, "min_pa": MIN_PA,
           "rehab_mlb_pitches": REHAB_MLB_PITCHES, "rehab_mlb_pa": REHAB_MLB_PA,
           "bootstrap_samples": BOOT, "metrics": {}, "sides": {}}
org_rows, player_rows = [], []
for side in ("pitchers", "hitters"):
    cols = [m for m, v in METRICS.items() if v[0] == side]
    inc, promoted = changes(side)
    inc, means = adjust(inc, cols)
    st = status[status.side == side]
    first_half_n = (pit if side == "pitchers" else hit).query("level == 'AAA' and half == 1")
    q1 = first_half_n[first_half_n.n >= (MIN_PITCHES if side == "pitchers" else MIN_PA)][["season", "player_id"]]
    q1 = q1.merge(st, on=["season", "player_id"])
    results["sides"][side] = {
        str(y): {"included": int((inc.season == y).sum()),
                 "first_half_qualified": int((q1.season == y).sum()),
                 "excluded_multi_club": int(((q1.season == y) & q1.multi_club).sum()),
                 "excluded_mlb_first": int(((q1.season == y) & q1.mlb_first & ~q1.multi_club).sum()),
                 "promoted": int((promoted.season == y).sum())}
        for y in SEASONS}
    results["sides"][side]["league_mean_change"] = {c: {str(k): v for k, v in means[c].items()} for c in cols}
    prom_n = promoted.groupby(["season", "org"]).size()
    if side == "pitchers":
        sens, _ = adjust(pd.concat([inc, promoted.assign(promoted=True)], ignore_index=True), cols)
    for c in cols:
        per = {}
        for y in SEASONS:
            o = eb(inc[inc.season == y], f"{c}_adj", rng)
            per[y] = o
            for org, r in o.iterrows():
                org_rows.append({"metric": c, "season": y, "org": org, "n": int(r.n), "raw": r.raw,
                                 "score": r.score, "lo": r.lo, "hi": r.hi, "shrink": r.shrink,
                                 "promoted": int(prom_n.get((y, org), 0))})
        res = {"label": METRICS[c][1], "side": side,
               "consistency": consistency(per[y0]["raw"], per[y1]["raw"], rng),
               "by_season": {str(y): {"n_players": int(per[y]["n"].sum()), "n_orgs": int(len(per[y])),
                                      "median_org_n": float(per[y]["n"].median()),
                                      "between_org_sd": float(per[y]["tau"].iloc[0]),
                                      "within_org_sd": float(per[y]["within_sd"].iloc[0]),
                                      "reliability": float(per[y]["reliability"].iloc[0]),
                                      "raw_sd": float(per[y]["raw"].std(ddof=1)),
                                      "player_sd": float(inc.loc[inc.season == y, f"{c}_adj"].std(ddof=1))}
                              for y in SEASONS}}
        if side == "pitchers":
            sp = {y: sens[sens.season == y].groupby("org")[f"{c}_adj"].mean() for y in SEASONS}
            res["consistency_with_promoted"] = consistency(sp[y0], sp[y1], rng)
            pm = promoted.copy()
            pm[f"{c}_chg_adj"] = pm[f"{c}_chg"] - pm["season"].map(means[c])
            res["promoted_mean_change"] = {str(y): float(pm.loc[pm.season == y, f"{c}_chg_adj"].mean())
                                           for y in SEASONS}
        results["metrics"][c] = res
    keep = ["season", "player_id", "org", "n1", "n2"] + [f"{c}{h}" for c in cols for h in (1, 2)] \
        + [f"{c}_adj" for c in cols]
    pl = inc[keep].assign(side=side, promoted=False)
    if side == "pitchers":  # hitters' MLB swing and contact rates aren't comparable; counted, not shown
        pr, _ = adjust(promoted, cols, means)
        pl = pd.concat([pl, pr[keep].assign(side=side, promoted=True)], ignore_index=True)
    player_rows.append(pl)

orgs = pd.DataFrame(org_rows).merge(aff[["season", "org", "team", "team_name"]].rename(
    columns={"team": "club", "team_name": "club_name"}), on=["season", "org"], how="left")
players = pd.concat(player_rows, ignore_index=True)
players = players.merge(roles, on=["season", "player_id"], how="left")
players = players.merge(names, on="player_id", how="left")

orgs.sort_values(["metric", "season", "org"]).to_csv(OUT / "org_scores.csv", index=False, float_format="%.6g")
players.sort_values(["side", "season", "org", "player_id"]).to_csv(
    OUT / "player_changes.csv", index=False, float_format="%.6g")
results["provenance"] = provenance([RAW / f"statcast_{lv}_{y}.parquet" for lv in ("aaa", "mlb") for y in SEASONS],
                                   seeds={"bootstrap": SEED})
(OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps({k: {"r": v["consistency"]["r"], "ci": v["consistency"]["ci90"],
                      "rel": {s: round(b["reliability"], 3) for s, b in v["by_season"].items()},
                      "n": {s: b["n_players"] for s, b in v["by_season"].items()},
                      **({"with_promoted": v["consistency_with_promoted"]["r"]} if "consistency_with_promoted" in v else {})}
                  for k, v in results["metrics"].items()}, indent=1))
print(json.dumps(results["sides"], indent=1))
