"""Export aggregated, display-ready data for the website.

Only aggregates leave this script: pitcher-season and pitch-type summaries, player
report rows, and per-game averages. No pitch-level data is published.

Writes:
  site/data/baseball/*.json        tables and results the pages' explorers read
  site/_generated/baseball.yml     every number quoted in article prose, as
                                   {{< meta baseball.<key> >}}; see NUMBERS below

Run after pitch_quality/train.py, pitch_quality/robustness.py,
shape_stability/analysis.py, `dbt build` in pd_pipeline/, and two_week_signal/analysis.py.
"""
import json
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from sparvi.config import ROOT, baseball, section
from sparvi.io import write_json, write_numbers, write_records

CFG = baseball()
SEASON, PREV = CFG.season, CFG.prev
HERE = Path(__file__).resolve().parent
OUT = ROOT / "site" / "data" / "baseball"
PQ = HERE / "pitch_quality" / "output"
SS = HERE / "shape_stability" / "output"
TW = HERE / "two_week_signal" / "output"
DB = HERE / "pd_pipeline" / "pd.duckdb"


def first_last(name: str) -> str:
    """Savant uses 'Last, First'; display as 'First Last'."""
    if isinstance(name, str) and ", " in name:
        last, first = name.split(", ", 1)
        return f"{first} {last}"
    return name


def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def floats(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]


con = duckdb.connect(str(DB), read_only=True)

# ---- Pitcher roles ----------------------------------------------------------
# Role thresholds live in dbt_project.yml (dim_pitcher_role uses them); read them from there so the
# glossary can't drift from the model.
DBT_VARS = yaml.safe_load((HERE / "pd_pipeline" / "dbt_project.yml").read_text(encoding="utf-8"))["vars"]
ROLE_RULES = {k: DBT_VARS[k] for k in ("role_sp_min_share", "role_sp_min_starts", "role_rp_max_share")}
# Default pitch minimums for "qualified" pitchers on the leaderboard and explorer, by level and role.
# Swing pitchers get the reliever minimum so spot starters aren't dropped. Pages read these from meta.json.
QUALIFIERS = {"MLB": {"SP": 500, "RP": 250, "Swing": 250}, "AAA": {"SP": 300, "RP": 150, "Swing": 150}}
roles = con.sql(f"""
    select pitcher_id as id, level, role, games as g, starts as gs
    from dim_pitcher_role where season = {SEASON}
""").df()

# ---- Stuff+ ---------------------------------------------------------------
p = pd.concat([pd.read_csv(PQ / f"pitcher_season_{lvl}_{SEASON}.csv").assign(level=lvl.upper()) for lvl in ["mlb", "aaa"]])
p["name"] = p["name"].map(first_last)
p = p.rename(columns={"pitcher": "id"}).merge(roles, on=["id", "level"], how="left")
GRADES = ["stuff_plus", "loc_plus", "pitching_plus"]  # Stuff+, Location+, Pitching+ (pitch_quality/train.py)
cols = ["id", "name", "team", "level", "throws", "role", "g", "gs", "pitches", *GRADES, "rv_plus"]
write_records(OUT / "stuff_pitchers.json", p[cols].sort_values(["level", "id"]),
              required=cols, not_null=["id", "name", "level", "role"], min_rows=500, digits=dict.fromkeys([*GRADES, "rv_plus"], 1))

t = pd.concat([pd.read_csv(PQ / f"pitch_type_{lvl}_{SEASON}.csv").query("n >= 30").assign(level=lvl.upper())
               for lvl in ["mlb", "aaa"]])
t["name"] = t["player_name"].map(first_last)
t = t.rename(columns={"pitcher": "id", "pitch_type": "pt"}).merge(roles[["id", "level", "role"]], on=["id", "level"], how="left")
cols = ["id", "name", "team", "level", "role", "pt", "group", "n", "velo", "ivb", "hb", *GRADES]
write_records(OUT / "stuff_pitches.json", t[cols].sort_values(["level", "id", "pt"]),
              required=cols, not_null=["id", "name", "pt", "role"], min_rows=1000,
              digits={"velo": 1, "ivb": 1, "hb": 1} | dict.fromkeys(GRADES, 1))

yy = pd.read_csv(PQ / "yoy_pairs.csv")
yy["name"] = yy["name_prev"].map(first_last)
cols = ["id", "name", "team", "xrv100_prev", "rv100_prev", "rv100_cur"]
write_records(OUT / "stuff_yoy.json", yy.rename(columns={"pitcher": "id", "team_cur": "team"})[cols],
              required=cols, not_null=cols[3:], min_rows=100, digits={c: 3 for c in cols[3:]})

imp = pd.read_csv(PQ / "importance.csv")
write_records(OUT / "stuff_importance.json", imp, required=["group", "feature", "gain"], min_rows=12, digits={"gain": 4})

# ---- Shape persistence -----------------------------------------------------
sc = pd.read_csv(SS / "pitch_type_changes.csv")
sc["name"] = sc["player_name"].map(first_last)
cols = ["pitcher", "name", "pitch_type", "pair"]
for m in ["velo", "ivb", "hb"]:
    cols += [f"{m}1", f"{m}2", f"{m}_next", f"{m}_chg", f"{m}_kept"]
shape = sc[cols].rename(columns={"pitcher": "id", "pitch_type": "pt"})
write_records(OUT / "shape_changes.json", shape, required=list(shape.columns), not_null=["id", "name", "pt", "pair"],
              min_rows=500, digits={c: 2 for c in floats(shape)})

# ---- Triple-A player reports (dbt marts) ----------------------------------
PCT_H = ["xwoba", "ev90", "hard_hit_pct", "chase_pct", "zone_swing_pct", "zone_contact_pct", "whiff_pct", "k_pct", "bb_pct"]
h = con.sql("select * from fct_hitter_report order by player_id").df().drop(columns=["last_game"])
write_records(OUT / "aaa_hitters.json", h, min_rows=300, not_null=["player_id", "team"], digits={c: 4 for c in floats(h)},
              required=["player_id", "name", "team", "age", "pa", "recent_pa", "recent_chase_pct", "recent_whiff_pct",
                        "recent_hard_hit_pct", "recent_xwoba", "chase_delta", "whiff_delta", "hard_hit_delta"]
              + PCT_H + [f"{c}_pctl" for c in PCT_H])

PCT_P = ["k_pct", "bb_pct", "csw_pct", "whiff_pct", "chase_pct", "hard_hit_pct", "xwoba"]
pr = con.sql("select * from fct_pitcher_report order by player_id").df().drop(columns=["last_game"])
write_records(OUT / "aaa_pitchers.json", pr, min_rows=300, not_null=["player_id", "team", "role"], digits={c: 4 for c in floats(pr)},
              required=["player_id", "name", "team", "age", "role", "games", "games_started", "pitches", "bf", "fb_velo",
                        "fb_velo_delta"] + PCT_P + [f"{c}_pctl" for c in PCT_P])

ars = con.sql("select * from fct_pitcher_arsenal order by player_id, pitch_type").df()
# Arsenal and per-game velocity only feed the per-player files (no page reads them as league tables).
ars_records = write_records(None, ars, min_rows=500, not_null=["player_id", "pitch_type"], digits={c: 3 for c in floats(ars)},
              required=["player_id", "pitch_type", "pitches", "usage_pct", "velo", "ivb", "hb", "whiff_pct", "stuff_plus",
                        "recent_pitches", "velo_delta", "ivb_delta"])

g = con.sql("""
    select pitcher_id as id, strftime(game_date, '%Y-%m-%d') as d, pitch_type as pt, pitches as n,
           round(velo, 1) as velo
    from int_pitcher_games
    qualify row_number() over (partition by pitcher_id, game_pk order by pitches desc, pitch_type) = 1
    order by id, d
""").df()
fb_records = write_records(None, g, required=["id", "d", "pt", "n", "velo"], not_null=["id", "d"], min_rows=1000)

fl = con.sql("select * exclude (as_of_date) from rpt_flags order by player_id, role").df()
write_records(OUT / "aaa_flags.json", fl, min_rows=0, not_null=["player_id"],
              required=["player_id", "name", "team", "role", "flag_discipline", "flag_contact"])

# ---- Rule 5 board -------------------------------------------------------------
# Triple-A players estimated to be Rule 5 eligible and unprotected (baseball/rule5/analysis.py).
R5 = HERE / "rule5" / "output"
R5_STATUSES = ["Eligible", "Likely eligible", "Unknown"]
elig = pd.read_csv(R5 / "eligibility.csv")
r5res = read_json(R5 / "results.json")
aff = pd.read_csv(R5 / "aaa_affiliates.csv")
team_org = dict(zip(aff["team"], aff["org"], strict=True))
board_ids = elig[elig["status"].isin(R5_STATUSES)]
if board_ids["on_40man"].any():
    raise SystemExit("Rule 5 board: a player on a 40-man roster passed the eligibility filter")
e5 = board_ids[["player_id", "org", "status", "reason"]].rename(columns={"org": "cur_org"})

best = (t[(t["level"] == "AAA") & (t["n"] >= 50)].sort_values(["id", "stuff_plus"], ascending=[True, False])
        .drop_duplicates("id")[["id", "pt", "stuff_plus"]].rename(columns={"id": "player_id", "pt": "best_pt",
                                                                             "stuff_plus": "best_pt_stuff"}))
aaa_stuff = p[p["level"] == "AAA"][["id", *GRADES]].rename(columns={"id": "player_id"})
mlb_g = roles[roles["level"] == "MLB"][["id", "g"]].rename(columns={"id": "player_id", "g": "mlb_g"})


def with_org(df: pd.DataFrame) -> pd.DataFrame:
    # Current org from the roster snapshot; else the parent org of his 2026 Triple-A club.
    df["org"] = df["cur_org"].fillna(df["team"].map(team_org))
    return df.drop(columns=["cur_org"])


r5p = with_org(pr.merge(e5, on="player_id").merge(aaa_stuff, on="player_id", how="left")
               .merge(best, on="player_id", how="left").merge(mlb_g, on="player_id", how="left"))
r5p["k_bb_pct"] = r5p["k_pct"] - r5p["bb_pct"]
r5p["mlb_g"] = r5p["mlb_g"].fillna(0).astype(int)
cols = ["player_id", "name", "org", "team", "age", "role", "games", "games_started", "pitches", "bf", *GRADES,
        "k_pct", "bb_pct", "k_bb_pct", "csw_pct", "fb_velo", "best_pt", "best_pt_stuff", "mlb_g", "status", "reason"]
write_records(OUT / "rule5_pitchers.json", r5p[cols].sort_values("player_id"), required=cols,
              not_null=["player_id", "name", "status", "reason", "role"], min_rows=50,
              digits=dict.fromkeys(GRADES, 1) | {"best_pt_stuff": 1, "k_pct": 4, "bb_pct": 4, "k_bb_pct": 4, "csw_pct": 4, "fb_velo": 1})

R5_HIT = ["chase_pct", "zone_contact_pct", "ev90", "hard_hit_pct"]
r5h = with_org(h.merge(e5, on="player_id"))
cols = (["player_id", "name", "org", "team", "age", "position", "pa", "xwoba", "k_pct", "bb_pct"] + R5_HIT
        + [f"{c}_pctl" for c in R5_HIT] + ["status", "reason"])
write_records(OUT / "rule5_hitters.json", r5h[cols].sort_values("player_id"), required=cols,
              not_null=["player_id", "name", "status", "reason"], min_rows=50,
              digits={"xwoba": 3, "k_pct": 4, "bb_pct": 4, "chase_pct": 4, "zone_contact_pct": 4, "ev90": 1, "hard_hit_pct": 4})

# ---- Snapshots: weekly values for the date slider ------------------------------------
# League files are columnar to stay small:
#   {dates, fields, players: [{id, name, series: {std|last30: {field: [one value per date]}}}]}
# Per-player files (players/{id}.json) carry everything a player page needs, so a page loads one
# small file instead of every league table.
P_FIELDS = {"team": None, "role": None, "games": 0, "starts": 0, "pitches": 0, "bf": 0, "qualified": "bool",
            "stuff_plus": 1, "loc_plus": 1, "pitching_plus": 1, "rv_plus": 1, "k_pct": 3, "bb_pct": 3, "csw_pct": 3, "whiff_pct": 3, "chase_pct": 3,
            "hard_hit_pct": 3, "xwoba": 3, "fb_velo": 1}
P_PCTL = {f"{m}_pctl": 0 for m in ["k_pct", "bb_pct", "csw_pct", "whiff_pct", "chase_pct", "hard_hit_pct", "xwoba"]}
H_FIELDS = {"team": None, "pa": 0, "qualified": "bool", "chase_pct": 3, "zone_swing_pct": 3, "zone_contact_pct": 3,
            "whiff_pct": 3, "k_pct": 3, "bb_pct": 3, "hard_hit_pct": 3, "ev90": 1, "xwoba": 3}
H_PCTL = {f"{m}_pctl": 0 for m in ["chase_pct", "zone_swing_pct", "zone_contact_pct", "whiff_pct", "k_pct", "bb_pct",
                                    "hard_hit_pct", "ev90", "xwoba"]}
PT_FIELDS = {"pitches": 0, "usage_pct": 3, "velo": 1, "ivb": 1, "hb": 1, "spin": 0, "whiff_pct": 3,
             "stuff_plus": 1, "loc_plus": 1, "pitching_plus": 1}
PT_KEEP = ["pitches", "velo", "ivb", "hb", "stuff_plus", "loc_plus", "pitching_plus"]  # what the leaderboard shows


def _clean(v):
    if v is None or pd.isna(v):  # None, NaN, pd.NA, NaT
        return None
    return v.item() if hasattr(v, "item") else v


def columnar(df: pd.DataFrame, keys: list[str], fields: dict, by_window: bool = True):
    """(dates, {key tuple: {window: {field: [..]}}}) with values rounded per `fields`."""
    df = df.assign(d=df["snapshot_date"].astype(str))
    dates = sorted(df["d"].unique())
    idx = keys + (["window_name"] if by_window else [])
    out: dict = {}
    for f, dg in fields.items():
        col = df[f]
        if dg == "bool":
            col = col.astype("Int64")
        elif isinstance(dg, int):
            col = col.round(dg) if dg else col.round(0).astype("Int64")
        piv = df.assign(v=col).pivot(index=idx, columns="d", values="v").reindex(columns=dates)
        for k, row in zip(piv.index, piv.itertuples(index=False), strict=True):
            k = k if isinstance(k, tuple) else (k,)
            node = out.setdefault(k[:len(keys)], {})
            if by_window:
                node = node.setdefault(k[-1], {})
            node[f] = [_clean(v) for v in row]
    return dates, out


snap_p = con.sql("select * from fct_pitcher_snapshots order by level, pitcher_id, window_name, snapshot_date").df()
snap_h = con.sql("select * from fct_hitter_snapshots where level = 'AAA' order by batter_id, window_name, snapshot_date").df()
snap_pt = con.sql("select * from fct_pitch_type_snapshots order by level, pitcher_id, pitch_type, snapshot_date").df()

names: dict[int, str] = {}
for df_, idc in [(p, "id"), (t, "id"), (h, "player_id"), (pr, "player_id")]:
    names.update(dict(zip(df_[idc], df_["name"], strict=True)))
group_of = {k: _clean(g) for k, g in zip(zip(snap_pt["pitcher_id"], snap_pt["pitch_type"], strict=True),
                                         snap_pt["pitch_group"], strict=True)}  # None for ungraded types (EP, FA)

# Each league file is split by window: snapshots_X.json (season to date, what pages load first) and
# snapshots_X_last30.json (fetched only when someone switches to the trailing window). Fields that
# don't change by date (handedness, age, final-14-day changes) ride along as `static`, so the tools
# don't need to download the season-level tables as well.
def write_league(stem: str, level: str, dates: list, fields: dict, ser: dict, statics: dict, keep: list) -> list[dict]:
    """Write the league file(s) with only the `keep` fields the tools show; return full series for player files."""
    players = [{"id": int(k[0]), "name": names.get(k[0], str(k[0])), "static": statics.get(int(k[0]), {}), "series": v}
               for k, v in sorted(ser.items())]
    for win, suffix in [("std", ""), ("last30", "_last30")]:
        write_json(OUT / f"{stem}{suffix}.json", {"level": level, "window": win, "dates": dates, "fields": keep,
                   "players": [{**pl, "series": {win: {f: pl["series"][win][f] for f in keep}}}
                               for pl in players if win in pl["series"]]})
    print(f"{stem}.json(+_last30) {len(players):>7,} players x {len(dates)} dates")
    return players


static_p = {lvl: {int(r["id"]): {"throws": r["throws"]} for r in json.loads((OUT / "stuff_pitchers.json").read_text(encoding="utf-8"))
                  if r["level"] == lvl} for lvl in ["MLB", "AAA"]}
for r in json.loads((OUT / "aaa_pitchers.json").read_text(encoding="utf-8")):
    static_p["AAA"].setdefault(int(r["player_id"]), {}).update({"age": r["age"], "throws": r.get("throws"), "fb_velo_delta": r["fb_velo_delta"]})
static_h = {int(r["player_id"]): {"age": r["age"], "position": r.get("position"), "chase_delta": r["chase_delta"]}
            for r in json.loads((OUT / "aaa_hitters.json").read_text(encoding="utf-8"))}

timelines: dict[int, dict] = {}  # id -> level -> {...}, reused for the player files
for lvl in ["MLB", "AAA"]:
    fields = P_FIELDS | (P_PCTL if lvl == "AAA" else {})
    dates, ser = columnar(snap_p[snap_p["level"] == lvl], ["pitcher_id"], fields)
    # The leaderboard reads the grades file (both levels); the Triple-A explorer reads the pitchers file,
    # which leaves the grades out to stay small (it is the largest page; see tests/test_perf.py).
    base = ["team", "role", "games", "starts", "pitches", "bf", "qualified"]
    players = write_league(f"snapshots_grades_{lvl.lower()}", lvl, dates, fields, ser, static_p[lvl], base + GRADES + ["rv_plus"])
    if lvl == "AAA":
        write_league("snapshots_pitchers_aaa", lvl, dates, fields, ser, static_p[lvl], base + [
            "k_pct", "bb_pct", "csw_pct", "whiff_pct", "hard_hit_pct", "xwoba", "fb_velo",
            "k_pct_pctl", "bb_pct_pctl", "csw_pct_pctl", "whiff_pct_pctl", "hard_hit_pct_pctl", "xwoba_pctl"])
    for pl in players:
        timelines.setdefault(pl["id"], {}).setdefault(lvl, {"dates": dates})["pitcher"] = pl["series"]

    pdates, pser = columnar(snap_pt[snap_pt["level"] == lvl], ["pitcher_id", "pitch_type"], PT_FIELDS, by_window=False)
    rows = [{"id": int(k[0]), "pt": k[1], "group": group_of.get(k), "name": names.get(k[0], str(k[0])), "series": v}
            for k, v in sorted(pser.items())]
    write_json(OUT / f"snapshots_pitches_{lvl.lower()}.json", {"level": lvl, "dates": pdates, "fields": PT_KEEP, "pitches": [
        {**r, "series": {f: r["series"][f] for f in PT_KEEP}} for r in rows]})
    print(f"snapshots_pitches_{lvl.lower()}.json  {len(rows):>7,} pitch types")
    for r in rows:
        timelines.setdefault(r["id"], {}).setdefault(lvl, {"dates": pdates}).setdefault("pitch_types", []).append(
            {"pt": r["pt"], "group": r["group"], "series": r["series"]})

hfields = H_FIELDS | H_PCTL
dates, ser = columnar(snap_h, ["batter_id"], hfields)
players = write_league("snapshots_hitters_aaa", "AAA", dates, hfields, ser, static_h,
                       ["team", "pa", "qualified"] + [c for m in ["xwoba", "ev90", "hard_hit_pct", "chase_pct", "zone_contact_pct",
                                                                  "k_pct", "bb_pct"] for c in (m, m + "_pctl")])
for pl in players:
    timelines.setdefault(pl["id"], {}).setdefault("AAA", {"dates": dates})["hitter"] = pl["series"]

# Per-player files. Season-level rows are read back from the league JSON just written, so a player
# page and the league tools show identical values.
def by_id(file: str, key: str) -> dict:
    out: dict = {}
    for r in json.loads((OUT / file).read_text(encoding="utf-8")):
        out.setdefault(r[key], []).append(r)
    return out


season_rows = {f: by_id(f"{f}.json", k) for f, k in [("aaa_hitters", "player_id"), ("aaa_pitchers", "player_id"),
                                                       ("aaa_flags", "player_id"), ("stuff_pitchers", "id"),
                                                       ("stuff_pitches", "id")]}
for f, key, recs in [("aaa_arsenal", "player_id", ars_records), ("aaa_fb_velo_by_game", "id", fb_records)]:
    season_rows[f] = {}
    for r in recs:
        season_rows[f].setdefault(r[key], []).append(r)
PLAYERS = OUT / "players"
if PLAYERS.exists():
    for old in PLAYERS.glob("*.json"):
        old.unlink()
ids = sorted(set(season_rows["aaa_hitters"]) | set(season_rows["aaa_pitchers"]) | set(season_rows["stuff_pitchers"]))
for pid in ids:
    one = lambda f, pid=pid: (season_rows[f].get(pid) or [None])[0]  # noqa: E731
    write_json(PLAYERS / f"{pid}.json", {
        "id": pid, "name": names.get(pid, str(pid)), "season": SEASON,
        "hitter_aaa": one("aaa_hitters"), "pitcher_aaa": one("aaa_pitchers"),
        "stuff": season_rows["stuff_pitchers"].get(pid, []), "pitches": season_rows["stuff_pitches"].get(pid, []),
        "arsenal_aaa": season_rows["aaa_arsenal"].get(pid, []), "fb_games": season_rows["aaa_fb_velo_by_game"].get(pid, []),
        "flags": season_rows["aaa_flags"].get(pid, []), "timeline": timelines.get(pid, {}),
    })
write_json(OUT / "players_index.json", ids)
sizes = sorted(f.stat().st_size for f in PLAYERS.glob("*.json"))
print(f"players/*.json {len(sizes):>14,} files, median {sizes[len(sizes) // 2] / 1024:.0f} KB, "
      f"max {sizes[-1] / 1024:.0f} KB, total {sum(sizes) / 1e6:.1f} MB")

metrics, robust, locm = read_json(PQ / "metrics.json"), read_json(PQ / "robustness.json"), read_json(PQ / "metrics_location.json")
shape_res, tw = read_json(SS / "results.json"), read_json(TW / "results.json")
strip = lambda d: {k: v for k, v in d.items() if k != "provenance"}  # noqa: E731  (provenance stays in the repo, not on the site)
as_of = str(con.sql("select max(last_game) from fct_hitter_report").fetchone()[0])
meta = {
    "season": SEASON,
    "as_of": as_of,
    "trend_days": 14,
    "min_pa_qualified": 150,
    "min_bf_qualified": 150,
    "role_rules": ROLE_RULES,
    "qualifiers": QUALIFIERS,
    "stuff_metrics": strip(metrics),
    "stuff_robustness": strip(robust),
    "location_metrics": locm,
    "shape_results": strip(shape_res),
    "rule5": {"snapshot_date": r5res["snapshot_date"], "statuses": R5_STATUSES, **section("baseball", "rule5")},
    # Snapshot calendars, for the date slider and for prorating pitch minimums at earlier dates.
    "snapshots": {lvl: {"dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in g["snapshot_date"]], "team_games": g["team_games"].tolist(),
                        "team_games_last": g["team_games_last"].tolist(), "full_season_games": DBT_VARS["full_season_games"][lvl]}
                  for lvl, g in con.sql("select * from int_snapshot_dates order by level, snapshot_date").df().groupby("level")},
    "last_days": DBT_VARS["last_days"],
    "scale_reference": read_json(PQ / "scale_reference.json"),
}
write_json(OUT / "meta.json", meta, indent=1)

# ---- Two-week signal --------------------------------------------------------
write_json(OUT / "two_week_results.json", strip(tw))
for side, keys in [("hitter", ["chase_pct", "whiff_pct", "hard_hit_pct", "xwoba"]),
                   ("pitcher", ["fb_velo", "whiff_pct", "bb_pct", "xwoba"])]:
    d = pd.read_csv(TW / f"{side}_pairs.csv")  # already sorted by two_week_signal/analysis.py
    keep = [c for k in keys for c in (k, f"{k}_next")]
    write_records(OUT / f"two_week_{side}_pairs.json", d[keep], required=keep, min_rows=500, digits={c: 4 for c in keep})

# ---- Prose numbers ----------------------------------------------------------
# Every number an article states comes from here, formatted the way the prose prints it.
# Never hand-type a number into an article; add it here instead.
MINUS = "−"


def r2(x: float) -> str:
    """Correlation-style number, 2 decimals, typographic minus."""
    return f"{x:.2f}".replace("-", MINUS)


def signed2(x: float) -> str:
    return ("+" if x >= 0 else MINUS) + f"{abs(x):.2f}"


def pct0(x: float) -> str:
    return f"{100 * x:.0f}"


def span(years: list[int]) -> str:
    return f"{years[0]}–{years[-1]}" if len(years) > 1 else str(years[0])


def millions(n: int) -> str:
    return f"{n / 1e6:.1f}M"


class StaleProse(AssertionError):
    """The data no longer supports a sentence in an article. Rewrite the sentence."""


N = {
    "season": SEASON, "season_prev": PREV, "season_prev2": PREV - 1,
    "mlb_span": span(CFG.mlb_seasons), "aaa_span": span(CFG.aaa_seasons),
    "train_span": span(CFG.train_seasons), "as_of": as_of,
}

# Stuff+ article. The headline YoY test is the fully out-of-sample one (robustness.py).
N |= {
    "stuff_yoy_r_stuff": r2(robust["r_stuff_prev_to_rv_cur"]),
    "stuff_yoy_r_results": r2(robust["r_rv_prev_to_rv_cur"]),
    "stuff_yoy_n": f"{robust['n']:,}",
    "stuff_yoy_train_season": robust["train_season"],
    "stuff_gap_lo": signed2(robust["gap_ci90"][0]),
    "stuff_gap_hi": signed2(robust["gap_ci90"][1]),
    "stuff_min_pitches": metrics["min_pitches"],
    "stuff_aaa_min_pitches": metrics["aaa_min_pitches"],
    "stuff_aaa_n": f"{metrics['aaa_to_mlb_n']:,}",
    "stuff_aaa_r_stuff": r2(metrics["aaa_to_mlb_r_stuff"]),
    "stuff_aaa_r_results": r2(metrics["aaa_to_mlb_r_actual"]),
    "stuff_aaa_mean": f"{metrics['aaa_mean_stuff_plus_qualified']:.0f}",
    "stuff_holdout_r2": f"{metrics['holdout_pitch_r2']:.4f}",
    "stuff_split_half_stuff": r2(metrics["split_half_r_stuff"]),
    "stuff_split_half_rv": r2(metrics["split_half_r_actual_rv"]),
    "stuff_pitches_mlb": millions(metrics["n_pitches_mlb"]),
    "stuff_pitches_aaa": millions(metrics["n_pitches_aaa"]),
}
# The finding says Stuff+ predicted next season at least as well as results did; the certainty sentence
# below says whether the edge is distinguishable from zero.
if not robust["r_stuff_prev_to_rv_cur"] > robust["r_rv_prev_to_rv_cur"]:
    raise StaleProse("Stuff+ article says last season's Stuff+ predicted this season's run value at least as well as results.")
# ...and that its clearer advantage is stability (split-half reliability).
if not metrics["split_half_r_stuff"] > metrics["split_half_r_actual_rv"] + 0.3:
    raise StaleProse("Stuff+ article says Stuff+ settles almost immediately while a season of results is mostly noise.")
# How sure: worded from the interval itself. Its low end sits near zero, and tiny platform differences in
# the model (Windows vs the Linux runner) can move it across zero, so the sentence follows the number.
lo = robust["gap_ci90"][0]
N["stuff_gap_sentence"] = (
    "The whole interval is above zero, so Stuff+ is reliably the better predictor, though the margin is modest."
    if lo >= 0.05 else
    "The interval sits just above zero, so Stuff+ is better, but the true margin could be close to nothing."
    if lo > 0 else
    "The interval includes zero, so this sample can't rule out that the two predict equally well: Stuff+ comes "
    "out ahead, but not by a margin the data can confirm.")
# The finding box says how sure, in one line, from the same interval.
N["stuff_finding_certainty"] = (
    "Stuff+ came out ahead, and the edge held up across resamples, though it's modest."
    if lo > 0 else
    f"Stuff+ came out ahead in this sample, but not by enough to call it the better forecaster: across resamples "
    f"of these pitchers the gap ranged from {signed2(robust['gap_ci90'][0])} to {signed2(robust['gap_ci90'][1])}.")
if not 0 < metrics["aaa_mean_stuff_plus_qualified"] < 100:
    raise StaleProse("Stuff+ article says the average qualified Triple-A pitcher grades below an MLB average of 100.")
FEATURE_WORDS = {"velo": "velocity", "ivb": "induced vertical break", "hb": "horizontal break", "spin": "spin rate",
                 "spin_axis": "spin axis", "extension": "extension", "rel_z": "release height", "rel_x": "release side",
                 "arm_angle": "arm angle", "velo_diff": "the velocity gap from the pitcher's fastball",
                 "ivb_diff": "the IVB gap from the pitcher's fastball", "hb_diff": "the horizontal-break gap from the pitcher's fastball"}
rank = imp.groupby("feature")["gain"].mean().sort_values(ascending=False)
top = [FEATURE_WORDS[f] for f in rank.index[:4]]
N["stuff_top_features"] = f"{top[0]}, {top[1]}, {top[2]}, and {top[3]}"
N["stuff_least_feature"] = FEATURE_WORDS[rank.index[-1]]
by_group = imp.pivot(index="feature", columns="group", values="gain")
diffs = by_group.loc[["velo_diff", "ivb_diff", "hb_diff"]].sum()
# How much the fastball-gap features carry in each non-fastball model, worded from the shares. A gap of
# under 2 points counts as "about the same": the breaking-ball share sits within a fraction of a point of
# the fastball model's and flips between the Windows and Linux builds.
GAP_MARGIN = 0.02
pct = lambda x: f"{100 * x:.0f}%"  # noqa: E731
parts = [f"for {name}, {'more' if diffs[g] - diffs['fastball'] >= GAP_MARGIN else 'less' if diffs['fastball'] - diffs[g] >= GAP_MARGIN else 'about the same'} ({pct(diffs[g])})"
         for g, name in [("offspeed", "offspeed pitches"), ("breaking", "breaking balls")]]
N["stuff_gap_mix"] = (f"The gaps from the fastball (velocity, ride, and run) carry {pct(diffs['fastball'])} of the "
                      f"fastball model's gain; {parts[0]}; {parts[1]}.")

# Stuff-or-command article (Location+ and Pitching+). Validation from robustness.py: the same fully
# out-of-sample year-over-year test as the Stuff+ article, run for every grade with one bootstrap.
YO, SH, AM = locm["yoy"], locm["split_half"], locm["aaa_to_mlb"]
KIND_WORDS = {"stuff": "Stuff+", "loc": "Location+", "pitching": "Pitching+", "results": "results"}


def ci_words(lo: float, hi: float) -> str:
    return f"{r2(lo)} to {r2(hi)}"


N |= {"loc_yoy_n": f"{YO['n']:,}", "loc_aaa_n": f"{AM['n']:,}", "loc_boot": f"{YO['bootstrap_samples']:,}",
      "loc_train_season": locm["seasons"]["train"],
      "loc_zone_top": f"{100 * locm['zone_share_of_height']['top']:.1f}%",
      "loc_zone_bottom": f"{100 * locm['zone_share_of_height']['bottom']:.1f}%",
      "loc_mean_mlb_cur": f"{metrics['mlb_mean_loc_plus_qualified']:.0f}",
      "loc_mean_aaa_cur": f"{metrics['aaa_mean_loc_plus_qualified']:.0f}",
      "pitching_mean_aaa_cur": f"{metrics['aaa_mean_pitching_plus_qualified']:.0f}",
      "loc_holdout_r2": f"{locm['holdout_pitch_r2']['loc']:.3f}",
      "loc_holdout_r2_pitching": f"{locm['holdout_pitch_r2']['pitching']:.3f}",
      "loc_holdout_r2_stuff": f"{locm['holdout_pitch_r2']['stuff']:.4f}"}
for k in ["stuff", "loc", "pitching", "results"]:
    v = YO["r_next_rv"][k]
    N |= {f"loc_split_{k}": r2(SH[k]), f"loc_next_{k}": r2(v["r"]), f"loc_next_{k}_ci": ci_words(*v["ci90"]),
          f"loc_self_{k}": r2(YO["r_self"][k]), f"loc_aaa_{k}": r2(AM[k])}
for g, v in YO["gaps"].items():
    N |= {f"loc_gap_{g}": signed2(v["est"]), f"loc_gap_{g}_ci": f"{signed2(v['ci90'][0])} to {signed2(v['ci90'][1])}"}

# Directional claims the article makes regardless of wording.
if not SH["results"] < SH["loc"] < SH["stuff"]:
    raise StaleProse("Stuff-or-command article: Location+ is more reliable within a season than results, less than Stuff+.")
if not YO["r_self"]["loc"] > YO["r_self"]["results"]:
    raise StaleProse("Stuff-or-command article: Location+ carries from one season to the next better than results do.")
if not YO["gaps"]["loc_minus_stuff"]["ci90"][1] < 0:
    raise StaleProse("Stuff-or-command article: Stuff+ predicts next season reliably better than Location+.")

# The rest is worded from the intervals, so a refresh (or the Linux vs Windows model builds) can't strand it.
lr, (llo, lhi) = YO["r_next_rv"]["loc"]["r"], YO["r_next_rv"]["loc"]["ci90"]
N["loc_next_sentence"] = (
    f"Location+ on its own predicted next season's run value at r = {r2(lr)}, with an interval ({ci_words(llo, lhi)}) "
    "that runs across zero: no detectable signal." if llo <= 0 <= lhi else
    f"Location+ on its own predicted next season's run value at r = {r2(lr)} ({ci_words(llo, lhi)}), "
    + ("a real but small signal." if llo > 0 else "in the wrong direction."))
pg = YO["gaps"]["pitching_minus_stuff"]
N["pitching_vs_stuff_sentence"] = (
    f"Pitching+, which adds location and count to the same stuff inputs, predicted it at r = {N['loc_next_pitching']}, "
    f"against {N['loc_next_stuff']} for Stuff+ alone. The difference ({signed2(pg['est'])}, interval "
    f"{N['loc_gap_pitching_minus_stuff_ci']}) is " + (
        "indistinguishable from zero: adding location did not improve the forecast." if pg["ci90"][0] <= 0 <= pg["ci90"][1] else
        "reliably above zero, so location adds a little to the forecast." if pg["ci90"][0] > 0 else
        "reliably below zero: adding location made the forecast worse."))
# The finding box, in one sentence about the forecast, from the same two intervals.
N["loc_finding_forecast"] = (
    "but it did almost nothing to predict next season's run value, and adding it to Stuff+ didn't improve the forecast"
    if llo <= 0 <= lhi and pg["ci90"][0] <= 0 <= pg["ci90"][1] else
    "and adding it to Stuff+ improved the forecast" if pg["ci90"][0] > 0 else
    "but adding it to Stuff+ didn't improve the forecast")
N["loc_sowhat_forecast"] = (
    f"**Forecast with Stuff+.** Adding location (Pitching+) predicted next season no better ({N['loc_next_pitching']} "
    f"vs. {N['loc_next_stuff']}), so the bigger model adds nothing to a projection." if pg["ci90"][0] <= 0 <= pg["ci90"][1] else
    f"**Forecast with Pitching+.** Adding location predicted next season reliably better ({N['loc_next_pitching']} "
    f"vs. {N['loc_next_stuff']} for Stuff+ alone)." if pg["ci90"][0] > 0 else
    f"**Forecast with Stuff+.** Adding location (Pitching+) made the forecast worse ({N['loc_next_pitching']} "
    f"vs. {N['loc_next_stuff']}).")
pr_ = YO["gaps"]["pitching_minus_results"]
N["pitching_vs_results_sentence"] = (
    f"Pitching+ still beat last season's results ({N['loc_next_results']}), with an interval for the gap of "
    f"{N['loc_gap_pitching_minus_results_ci']}" + (" that stays above zero." if pr_["ci90"][0] > 0 else ", which includes zero."))
am = AM["loc"]
N["loc_aaa_sentence"] = (
    f"Triple-A Location+ said nothing about the same pitchers' MLB run value the next season (r = {r2(am)}, "
    f"n = {N['loc_aaa_n']})" if abs(am) < 0.1 else
    f"Triple-A Location+ tracked the same pitchers' MLB run value the next season at r = {r2(am)} (n = {N['loc_aaa_n']})")
if not AM["stuff"] > AM["loc"]:
    raise StaleProse("Stuff-or-command article: Triple-A Stuff+ carried to MLB better than Triple-A Location+.")

# Same-season picture among qualified MLB pitchers this season: who leads, and how the grades relate.
qm = p[(p["level"] == "MLB") & (p["pitches"] >= metrics["min_pitches"])]
cs = qm[["stuff_plus", "loc_plus", "pitching_plus", "rv_plus"]].corr()
N |= {"loc_q_mlb_n": f"{len(qm):,}", "loc_r_stuff_loc": r2(cs.loc["stuff_plus", "loc_plus"]),
      "loc_r_loc_rv": r2(cs.loc["loc_plus", "rv_plus"]), "loc_r_stuff_rv": r2(cs.loc["stuff_plus", "rv_plus"]),
      "loc_r_pitching_rv": r2(cs.loc["pitching_plus", "rv_plus"]),
      "loc_top_mlb": "; ".join(f"{d['name']} ({d['team']}, {d['loc_plus']:.0f})"
                               for _, d in qm.sort_values("loc_plus", ascending=False).head(5).iterrows()),
      "pitching_top_mlb": "; ".join(f"{d['name']} ({d['team']}, {d['pitching_plus']:.0f})"
                                    for _, d in qm.sort_values("pitching_plus", ascending=False).head(5).iterrows())}
if not cs.loc["stuff_plus", "loc_plus"] < 0:
    raise StaleProse("Stuff-or-command article: among qualified MLB pitchers, better stuff goes with slightly worse location.")
if not cs.loc["loc_plus", "rv_plus"] < cs.loc["stuff_plus", "rv_plus"] < cs.loc["pitching_plus", "rv_plus"]:
    raise StaleProse("Stuff-or-command article: within a season, Pitching+ tracks results best, then Stuff+, then Location+.")
if not cs.loc["loc_plus", "rv_plus"] > 0:
    raise StaleProse("Stuff-or-command article: within a season, Location+ goes with better results.")
for k in ["loc", "pitching"]:
    if not metrics[f"aaa_mean_{k}_plus_qualified"] < min(100, metrics[f"mlb_mean_{k}_plus_qualified"]):
        raise StaleProse(f"Stuff-or-command article: qualified Triple-A pitchers average below big leaguers on {KIND_WORDS[k]}.")
if not SH["pitching"] > SH["results"]:
    raise StaleProse("Stuff-or-command article: every grade is more reliable within a season than results.")
# Triple-A paragraph: "stuff carried a little, location and results didn't".
if not (AM["stuff"] >= 0.1 and abs(AM["loc"]) < 0.1 and abs(AM["results"]) < 0.1):
    raise StaleProse("Stuff-or-command article: Triple-A Stuff+ carried to MLB a little; Location+ and results didn't.")

# Shape-changes article
M, V = shape_res["metrics"], shape_res["velo_by_pitch_class"]
move = [M[m][d]["persistence"] for m in ("ivb", "hb") for d in ("gain", "loss")]
drift = shape_res["baseline_offseason_drift_sd"]
move_drift = sorted([drift["ivb"], drift["hb"]])
N |= {
    "shape_n": f"{shape_res['n_pitch_types']:,}",
    "shape_pairs": " and ".join(p.replace("->", "→") for p in shape_res["pairs"]),
    "shape_move_lo": pct0(min(move)), "shape_move_hi": pct0(max(move)),
    "shape_velo_gain": pct0(M["velo"]["gain"]["persistence"]),
    "shape_velo_loss": pct0(M["velo"]["loss"]["persistence"]),
    "shape_fb_loss": pct0(V["fastball_loss"]["persistence"]),
    "shape_fb_loss_n": V["fastball_loss"]["n"],
    "shape_nonfb_loss": pct0(V["non_fastball_loss"]["persistence"]),
    "shape_league_velo": f"{sum(shape_res['league_mean_adjustments']['velo_chg'].values()) / len(shape_res['pairs']):.1f}",
    "shape_drift_velo": f"{drift['velo']:.1f}",
    "shape_drift_move": f"{move_drift[0]:.1f}–{move_drift[1]:.1f}",
    "shape_min_pitches": shape_res["min_pitches_per_half"],
    "shape_bootstrap": f"{shape_res['bootstrap_samples']:,}",
}
if not V["non_fastball_loss"]["persistence"] > V["fastball_loss"]["persistence"]:
    raise StaleProse("Shape article says velocity drops on non-fastballs hold more than fastball drops.")
if not M["velo"]["loss"]["persistence"] < min(move):
    raise StaleProse("Shape article says lost velocity holds less than movement changes.")

# The two worked examples are pinned to a historical season pair. If that pair ages out of
# the config window, this fails and the paragraph needs new examples.
EXAMPLES = [("ex1", "Cole Ragans", "SL", "2024->2025", "held"), ("ex2", "Ryne Nelson", "SL", "2024->2025", "reverted")]
for key, name, pt, pair, story in EXAMPLES:
    row = sc[(sc["name"] == name) & (sc["pitch_type"] == pt) & (sc["pair"] == pair)]
    if len(row) != 1:
        raise StaleProse(f"Shape article example {name} {pt} {pair} is no longer in the data; pick a new example.")
    r = row.iloc[0]
    held = abs(r["velo_next"] - r["velo2"]) < abs(r["velo_next"] - r["velo1"])
    if held != (story == "held"):
        raise StaleProse(f"Shape article says {name}'s {pt} change {story}; the data disagrees.")
    N |= {f"shape_{key}_h1": f"{r['velo1']:.1f}", f"shape_{key}_h2": f"{r['velo2']:.1f}", f"shape_{key}_next": f"{r['velo_next']:.1f}"}

# Two-week article
H, P_ = tw["hitters"], tw["pitchers"]
pit_pairs = sorted({v["n_pairs"] for v in P_.values()})
N |= {
    "tw_seasons": " and ".join(map(str, tw["seasons"])),
    "tw_block_days": tw["block_days"], "tw_min": tw["min_pa_or_bf"],
    "tw_hit_chase": r2(H["chase_pct"]["r"]), "tw_hit_whiff": r2(H["whiff_pct"]["r"]),
    "tw_hit_xwoba": r2(H["xwoba"]["r"]), "tw_hit_woba": r2(H["woba"]["r"]), "tw_hit_hardhit": r2(H["hard_hit_pct"]["r"]),
    "tw_pit_velo": r2(P_["fb_velo"]["r"]), "tw_pit_k": r2(P_["k_pct"]["r"]), "tw_pit_bb": r2(P_["bb_pct"]["r"]),
    "tw_pit_whiff": r2(P_["whiff_pct"]["r"]),
    "tw_hit_pairs": f"{H['chase_pct']['n_pairs']:,}",
    "tw_pit_pairs": f"{pit_pairs[0]:,}" if len(pit_pairs) == 1 else f"{pit_pairs[0]:,}–{pit_pairs[-1]:,}",
}
if not min(H["chase_pct"]["r"], H["whiff_pct"]["r"]) > H["xwoba"]["r"] > H["woba"]["r"]:
    raise StaleProse("Two-week article ranks chase/whiff above xwOBA above wOBA for hitters.")
if not H["hard_hit_pct"]["r"] > H["xwoba"]["r"]:
    raise StaleProse("Two-week article says hard-hit rate is more stable than xwOBA.")

# Pitcher roles and role-aware qualifiers (glossary)
if any(q["Swing"] != q["RP"] for q in QUALIFIERS.values()):
    raise StaleProse("Glossary says relievers and swingmen share a pitch minimum.")
N |= {
    "role_sp_min_share": f"{ROLE_RULES['role_sp_min_share']:.0%}",
    "role_sp_min_starts": ROLE_RULES["role_sp_min_starts"],
    "role_rp_max_share": f"{ROLE_RULES['role_rp_max_share']:.0%}",
    **{f"q_{lvl.lower()}_{r.lower()}": q for lvl, by_role in QUALIFIERS.items() for r, q in by_role.items()},
}

# Rule 5 board article
R5CAL = {k: date.fromisoformat(v) for k, v in section("baseball", "rule5").items()}
AP_MONTH = {1: "Jan.", 2: "Feb.", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "Aug.", 9: "Sept.",
            10: "Oct.", 11: "Nov.", 12: "Dec."}
nice = lambda d: f"{AP_MONTH[d.month]} {d.day}"  # noqa: E731  (AP style dates: Nov. 17, June 5)
snap = date.fromisoformat(r5res["snapshot_date"])
qp = r5p[r5p.apply(lambda d: d["pitches"] >= QUALIFIERS["AAA"][d["role"]], axis=1) & r5p["stuff_plus"].notna()]
top_p = qp.sort_values("stuff_plus", ascending=False).head(5)
qh = r5h[r5h["pa"] >= meta["min_pa_qualified"]].copy()
qh["composite"] = qh[[f"{c}_pctl" for c in R5_HIT]].mean(axis=1)
top_h = qh.sort_values("composite", ascending=False).head(5)
if len(top_p) < 5 or len(top_h) < 5:
    raise StaleProse("Rule 5 article names five pitchers and five hitters.")
val = r5res["validation"]
if val["agreement"] < 0.8:
    raise StaleProse("Rule 5 method box says the eligibility estimate met the 80% validation target.")
N |= {
    "r5_snapshot": nice(snap), "r5_deadline": nice(R5CAL["protection_deadline"]), "r5_draft": nice(R5CAL["draft_date"]),
    "r5_cba": nice(R5CAL["cba_expires"]),
    "r5_snapshot_after_deadline": "after" if snap > R5CAL["protection_deadline"] else "before",
    "r5_n_pitchers": f"{len(r5p):,}", "r5_n_hitters": f"{len(r5h):,}",
    "r5_n_qual_pitchers": f"{len(qp):,}", "r5_n_qual_hitters": f"{len(qh):,}",
    "r5_n_stuff_100": f"{int((qp['stuff_plus'] >= 100).sum()):,}",
    "r5_top_pitchers": "; ".join(f"{d['name']} ({d['org']}, {d['role']}, Stuff+ {d['stuff_plus']:.0f})" for _, d in top_p.iterrows()),
    "r5_top_hitters": "; ".join(f"{d['name']} ({d['org']}, {d['position']}, {d['pa']} PA)" for _, d in top_h.iterrows()),
    "r5_n_mlb_time": f"{int((r5p['mlb_g'] > 0).sum()):,}",
    "r5_n_likely": f"{int((r5p['status'] == 'Likely eligible').sum() + (r5h['status'] == 'Likely eligible').sum()):,}",
    "r5_n_unknown": f"{int((r5p['status'] == 'Unknown').sum() + (r5h['status'] == 'Unknown').sum()):,}",
    "r5_val_n": val["n"], "r5_val_agree": val["agree"], "r5_val_pct": f"{val['agreement']:.0%}",
    "r5_val_n_pos": val["n_listed_eligible"], "r5_val_n_neg": val["n_listed_not_yet"],
    "r5_n_free_agents": f"{r5res['status_counts'].get('Free agent', 0):,}",
    # Before the deadline the article says the board is preliminary; after it, that the pool is final.
    "r5_prelim_note": (
        f"*Built from {nice(snap)} rosters, after the {nice(R5CAL['protection_deadline'])} protection deadline, "
        "so the eligible pool is set.*") if snap > R5CAL["protection_deadline"] else (
        f"**Preliminary:** built from {nice(snap)} rosters, before the {nice(R5CAL['protection_deadline'])} "
        "protection deadline. Some of these players will be added to 40-man rosters by then."),
    "min_pa_qualified": meta["min_pa_qualified"],
}

# Frozen Stuff+ scale (prior-season qualified MLB pitchers), quoted in the Stuff+ method and leaderboard.
SCALE = read_json(PQ / "scale_reference.json")
N |= {
    "scale_ref_season": SCALE["reference_season"], "scale_ref_n": f"{SCALE['stuff']['n']:,}",
    "scale_ref_min": SCALE["min_pitches"],
    "scale_ref_mean": f"{SCALE['stuff']['mean']:+.3f}".replace("-", MINUS),  # runs per 100 pitches
    "scale_ref_sd": f"{SCALE['stuff']['sd']:.3f}",
    "stuff_mlb_mean_cur": f"{metrics['mlb_mean_stuff_plus_qualified']:.1f}",
}
# Qualified-pool proration at the final Triple-A date (glossary).
aaa_cal = meta["snapshots"]["AAA"]
N |= {"aaa_final_team_games": f"{aaa_cal['team_games'][-1]:.0f}", "aaa_full_season_games": aaa_cal["full_season_games"],
      "aaa_final_qual": f"{max(DBT_VARS['min_qualified_floor'], round(meta['min_bf_qualified'] * min(1, aaa_cal['team_games'][-1] / aaa_cal['full_season_games']))):.0f}"}

write_numbers("baseball", N, source="baseball/export_site_data.py")
