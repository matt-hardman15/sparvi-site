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
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from sparvi.config import ROOT, baseball
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
cols = ["id", "name", "team", "level", "throws", "role", "g", "gs", "pitches", "stuff_plus", "rv_plus"]
write_records(OUT / "stuff_pitchers.json", p[cols].sort_values(["level", "id"]),
              required=cols, not_null=["id", "name", "level", "role"], min_rows=500, digits={"stuff_plus": 1, "rv_plus": 1})

t = pd.concat([pd.read_csv(PQ / f"pitch_type_{lvl}_{SEASON}.csv").query("n >= 30").assign(level=lvl.upper())
               for lvl in ["mlb", "aaa"]])
t["name"] = t["player_name"].map(first_last)
t = t.rename(columns={"pitcher": "id", "pitch_type": "pt"}).merge(roles[["id", "level", "role"]], on=["id", "level"], how="left")
cols = ["id", "name", "team", "level", "role", "pt", "group", "n", "velo", "ivb", "hb", "stuff_plus"]
write_records(OUT / "stuff_pitches.json", t[cols].sort_values(["level", "id", "pt"]),
              required=cols, not_null=["id", "name", "pt", "role"], min_rows=1000,
              digits={"velo": 1, "ivb": 1, "hb": 1, "stuff_plus": 1})

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
write_records(OUT / "aaa_arsenal.json", ars, min_rows=500, not_null=["player_id", "pitch_type"], digits={c: 3 for c in floats(ars)},
              required=["player_id", "pitch_type", "pitches", "usage_pct", "velo", "ivb", "hb", "whiff_pct", "stuff_plus",
                        "recent_pitches", "velo_delta", "ivb_delta"])

g = con.sql("""
    select pitcher_id as id, strftime(game_date, '%Y-%m-%d') as d, pitch_type as pt, pitches as n,
           round(velo, 1) as velo
    from int_pitcher_games
    qualify row_number() over (partition by pitcher_id, game_pk order by pitches desc, pitch_type) = 1
    order by id, d
""").df()
write_records(OUT / "aaa_fb_velo_by_game.json", g, required=["id", "d", "pt", "n", "velo"], not_null=["id", "d"], min_rows=1000)

fl = con.sql("select * exclude (as_of_date) from rpt_flags order by player_id, role").df()
write_records(OUT / "aaa_flags.json", fl, min_rows=0, not_null=["player_id"],
              required=["player_id", "name", "team", "role", "flag_discipline", "flag_contact"])

metrics, robust = read_json(PQ / "metrics.json"), read_json(PQ / "robustness.json")
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
    "shape_results": strip(shape_res),
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
# The finding states the point estimate: Stuff+ predicted next season better than results did.
if not robust["r_stuff_prev_to_rv_cur"] > robust["r_rv_prev_to_rv_cur"]:
    raise StaleProse("Stuff+ article says last season's Stuff+ predicted this season's run value better than results.")
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
if not (diffs["breaking"] > diffs["fastball"] and diffs["offspeed"] > diffs["fastball"]):
    raise StaleProse("Stuff+ article says fastball-gap features matter more for breaking and offspeed models.")

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

write_numbers("baseball", N, source="baseball/export_site_data.py")
