"""Development folds for the Triple-A translation (M5-EB): does shrinking a Triple-A rate by its reliability, adding
age, or sizing the band by sample beat the published line (one weighted line per metric, one band for everyone)?

DEVELOPMENT ONLY. A fold is an MLB outcome season t before the current one: the pairs to predict are last season's
Triple-A line -> season t's MLB line ("across") and a player's Triple-A line before his first MLB game of season t ->
his MLB line that season ("within"); everything is fit on pairs whose MLB season is before t. Nothing here reads an
MLB line from the current season, so whatever is chosen here can still be tested on it. Needs the history seasons
(`python ingest/baseball/pull.py --history`).

Lines are rebuilt here from the raw pitches with the pipeline's definitions (stg_pitches flags and
`plate_discipline_rates`), because the dbt marts hold only the site's seasons; `check_against_mart` compares them
with fct_translation_lines where both exist. The pitch-quality grades (Stuff+, Location+, Pitching+, RV+) are left
out: grading the history seasons needs models trained before them.

Variants, per metric (a Triple-A rate x from n opportunities -> the same rate in MLB):
    raw      the published line: MLB on x, weighted by the smaller sample, pairs with 150+ PA/BF in Triple-A
    eb       x shrunk toward the Triple-A mean by its reliability n / (n + k) first; k from split halves (odd and even
             games) of Triple-A seasons before t
    eb_wide  eb, fit on pairs down to 50 PA/BF in Triple-A
    eb_age   eb_wide plus the player's age
Scored on pairs with 100+ PA/BF in MLB by root mean squared error, on the published pool (150+ in Triple-A) and on the
small-sample pool (50-149) the published line doesn't cover. 90% intervals: bootstrap over players.

Bands (80%): the published one (the same residual quantiles for everyone) against one whose width grows as the
Triple-A sample shrinks; coverage on the fold's pairs, by sample size.

Combined model: MLB xwOBA from every shrunk metric and age (ridge, alpha by grouped CV inside the training pairs)
against xwOBA's own line.

    python baseball/aaa_translation/develop.py            # about two minutes

Writes output/development.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from sparvi.config import RAW, ROOT, baseball

sys.path.insert(0, str(ROOT))
from backtest.asof import Loader  # noqa: E402

CFG = baseball()
HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
SEED, BOOT = 20261125, 2000
MIN_AAA, MIN_AAA_WIDE, MIN_MLB = 150, 50, 100
MIN_HALF = 20                   # opportunities in each half for a player to count toward the noise estimate
Z80 = 1.2815515655446004
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]

# metric -> the column counting its opportunities
METRICS = {
    "hitter": {"chase_pct": "n_out", "zone_swing_pct": "n_in", "zone_contact_pct": "n_in_swing", "whiff_pct": "n_swing",
               "k_pct": "pa", "bb_pct": "pa", "hard_hit_pct": "n_bip", "ev90": "n_ev", "xwoba": "pa", "woba": "pa"},
    "pitcher": {"k_pct": "pa", "bb_pct": "pa", "csw_pct": "pitches", "whiff_pct": "n_swing", "chase_pct": "n_out",
                "xwoba": "pa"},
}
VARIANTS = ["raw", "eb", "eb_wide", "eb_age"]
COLS = ["game_pk", "game_date", "game_year", "game_type", "pitcher", "batter", "at_bat_number", "pitch_number", "zone",
        "description", "events", "launch_speed", "estimated_woba_using_speedangle", "woba_value", "woba_denom",
        "plate_x", "plate_z"]


def seasons() -> dict[str, list[int]]:
    """Seasons on file at each level, the current season left out (its MLB lines are the test's outcomes)."""
    out = {}
    for lvl, ss in (("aaa", CFG.aaa_history_seasons + CFG.aaa_seasons), ("mlb", CFG.history_seasons + CFG.mlb_seasons)):
        out[lvl] = [s for s in ss if s < CFG.season and (RAW / f"statcast_{lvl}_{s}.parquet").exists()]
    return out


def build_lines(loader: Loader) -> pd.DataFrame:
    """One row per side x player x season x level x segment ('full'; Triple-A also 'pre_debut', 'half0', 'half1')."""
    v = yaml.safe_load((HERE.parent / "pd_pipeline" / "dbt_project.yml").read_text(encoding="utf-8"))["vars"]
    S = seasons()
    con = duckdb.connect(config={"threads": 1})
    frames = []
    for lvl, ss in S.items():
        for s in ss:
            d = loader.load(f"statcast_{lvl}_{s}", f"{s + 1}-03-01", COLS)
            frames.append(d.assign(level=lvl.upper(), game_date=pd.to_datetime(d["game_date"]).dt.date))
    raw = pd.concat(frames, ignore_index=True)  # noqa: F841  (read by DuckDB)
    h = loader.load("batter_heights", f"{CFG.season}-03-01")
    heights = h.assign(height_ft=h["height_in"] / 12.0)[["batter", "height_ft"]]  # noqa: F841
    lines = con.sql(f"""
        with p as (
            select level, game_pk, cast(game_date as date) as game_date, game_year as season, pitcher, batter,
                   launch_speed, estimated_woba_using_speedangle as xwoba_bip, woba_value, woba_denom,
                   case when plate_x is null or plate_z is null then null
                        when h.height_ft is null then zone between 1 and 9
                        else abs(plate_x) <= {v['zone_half_width_ft']}
                         and plate_z between {v['zone_bot_share']} * h.height_ft - {v['ball_radius_ft']}
                                         and {v['zone_top_share']} * h.height_ft + {v['ball_radius_ft']} end as in_zone,
                   description in ('foul', 'foul_tip', 'hit_into_play', 'swinging_strike', 'swinging_strike_blocked') as is_swing,
                   description in ('swinging_strike', 'swinging_strike_blocked', 'foul_tip') as is_whiff,
                   description in ('called_strike', 'swinging_strike', 'swinging_strike_blocked', 'foul_tip') as is_csw,
                   description = 'hit_into_play' as is_bip,
                   events is not null as ends_pa,
                   events in ('strikeout', 'strikeout_double_play') as is_k,
                   events in ('walk', 'intent_walk') as is_bb,
                   description = 'hit_into_play' and launch_speed >= 95 as is_hard_hit
            from raw left join heights h using (batter)
            where game_type = 'R' and description not in ('pitchout', 'intent_ball', 'automatic_ball', 'automatic_strike')
        ),
        sided as (
            select 'hitter' as side, batter as player_id, p.* from p
            union all
            select 'pitcher', pitcher, p.* from p
        ),
        debuts as (
            select side, season, player_id, min(game_date) as first_mlb_game from sided where level = 'MLB' group by all
        ),
        tagged as (
            select s.*, 'full' as segment from sided s
            union all
            select s.*, 'half' || cast(game_pk % 2 as varchar) from sided s where level = 'AAA'
            union all
            select s.*, 'pre_debut' from sided s join debuts d using (side, season, player_id)
            where s.level = 'AAA' and s.game_date < d.first_mlb_game
        )
        select side, player_id, season, level, segment,
               count(*) as pitches,
               count(*) filter (where ends_pa) as pa,
               count(*) filter (where not in_zone) as n_out,
               count(*) filter (where in_zone) as n_in,
               count(*) filter (where is_swing and in_zone) as n_in_swing,
               count(*) filter (where is_swing) as n_swing,
               count(*) filter (where is_bip) as n_bip,
               count(launch_speed) filter (where is_bip) as n_ev,
               avg(is_swing::int) filter (where not in_zone) as chase_pct,
               avg(is_swing::int) filter (where in_zone) as zone_swing_pct,
               1 - avg(is_whiff::int) filter (where is_swing and in_zone) as zone_contact_pct,
               avg(is_whiff::int) filter (where is_swing) as whiff_pct,
               avg(is_csw::int) as csw_pct,
               avg(is_k::int) filter (where ends_pa) as k_pct,
               avg(is_bb::int) filter (where ends_pa) as bb_pct,
               avg(is_hard_hit::int) filter (where is_bip) as hard_hit_pct,
               quantile_cont(launch_speed, 0.9) filter (where is_bip) as ev90,
               sum(case when is_bip and xwoba_bip is not null then xwoba_bip else woba_value end) filter (where ends_pa)
                   / nullif(sum(woba_denom) filter (where ends_pa), 0) as xwoba,
               sum(woba_value) filter (where ends_pa) / nullif(sum(woba_denom) filter (where ends_pa), 0) as woba
        from tagged group by all order by all
    """).df()
    con.close()
    return lines


def ages() -> pd.Series:
    """player_id -> birth date. A fixed attribute, so every roster file on the machine is used."""
    files = sorted(RAW.glob("milb_players_*.parquet"))
    b = pd.concat([pd.read_parquet(f, columns=["player_id", "birth_date"]) for f in files]).dropna().drop_duplicates("player_id")
    return pd.to_datetime(b.set_index("player_id")["birth_date"])


def pairs(lines: pd.DataFrame, side: str, born: pd.Series) -> pd.DataFrame:
    """Every Triple-A -> MLB pair on file with its outcome season; samples as low as MIN_AAA_WIDE."""
    L = lines[lines["side"].eq(side)].drop(columns="side")
    aaa = L[L["level"].eq("AAA") & L["pa"].ge(MIN_AAA_WIDE)]
    mlb = L[L["level"].eq("MLB") & L["segment"].eq("full") & L["pa"].ge(MIN_MLB)]
    full = aaa[aaa["segment"].eq("full")]
    across = full.assign(season=full["season"] + 1).merge(mlb, on=["player_id", "season"], suffixes=("_aaa", "_mlb")).assign(cohort="across")
    within = aaa[aaa["segment"].eq("pre_debut")].merge(mlb, on=["player_id", "season"], suffixes=("_aaa", "_mlb")).assign(cohort="within")
    df = pd.concat([across, within], ignore_index=True).rename(columns={"season": "outcome_season"})
    mid = pd.to_datetime(df["outcome_season"].astype(str) + "-07-01")
    df["age"] = (mid - df["player_id"].map(born)).dt.days / 365.25
    return df.sort_values(["outcome_season", "cohort", "player_id"]).reset_index(drop=True)


def reliability(lines: pd.DataFrame, side: str, before: int) -> dict:
    """Per metric, from Triple-A seasons before `before`: the league mean, the noise variance of one opportunity
    (from odd-game and even-game halves) and the spread of true talent. k = noise / talent is the sample at which a
    rate is half signal."""
    L = lines[lines["side"].eq(side) & lines["level"].eq("AAA") & lines["season"].lt(before)]
    key = ["player_id", "season"]
    h0, h1, full = (L[L["segment"].eq(s)].set_index(key) for s in ("half0", "half1", "full"))
    out = {}
    for m, ncol in METRICS[side].items():
        j = h0[[m, ncol]].add_suffix("_0").join(h1[[m, ncol]].add_suffix("_1"), how="inner").dropna()
        j = j[(j[f"{ncol}_0"] >= MIN_HALF) & (j[f"{ncol}_1"] >= MIN_HALF)]
        noise = float(np.mean((j[f"{m}_0"] - j[f"{m}_1"]) ** 2 / (1 / j[f"{ncol}_0"] + 1 / j[f"{ncol}_1"])))
        f = full[full["pa"].ge(MIN_AAA_WIDE)][[m, ncol]].dropna()
        mean = float(np.average(f[m], weights=f[ncol]))
        talent = float(np.mean((f[m] - mean) ** 2 - noise / f[ncol]))
        out[m] = {"mean": mean, "noise": noise, "talent": max(talent, 1e-12), "k": noise / max(talent, 1e-12),
                  "players": int(len(f)), "split_half_players": int(len(j))}
    return out


def shrink(x, n, rel: dict):
    r = rel["talent"] / (rel["talent"] + rel["noise"] / np.maximum(n, 1))
    return rel["mean"] + r * (x - rel["mean"]), rel["talent"] * (1 - r)        # the estimate and what's still unknown


def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    sw = np.sqrt(w)
    return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]


def _wquantile(v: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(v, kind="stable")
    c = np.cumsum(w[o]) / w.sum()
    return float(v[o][min(np.searchsorted(c, q), len(v) - 1)])


def predict_metric(tr: pd.DataFrame, te: pd.DataFrame, m: str, ncol: str, rel: dict) -> pd.DataFrame:
    """Each variant's prediction of the MLB rate for the fold's pairs, and the two 80% bands."""
    a, y, n = f"{m}_aaa", f"{m}_mlb", f"{ncol}_aaa"
    tr = tr.dropna(subset=[a, y, "age"])
    te = te.dropna(subset=[a, y, "age"]).copy()
    pub = tr[tr["pa_aaa"].ge(MIN_AAA)]
    one = np.ones
    out = te[["player_id", "cohort", "pa_aaa", "pa_mlb"]].assign(n=te[n], y=te[y])
    # raw: the published line and band
    w = np.minimum(pub["pa_aaa"], pub["pa_mlb"]).to_numpy(float)
    b = _wls(np.column_stack([one(len(pub)), pub[a]]), pub[y].to_numpy(float), w)
    out["raw"] = b[0] + b[1] * te[a]
    res = pub[y].to_numpy(float) - (b[0] + b[1] * pub[a].to_numpy(float))
    out["band_raw_lo"], out["band_raw_hi"] = out["raw"] + _wquantile(res, w, 0.1), out["raw"] + _wquantile(res, w, 0.9)
    # eb variants: shrink first, weight by the MLB sample
    for name, fit, age in (("eb", pub, False), ("eb_wide", tr, False), ("eb_age", tr, True)):
        xs, _ = shrink(fit[a].to_numpy(float), fit[n].to_numpy(float), rel)
        X = np.column_stack([one(len(fit)), xs] + ([fit["age"].to_numpy(float)] if age else []))
        b = _wls(X, fit[y].to_numpy(float), fit["pa_mlb"].to_numpy(float))
        xt, left = shrink(te[a].to_numpy(float), te[n].to_numpy(float), rel)
        out[name] = np.column_stack([one(len(te)), xt] + ([te["age"].to_numpy(float)] if age else [])) @ b
        if name == "eb_wide":   # band: what the line leaves unexplained, plus what the Triple-A sample leaves unknown
            _, left_fit = shrink(fit[a].to_numpy(float), fit[n].to_numpy(float), rel)
            base = max(float(np.average((fit[y].to_numpy(float) - X @ b) ** 2 - b[1] ** 2 * left_fit, weights=fit["pa_mlb"])), 0.0)
            half = Z80 * np.sqrt(base + b[1] ** 2 * left)
            out["band_sized_lo"], out["band_sized_hi"] = out[name] - half, out[name] + half
    return out


def _rmse_edges(p: pd.DataFrame, rng) -> dict:
    """RMSE per variant and each variant's gain over raw (positive = smaller error), bootstrap over players."""
    if len(p) < 10:
        return {"n": int(len(p))}
    se = np.column_stack([(p[v] - p["y"]) ** 2 for v in VARIANTS])
    groups = list(p.groupby("player_id").indices.values())
    draws = np.array([np.sqrt(se[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])].mean(axis=0))
                      for _ in range(BOOT)])
    rmse = np.sqrt(se.mean(axis=0))
    out = {"n": int(len(p)), "sd_y": float(p["y"].std()), "rmse": dict(zip(VARIANTS, map(float, rmse), strict=True)),
           "r": {v: float(p[v].corr(p["y"])) for v in VARIANTS}, "gain_over_raw": {}}
    for i, v in enumerate(VARIANTS[1:], 1):
        g = draws[:, 0] - draws[:, i]
        out["gain_over_raw"][v] = {"est": float(rmse[0] - rmse[i]), "ci90": [float(np.percentile(g, 5)), float(np.percentile(g, 95))],
                                   "pct": float(100 * (rmse[0] - rmse[i]) / rmse[0])}
    return out


def _gate_candidates(long: pd.DataFrame, rng) -> dict:
    """Candidate gate statistics for the pre-registered test, on development folds: one number per side, the RMSE
    over every player x metric of the error in units of that metric's MLB spread (its SD in the fold's training
    pairs), for the published line against eb_wide. Gain = published minus candidate (positive = smaller error),
    bootstrap over players, for each pool the gate could be measured on."""
    out = {}
    pools = {"50+": long["pa_aaa"] >= MIN_AAA_WIDE, "150+": long["pa_aaa"] >= MIN_AAA, "50-149": long["pa_aaa"] < MIN_AAA}
    for scope, sel in [("pooled", slice(None))] + [(str(int(t)), long["fold"].eq(t)) for t in sorted(long["fold"].unique())]:
        d = long[sel]
        out[scope] = {}
        for pool, mask in pools.items():
            q = d[mask.reindex(d.index)]
            se = np.column_stack([((q[v] - q["y"]) / q["scale"]) ** 2 for v in ("raw", "eb_wide")])
            groups = list(q.groupby("player_id").indices.values())
            draws = np.array([np.sqrt(se[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])].mean(axis=0))
                              for _ in range(BOOT)])
            r = np.sqrt(se.mean(axis=0))
            g = draws[:, 0] - draws[:, 1]
            out[scope][pool] = {"players": len(groups), "rows": int(len(q)), "published": float(r[0]), "candidate": float(r[1]),
                                "gain": float(r[0] - r[1]), "gain_pct": float(100 * (r[0] - r[1]) / r[0]),
                                "ci90": [float(np.percentile(g, 5)), float(np.percentile(g, 95))]}
    return out


def _coverage(p: pd.DataFrame) -> dict:
    out = {}
    for name, sub in (("all", p), ("50-149", p[p["pa_aaa"] < MIN_AAA]), ("150-299", p[p["pa_aaa"].between(MIN_AAA, 299)]),
                      ("300+", p[p["pa_aaa"] >= 300])):
        if len(sub) < 10:
            continue
        out[name] = {"n": int(len(sub))}
        for b in ("raw", "sized"):
            inside = sub["y"].between(sub[f"band_{b}_lo"], sub[f"band_{b}_hi"])
            out[name][b] = {"coverage": float(inside.mean()), "width": float((sub[f"band_{b}_hi"] - sub[f"band_{b}_lo"]).mean())}
    return out


def combined(tr: pd.DataFrame, te: pd.DataFrame, side: str, rel: dict) -> dict:
    """MLB xwOBA from every shrunk Triple-A metric and age (ridge) against the shrunk xwOBA line alone."""
    ms = list(METRICS[side])
    need = [f"{m}_aaa" for m in ms] + ["xwoba_mlb", "age"]
    tr, te = tr.dropna(subset=need), te.dropna(subset=need)

    def X(d):
        cols = [shrink(d[f"{m}_aaa"].to_numpy(float), d[f"{METRICS[side][m]}_aaa"].to_numpy(float), rel[m])[0] for m in ms]
        return np.column_stack(cols + [d["age"].to_numpy(float)])
    Xtr, Xte, y, w = X(tr), X(te), tr["xwoba_mlb"].to_numpy(float), tr["pa_mlb"].to_numpy(float)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    g = tr["player_id"].to_numpy()
    errs = {a: sum(float(np.sum(w[b] * (y[b] - Ridge(alpha=a).fit(Ztr[f], y[f], sample_weight=w[f]).predict(Ztr[b])) ** 2))
                   for f, b in GroupKFold(n_splits=5).split(Ztr, y, g)) for a in ALPHAS}
    alpha = min(errs, key=errs.get)
    all_p = Ridge(alpha=alpha).fit(Ztr, y, sample_weight=w).predict(Zte)
    i = ms.index("xwoba")
    b = _wls(np.column_stack([np.ones(len(tr)), Xtr[:, i]]), y, w)
    one_p = b[0] + b[1] * Xte[:, i]
    yt = te["xwoba_mlb"].to_numpy(float)

    def rmse(p):
        return float(np.sqrt(np.mean((yt - p) ** 2)))
    return {"n_train": int(len(tr)), "n": int(len(te)), "alpha": alpha, "rmse_all_metrics": rmse(all_p), "rmse_xwoba_line": rmse(one_p),
            "rmse_mean_only": rmse(np.full(len(te), np.average(y, weights=w))),
            "r_all_metrics": float(np.corrcoef(all_p, yt)[0, 1]), "r_xwoba_line": float(np.corrcoef(one_p, yt)[0, 1])}


def check_against_mart(lines: pd.DataFrame) -> dict | None:
    """Largest absolute difference from fct_translation_lines for the lines both hold (None without a built pipeline)."""
    db = HERE.parent / "pd_pipeline" / "pd.duckdb"
    if not db.exists():
        return None
    con = duckdb.connect(str(db), read_only=True)
    mart = con.sql("select * from fct_translation_lines").df()
    con.close()
    key = ["side", "player_id", "season", "level", "segment"]
    j = lines.merge(mart, on=key, suffixes=("", "_mart"))
    cols = ["pa", *sorted({m for s in METRICS.values() for m in s})]
    return {"lines": int(len(j)), "max_abs_diff": {c: float((j[c] - j[f"{c}_mart"]).abs().max()) for c in cols}}


def main() -> None:
    S = seasons()
    assert all(s < CFG.season for ss in S.values() for s in ss), "a development fold may not use the current season"
    lines = build_lines(Loader())
    born = ages()
    res = {"seasons": S, "min_aaa": MIN_AAA, "min_aaa_wide": MIN_AAA_WIDE, "min_mlb": MIN_MLB, "bootstrap_samples": BOOT,
           "variants": VARIANTS, "check_against_mart": check_against_mart(lines)}
    for side in METRICS:
        P = pairs(lines, side, born)
        folds = [int(t) for t in sorted(P["outcome_season"].unique()) if (P["outcome_season"] < t).any()]
        if not folds:
            raise SystemExit(f"no fold has earlier pairs to fit on (seasons {S}): python ingest/baseball/pull.py --history")
        rng = np.random.default_rng(SEED)
        R = {"pairs_by_outcome_season": {int(t): {c: int(n) for c, n in g["cohort"].value_counts().sort_index().items()}
                                         for t, g in P.groupby("outcome_season")},
             "missing_age": int(P["age"].isna().sum()), "folds": {}, "pooled": {}}
        preds = {m: [] for m in METRICS[side]}
        for t in folds:
            tr, te = P[P["outcome_season"] < t], P[P["outcome_season"] == t]
            rel = reliability(lines, side, t)
            F = {"n_train": int(len(tr)), "n_test": int(len(te)), "reliability": rel, "metrics": {},
                 "combined": combined(tr, te, side, rel)}
            for m, ncol in METRICS[side].items():
                p = predict_metric(tr, te, m, ncol, rel[m]).assign(fold=t, metric=m, scale=float(tr[f"{m}_mlb"].std()))
                preds[m].append(p)
                F["metrics"][m] = {"published_pool": _rmse_edges(p[p["pa_aaa"] >= MIN_AAA], rng),
                                   "small_sample_pool": _rmse_edges(p[p["pa_aaa"] < MIN_AAA], rng), "bands": _coverage(p)}
            R["folds"][int(t)] = F
        for m in METRICS[side]:
            p = pd.concat(preds[m], ignore_index=True)
            R["pooled"][m] = {"published_pool": _rmse_edges(p[p["pa_aaa"] >= MIN_AAA], rng),
                              "small_sample_pool": _rmse_edges(p[p["pa_aaa"] < MIN_AAA], rng), "bands": _coverage(p)}
        R["gate_candidates"] = _gate_candidates(pd.concat([p for ps in preds.values() for p in ps], ignore_index=True),
                                                np.random.default_rng(SEED + 1))
        res[side] = R
        print(f"\n== {side}: folds {folds}, pairs {R['pairs_by_outcome_season']}")
        for m, v in R["pooled"].items():
            pp, sp = v["published_pool"], v["small_sample_pool"]
            line = f"  {m:17s} n {pp['n']:4d} rmse raw {pp['rmse']['raw']:.4f} " + " ".join(
                f"{k} {g['pct']:+.1f}% [{g['ci90'][0]:+.4f},{g['ci90'][1]:+.4f}]" for k, g in pp["gain_over_raw"].items())
            if "rmse" in sp:
                line += f" | small n {sp['n']} " + " ".join(f"{k} {g['pct']:+.1f}%" for k, g in sp["gain_over_raw"].items())
            print(line)
            print("    80% bands " + "; ".join(f"{k} n {b['n']}: same {b['raw']['coverage']:.2f}, sized {b['sized']['coverage']:.2f}"
                                           for k, b in v["bands"].items()))
        for scope, pools in R["gate_candidates"].items():
            print(f"  gate candidates ({scope}): " + "; ".join(
                f"{k} n {g['players']}: {g['gain_pct']:+.1f}% [{g['ci90'][0]:+.4f},{g['ci90'][1]:+.4f}]" for k, g in pools.items()))
        for t, F in R["folds"].items():
            c = F["combined"]
            print(f"  MLB xwOBA, outcome {t} (n {c['n']}, fit on {c['n_train']}): all metrics rmse {c['rmse_all_metrics']:.4f} "
                  f"(r {c['r_all_metrics']:.2f}), xwOBA line {c['rmse_xwoba_line']:.4f} (r {c['r_xwoba_line']:.2f}), "
                  f"mean only {c['rmse_mean_only']:.4f}")
    (OUT / "development.json").write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
