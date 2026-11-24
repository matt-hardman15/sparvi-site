"""Development folds for arsenal and deception features in Stuff+ (PQ-ARSENAL): do inputs for what the grade can't
see now raise what it predicts?

DEVELOPMENT ONLY, on the folds of develop_loc.py (outcome seasons before the current one; the current season is never
read). PRED-AUDIT found that, given Stuff+, last season's strikeout rate still predicts next-season RV/100: pitchers
miss more bats than the measured shape explains, and it persists. The published features describe one pitch and its
gap from the pitcher's primary fastball. The groups below add what they leave out. Every input is still something the
pitcher does physically: no location, count, outcome or usage rate as a value of its own.

Feature groups (fixed before the run; each is added to the published Stuff+ features, same training):
    mix       the pitch against the pitcher's other pitch types (usage-weighted season means of the others): velocity
              and both movement gaps; how many types he throws 5%+ of the time; how far apart they are in movement and
              in velocity. A primary fastball's gap from itself is zero in the published features, so the model sees
              a fastball without its arsenal
    release   the pitch type's season mean release point and extension against the primary fastball's (a tell), and
              the spread of the type's release point (repeatability)
    spin      movement the spin doesn't explain: the angle between the direction the pitch moved and the direction its
              spin axis implies (seam effects), and inches of movement per 1,000 rpm (how much of the spin moves the ball)
    angle     vertical and horizontal approach angle at the plate, as if the pitch had been thrown to the middle of the
              zone (2.5 ft, centre): the pitch's own geometry with where it was aimed taken out
    all       every group

Scored as the published test scores Stuff+: the pitcher's mean predicted value in the graded season against his RV/100
the next season (500+ pitches in both), by fold and as the mean of the folds; each variant's edge over the published
features is a paired bootstrap over pitchers. Also reported, because it is what the features are for: the partial
correlation of last season's strikeout rate with next-season RV/100 given the grade (falls if the grade now sees the
bat-missing), and the grade's own correlation with that strikeout rate. `base` and `all` are also fit at a second seed:
an edge smaller than the difference between seeds is not one.

    python baseball/pitch_quality/develop_arsenal.py                 # every variant
    python baseball/pitch_quality/develop_arsenal.py check           # the new columns' distributions, no fits

Writes output/development_arsenal.json.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from develop import BOOT, CACHE, CFG, KEY, OUT, _code_key, _pitches, folds
from develop_combine import _record
from develop_loc import WINDOW
from develop_loc_stable import SEEDS, _fit
from features import FASTBALLS, FEATURES
from train import LEARNING_RATE, MIN_PITCHES, N_BAGS, SEED, predict

from backtest.asof import Loader

GROUPS = {
    "mix": ["velo_vs_mix", "ivb_vs_mix", "hb_vs_mix", "n_types", "move_spread", "velo_range"],
    "release": ["rel_x_vs_fb", "rel_z_vs_fb", "ext_vs_fb", "rel_sd"],
    "spin": ["axis_dev", "move_per_rpm"],
    "angle": ["vaa_mid", "haa_mid"],
}
NEW = [c for cols in GROUPS.values() for c in cols]
VARIANTS = {"base": FEATURES, **{g: FEATURES + cols for g, cols in GROUPS.items()}, "all": FEATURES + NEW}
SECOND_SEED = ("base", "all")
PHYS = ["vx0", "vy0", "vz0", "ax", "ay", "az", "plate_x", "plate_z"]
MIN_SHARE = 0.05                      # a pitch type counts toward the arsenal's size and range at 5% of his pitches
Y0, PLATE, MID_Z = 50.0, 17 / 12, 2.5  # Statcast's start distance, the plate's front edge, mid-zone height (ft)


def _file() -> str:
    return _code_key() + "_" + __import__("hashlib").sha256(open(__file__, "rb").read().replace(b"\r\n", b"\n")).hexdigest()[:8]


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """The new columns, on build_features rows that also carry PHYS. Handedness is already mirrored in those rows."""
    d = d.copy()
    who, typ = ["pitcher", "game_year"], ["pitcher", "game_year", "pitch_type"]
    g = d.groupby(typ)
    m = g.agg(n=("velo", "size"), velo=("velo", "mean"), ivb=("ivb", "mean"), hb=("hb", "mean"), rel_x=("rel_x", "mean"),
              rel_z=("rel_z", "mean"), ext=("extension", "mean"), sd_x=("rel_x", "std"), sd_z=("rel_z", "std")).reset_index()
    tot = m.groupby(who)["n"].transform("sum")
    m["share"] = m["n"] / tot
    for c in ("velo", "ivb", "hb"):             # the other types' usage-weighted mean: undefined with one pitch type
        s = (m[c] * m["n"]).groupby([m[k] for k in who]).transform("sum")
        m[f"{c}_others"] = ((s - m[c] * m["n"]) / (tot - m["n"])).where(tot > m["n"])
    cx = (m["hb"] * m["share"]).groupby([m[k] for k in who]).transform("sum")
    cz = (m["ivb"] * m["share"]).groupby([m[k] for k in who]).transform("sum")
    m["move_spread"] = np.sqrt((m["share"] * ((m["hb"] - cx) ** 2 + (m["ivb"] - cz) ** 2)).groupby([m[k] for k in who]).transform("sum"))
    main = m[m["share"] >= MIN_SHARE]
    per = main.groupby(who).agg(n_types=("n", "size"), v_hi=("velo", "max"), v_lo=("velo", "min")).reset_index()
    per["velo_range"] = per["v_hi"] - per["v_lo"]
    m = m.merge(per[who + ["n_types", "velo_range"]], on=who, how="left")
    m["rel_sd"] = np.sqrt(m["sd_x"] ** 2 + m["sd_z"] ** 2)
    fb = (m[m["pitch_type"].isin(FASTBALLS)].sort_values("n", ascending=False, kind="stable").drop_duplicates(who)
          [who + ["rel_x", "rel_z", "ext"]].rename(columns={"rel_x": "fb_rel_x", "rel_z": "fb_rel_z", "ext": "fb_ext"}))
    m = m.merge(fb, on=who, how="left")         # no fastball: NaN, as the published gaps
    m["rel_x_vs_fb"], m["rel_z_vs_fb"], m["ext_vs_fb"] = m["rel_x"] - m["fb_rel_x"], m["rel_z"] - m["fb_rel_z"], m["ext"] - m["fb_ext"]
    keep = ["velo_others", "ivb_others", "hb_others", "n_types", "move_spread", "velo_range", "rel_sd", "rel_x_vs_fb", "rel_z_vs_fb", "ext_vs_fb"]
    d = d.merge(m[typ + keep], on=typ, how="left", validate="many_to_one")
    for c in ("velo", "ivb", "hb"):
        d[f"{c}_vs_mix"] = d[c] - d[f"{c}_others"]

    # Spin: 180 degrees is pure backspin (the ball moves up); in the mirrored frame (arm side positive) the implied
    # direction of movement is (-sin, -cos) of the axis. The deviation is the observed direction minus that one.
    a = np.deg2rad(d["spin_axis"])
    dev = np.rad2deg(np.arctan2(d["ivb"], d["hb"]) - np.arctan2(-np.cos(a), -np.sin(a)))
    d["axis_dev"] = (dev + 180) % 360 - 180
    d["move_per_rpm"] = 1000 * np.hypot(d["hb"], d["ivb"]) / d["spin"].where(d["spin"] > 0)

    # Approach angle at the front of the plate from the constant-acceleration fit, re-aimed at the middle of the zone:
    # moving the target by x ft changes the crossing velocity by x / flight time and nothing else.
    vy = -np.sqrt(d["vy0"] ** 2 - 2 * d["ay"] * (Y0 - PLATE))
    t = (vy - d["vy0"]) / d["ay"]
    vz = d["vz0"] + d["az"] * t - (d["plate_z"] - MID_Z) / t
    vx = d["vx0"] + d["ax"] * t - d["plate_x"] / t
    side = np.where(d["p_throws"].eq("L"), 1.0, -1.0)          # as features.py mirrors hb: arm side positive
    d["vaa_mid"] = -np.rad2deg(np.arctan(vz / vy))
    d["haa_mid"] = -np.rad2deg(np.arctan(vx * side / vy))
    return d.drop(columns=["velo_others", "ivb_others", "hb_others"])


def _rows(loader: Loader, season: int, as_of: str, heights: pd.Series) -> pd.DataFrame:
    raw = loader.load(f"statcast_mlb_{season}", as_of)
    d = _pitches(loader, season, as_of, heights).merge(raw[[*KEY, *PHYS]].drop_duplicates(KEY), on=KEY, how="left", validate="one_to_one")
    return add_features(d).drop(columns=PHYS)


def _grades(loader: Loader, t: int, seasons: list[int], heights: pd.Series, d: pd.DataFrame, todo: list[tuple[str, int]]) -> dict:
    """{(variant, seed): pitch values on the graded season}; each fit is kept so a stopped run resumes."""
    tr, out = None, {}
    for name, seed in todo:
        tag = f"ars_{t}_{'-'.join(map(str, seasons))}_{name}_s{seed}_{_file()}"
        f_p, f_t = CACHE / f"{tag}.parquet", CACHE / f"{tag}.json"
        if not f_p.exists():
            if tr is None:
                keep = sorted(set(FEATURES + NEW)) + ["pitcher", "group", "rv_x"]
                tr = pd.concat([_rows(loader, s, f"{t - 1}-03-01", heights)[keep] for s in seasons], ignore_index=True)
            bags = _fit(tr, VARIANTS[name], LEARNING_RATE["stuff"], N_BAGS, seed)
            CACHE.mkdir(exist_ok=True)
            pd.DataFrame({"p": predict(bags, d, VARIANTS[name])}).to_parquet(f_p, index=False)
            gain = {}
            for bag in bags.values():
                for mdl in bag:
                    for c, v in zip(VARIANTS[name], mdl.feature_importance("gain"), strict=True):
                        gain[c] = gain.get(c, 0.0) + float(v)
            total = sum(gain.values())
            f_t.write_text(json.dumps({"trees": {g: [min(x := [mdl.best_iteration for mdl in bag]), max(x)] for g, bag in bags.items()},
                                       "gain_share": {c: round(v / total, 4) for c, v in gain.items()}}), encoding="utf-8")
            print(f"  outcome {t}: fit {name} seed {seed}", flush=True)
        out[(name, seed)] = (pd.read_parquet(f_p)["p"].to_numpy(), json.loads(f_t.read_text(encoding="utf-8")))
    return out


def _partial(x: pd.Series, y: pd.Series, z: pd.Series) -> float:
    rxy, rxz, ryz = x.corr(y), x.corr(z), y.corr(z)
    return float((rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)))


def _table(loader: Loader, t: int, seasons: list[int], heights: pd.Series, names: list[str]) -> tuple[pd.DataFrame, dict]:
    """Pitcher rows of a fold: each variant's grade, last season's strikeout rate, next season's RV/100 (y)."""
    d = _rows(loader, t - 1, f"{t}-03-01", heights)
    todo = [(n, SEED) for n in names] + [(n, SEEDS[1]) for n in names if n in SECOND_SEED]
    fits = _grades(loader, t, seasons, heights, d, todo)
    g = d["pitcher"].to_numpy()
    ps = pd.DataFrame({(n if s == SEED else f"{n}_s{s}"): 100 * pd.Series(p).groupby(g).mean() for (n, s), (p, _) in fits.items()})
    prev = _record(loader, t - 1, f"{t}-03-01", heights)
    cur = _record(loader, t, f"{t + 1}-03-01", heights)
    q = (ps.join(prev.loc[prev["pitches"] >= MIN_PITCHES, ["k_pct"]], how="inner")
         .join(cur.loc[cur["pitches"] >= MIN_PITCHES, ["rv100"]].rename(columns={"rv100": "y"}), how="inner"))
    return q.rename_axis("pitcher").reset_index().assign(fold=t), {n: info for (n, s), (_, info) in fits.items() if s == SEED}


def _mean_r(d: pd.DataFrame, col: str) -> float:
    return float(np.mean([f[col].corr(f["y"]) for _, f in d.groupby("fold")]))


def check() -> None:
    """The new columns on the newest development season: shares missing and quartiles by pitch group. No fits."""
    loader = Loader()
    h = loader.load("batter_heights", f"{CFG.season}-03-01")
    t = max(folds())
    d = _rows(loader, t - 1, f"{t}-03-01", (h.set_index("batter")["height_in"] / 12).astype(float))
    print(d.groupby("group")[NEW].agg(lambda c: round(float(c.isna().mean()), 4)).T.to_string())
    print(d.groupby("group")[NEW].quantile([0.05, 0.5, 0.95]).round(2).T.to_string())
    print(d.groupby("pitch_type")[["axis_dev", "vaa_mid", "haa_mid", "move_per_rpm"]].median().round(2).T.to_string())


def main() -> None:
    names = list(VARIANTS)
    loader = Loader()
    h = loader.load("batter_heights", f"{CFG.season}-03-01")
    heights = (h.set_index("batter")["height_in"] / 12).astype(float)
    F = {t: w[WINDOW] for t, w in folds().items() if WINDOW in w}
    assert all(t < CFG.season for t in F), "a development fold may not use the current season's outcomes"
    tables, infos = {}, {}
    for t, seasons in F.items():
        tables[t], infos[t] = _table(loader, t, seasons, heights, names)
    data = pd.concat(tables.values(), ignore_index=True).dropna().reset_index(drop=True)
    groups = list(data.groupby("pitcher").indices.values())
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(BOOT)]
    boot_base = np.array([_mean_r(data.iloc[ix], "base") for ix in draws])
    res = {"folds": {str(t): {"train_seasons": s, "grade_season": t - 1, "n": int((data["fold"] == t).sum())} for t, s in F.items()},
           "n": int(len(data)), "pitchers": len(groups), "min_pitches": MIN_PITCHES, "seeds": SEEDS, "bootstrap_samples": BOOT,
           "reference": "base", "groups": GROUPS, "variants": {}}
    for n in names:
        by = {str(t): f for t, f in data.groupby("fold")}
        v = {"features": len(VARIANTS[n]), "r": round(_mean_r(data, n), 4),
             "r_by_fold": {t: round(float(f[n].corr(f["y"])), 4) for t, f in by.items()},
             "partial_r_k_given_grade": round(float(np.mean([_partial(f["k_pct"], f["y"], f[n]) for f in by.values()])), 4),
             "r_grade_with_k": round(float(np.mean([f[n].corr(f["k_pct"]) for f in by.values()])), 4),
             "r_with_base_grade": round(float(np.mean([f[n].corr(f["base"]) for f in by.values()])), 4),
             "trees": {str(t): infos[t][n]["trees"] for t in F},
             "gain_share_new": {str(t): {c: infos[t][n]["gain_share"][c] for c in VARIANTS[n] if c in NEW} for t in F}}
        if n in SECOND_SEED:
            v["r_other_seed"] = round(_mean_r(data, f"{n}_s{SEEDS[1]}"), 4)
        if n != "base":
            b = np.array([_mean_r(data.iloc[ix], n) for ix in draws]) - boot_base
            v["vs_base"] = {"r": round(v["r"] - _mean_r(data, "base"), 4), "r_ci90": [round(float(x), 4) for x in np.percentile(b, [5, 95])]}
        res["variants"][n] = v
        e = v.get("vs_base")
        print(f"{n:8s} r {v['r']:+.3f} ({' '.join(f'{x:+.3f}' for x in v['r_by_fold'].values())})  K given grade {v['partial_r_k_given_grade']:+.3f}  "
              f"grade with K {v['r_grade_with_k']:+.3f}" + (f"  seed {SEEDS[1]}: {v['r_other_seed']:+.3f}" if "r_other_seed" in v else "")
              + (f"  vs base {e['r']:+.3f} ({e['r_ci90'][0]:+.3f} to {e['r_ci90'][1]:+.3f})" if e else ""), flush=True)
    (OUT / "development_arsenal.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    check() if sys.argv[1:] == ["check"] else main()
