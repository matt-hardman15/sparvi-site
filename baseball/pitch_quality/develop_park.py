"""PQ-PARK: a park adjustment for Triple-A pitch readings before they are graded (development).

PQ-LEVEL found that the same pitchers grade about 3 Stuff+ points lower in Triple-A than in MLB, and that the gap is
park and measurement: less induced break read in the high-altitude parks, and extension read about 0.1 ft shorter at
every Triple-A park. This estimates offsets for those readings and checks, on a season the offsets were not fitted on,
whether the same pitcher then grades the same at both levels.

DEVELOPMENT ONLY. No model is fitted: pitches are graded with the stored Stuff+ models. No outcomes are read (the
measure is a pitcher's grade at one level against his grade at the other, same season). Aggregates only.

Offsets come from pitchers seen at both levels in a season: each Triple-A pitch against that pitcher's MLB average for
the same pitch type that season (MLB games at Coors Field left out of the average). Forms, fixed before any grading:
    none     readings as published
    ext      extension moved by one Triple-A-wide offset
    level    ext, and induced vertical and horizontal break moved by one Triple-A-wide offset each
    add      ext, and both breaks moved by an offset per park
    scale    ext, and both breaks multiplied by one factor per park (thinner air shrinks all movement in proportion)
A park with fewer than MIN_PARK such pitches takes the Triple-A-wide value. Velocity is not adjusted (PQ-LEVEL could
not separate measurement from effort).

Folds: grade 2025 with offsets from 2023-2024; grade 2026 with offsets from 2024-2025. The form is chosen on the
first and confirmed on the second.

    python baseball/pitch_quality/develop_park.py

Writes output/development_park.json.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from features import FEATURES, build_features
from train import GROUPS, N_BAGS, OUT

from sparvi.config import RAW

FOLDS = {2025: [2023, 2024], 2026: [2024, 2025]}
FORMS = ["none", "ext", "level", "add", "scale"]
MIN_BOTH = 100          # pitches at each level for a pitcher to count (PQ-LEVEL's sample)
MIN_TYPE = 30           # MLB pitches of a type for its average to be a reference
MIN_PARK = 2000         # reference pitches for a park to get its own offset
ALTITUDE = ["SL", "RNO", "ABQ", "ELP", "LV"]     # Salt Lake, Reno, Albuquerque, El Paso, Las Vegas (PQ-LEVEL)
RAW_COLS = ["pfx_z", "pfx_x", "release_extension"]


def _raw(level: str, year: int) -> pd.DataFrame:
    return pd.read_parquet(RAW / f"statcast_{level}_{year}.parquet")


def _reference(year: int) -> pd.DataFrame:
    """Triple-A pitches of `year` from pitchers with an MLB average for that pitch type, with that average beside each."""
    m = _raw("mlb", year)
    m = m[m["home_team"].ne("COL")]
    ref = m.groupby(["pitcher", "pitch_type"])[RAW_COLS].agg(["mean", "size"])
    ref = ref[ref[("pfx_z", "size")] >= MIN_TYPE]
    ref = ref.xs("mean", axis=1, level=1).add_suffix("_mlb").reset_index()
    a = _raw("aaa", year)[["pitcher", "pitch_type", "home_team", *RAW_COLS]]
    return a.merge(ref, on=["pitcher", "pitch_type"]).dropna(subset=RAW_COLS + [c + "_mlb" for c in RAW_COLS])


def offsets(seasons: list[int]) -> dict:
    """Per-park and Triple-A-wide offsets from the reference pitches of `seasons` (raw units: feet)."""
    d = pd.concat([_reference(y) for y in seasons], ignore_index=True)
    for c in RAW_COLS:
        d[f"d_{c}"] = d[c] - d[f"{c}_mlb"]

    def scale(x: pd.DataFrame) -> float:      # least squares through the origin, both break axes stacked
        num = (x["pfx_z"] * x["pfx_z_mlb"]).sum() + (x["pfx_x"] * x["pfx_x_mlb"]).sum()
        return float(num / ((x["pfx_z_mlb"] ** 2).sum() + (x["pfx_x_mlb"] ** 2).sum()))

    # Horizontal break is signed by hand in the raw file: an additive offset is estimated on its arm-side size.
    d["d_hb"] = np.sign(d["pfx_x_mlb"]) * d["d_pfx_x"]
    level = {"ext": float(d["d_release_extension"].mean()), "ivb": float(d["d_pfx_z"].mean()), "hb": float(d["d_hb"].mean()),
             "scale": scale(d), "n": int(len(d)), "pitchers": int(d["pitcher"].nunique())}
    parks = {}
    for p, x in d.groupby("home_team"):
        own = len(x) >= MIN_PARK
        parks[p] = {"n": int(len(x)), "own": bool(own), "ivb": float(x["d_pfx_z"].mean()) if own else level["ivb"],
                    "hb": float(x["d_hb"].mean()) if own else level["hb"], "scale": scale(x) if own else level["scale"],
                    "ext": float(x["d_release_extension"].mean())}       # reported; the adjustment uses the level's
    return {"level": level, "parks": parks}


def adjust(a: pd.DataFrame, off: dict, form: str) -> pd.DataFrame:
    """Triple-A raw pitches with their readings moved to what an MLB park would have read."""
    a = a.copy()
    if form == "none":
        return a
    a["release_extension"] = a["release_extension"] - off["level"]["ext"]
    if form == "ext":
        return a
    side = np.sign(a["pfx_x"])               # the offset is on arm-side size; keep each pitch's own direction
    park = a["home_team"].map(off["parks"])
    if form == "level":
        a["pfx_z"] = a["pfx_z"] - off["level"]["ivb"]
        a["pfx_x"] = a["pfx_x"] - side * off["level"]["hb"]
    elif form == "add":
        a["pfx_z"] = a["pfx_z"] - park.map(lambda o: o["ivb"] if isinstance(o, dict) else off["level"]["ivb"])
        a["pfx_x"] = a["pfx_x"] - side * park.map(lambda o: o["hb"] if isinstance(o, dict) else off["level"]["hb"])
    elif form == "scale":
        k = park.map(lambda o: o["scale"] if isinstance(o, dict) else off["level"]["scale"])
        a["pfx_z"], a["pfx_x"] = a["pfx_z"] / k, a["pfx_x"] / k
    return a


def _models() -> dict:
    return {g: [lgb.Booster(model_file=str(OUT / "models" / f"stuff_{g}_{f}.txt")) for f in range(N_BAGS)] for g in GROUPS}


def _grade(models: dict, d: pd.DataFrame) -> np.ndarray:
    p = np.full(len(d), np.nan)
    for g, bag in models.items():
        ix = d["group"].eq(g).to_numpy()
        if ix.any():
            p[ix] = np.mean([m.predict(d.loc[ix, FEATURES]) for m in bag], axis=0)
    return p


def _se(x: pd.Series) -> float:
    return float(x.std() / np.sqrt(len(x)))


def fold(models: dict, year: int, seasons: list[int], sd: float) -> dict:
    off = offsets(seasons)
    araw, m = _raw("aaa", year), build_features(_raw("mlb", year))
    na, nm = araw.groupby("pitcher").size(), m.groupby("pitcher").size()
    both = na[na >= MIN_BOTH].index.intersection(nm[nm >= MIN_BOTH].index)
    araw, m = araw[araw["pitcher"].isin(both)], m[m["pitcher"].isin(both)].copy()
    m["xrv"] = _grade(models, m)
    gm = m.groupby("pitcher")["xrv"].mean()
    tm = m.groupby(["pitcher", "pitch_type"]).agg(n=("xrv", "size"), xrv_mlb=("xrv", "mean"), ivb_mlb=("ivb", "mean"), hb_mlb=("hb", "mean"),
                                                  ext_mlb=("extension", "mean"), velo_mlb=("velo", "mean")).query("n >= @MIN_TYPE").reset_index()
    pts = 1000 / sd                                   # xrv per pitch -> Stuff+ points
    res = {"train_seasons": seasons, "pitchers": int(len(both)), "offsets": off, "forms": {}}
    for form in FORMS:
        a = build_features(adjust(araw, off, form))
        a["xrv"] = _grade(models, a)
        gap = (pts * (gm - a.groupby("pitcher")["xrv"].mean())).dropna()
        j = a.merge(tm, on=["pitcher", "pitch_type"])
        j["d"] = pts * (j["xrv_mlb"] - j["xrv"])
        park = j.groupby("home_team").agg(n=("d", "size"), gap=("d", "mean"), ivb=("ivb", "mean"), ivb_mlb=("ivb_mlb", "mean"))
        park = park[park["n"] >= MIN_PARK]
        res["forms"][form] = {
            "gap_mlb_minus_aaa": {"mean": float(gap.mean()), "se": _se(gap), "median": float(gap.median()), "n": int(len(gap)),
                                  "mean_abs": float(gap.abs().mean())},
            "same_pitch_type_gap": float(j.groupby("pitcher")["d"].mean().mean()),
            "parks": {"n": int(len(park)), "sd": float(park["gap"].std()), "max_abs": float(park["gap"].abs().max()),
                      "worst": str(park["gap"].abs().idxmax()), "over_1_point": int((park["gap"].abs() > 1).sum()),
                      "altitude_mean": float(park.loc[park.index.isin(ALTITUDE), "gap"].mean()),
                      "others_mean": float(park.loc[~park.index.isin(ALTITUDE), "gap"].mean()),
                      "by_park": {p: round(float(v), 2) for p, v in park["gap"].sort_values().items()}},
            "reading_mlb_minus_aaa": {"ivb_in": float((j["ivb_mlb"] - j["ivb"]).mean()), "hb_in": float((j["hb_mlb"] - j["hb"]).mean()),
                                      "extension_ft": float((j["ext_mlb"] - j["extension"]).mean()),
                                      "velo_mph": float((j["velo_mlb"] - j["velo"]).mean())}}
        f = res["forms"][form]
        print(f"  {year} {form:6s} gap {f['gap_mlb_minus_aaa']['mean']:+.2f} (se {f['gap_mlb_minus_aaa']['se']:.2f}) same types {f['same_pitch_type_gap']:+.2f} | "
              f"parks sd {f['parks']['sd']:.2f} max {f['parks']['max_abs']:.2f} ({f['parks']['worst']}) over 1: {f['parks']['over_1_point']} "
              f"altitude {f['parks']['altitude_mean']:+.2f} others {f['parks']['others_mean']:+.2f}", flush=True)
    return res


def main() -> None:
    sd = json.loads((OUT / "scale_reference.json").read_text(encoding="utf-8"))["stuff"]["sd"]
    models = _models()
    res = {"min_both": MIN_BOTH, "min_type": MIN_TYPE, "min_park": MIN_PARK, "altitude_parks": ALTITUDE, "forms": FORMS, "folds": {}}
    for year, seasons in FOLDS.items():
        res["folds"][str(year)] = fold(models, year, seasons, sd)
    (OUT / "development_park.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
