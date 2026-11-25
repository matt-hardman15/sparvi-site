"""Triple-A pitch grades at altitude (R5-ALTITUDE): how much lower the five high-altitude parks read a pitcher's stuff,
and what each Triple-A pitcher's Stuff+ would be on readings moved to what an MLB park would have recorded.

Method (PQ-PARK; chosen on a held-out season in pitch_quality/develop_park.py):
  - Park factors come from pitchers seen at both levels in the two seasons BEFORE the one graded: each Triple-A pitch
    against that pitcher's MLB average for the same pitch type (MLB games at Coors Field left out). A park's factor is
    the share of MLB movement it reads, both break axes together (least squares through the origin). Extension gets
    one Triple-A-wide offset. Velocity is not adjusted.
  - The season's Triple-A pitches are graded twice with the published Stuff+ models: as recorded, and with break
    divided by the park's factor and extension moved by the offset. Same formula and frozen scale as the published
    grade, so the as-recorded grade must reproduce the published one (checked here against the Rule 5 board's file).
  - Factor intervals: 90%, bootstrap over pitchers (a pitcher's pitches move together), BOOT draws, fixed seed.

This is an analysis beside the published grades, not a change to them: the site's Stuff+ is as recorded until the
models are retrained with the adjustment.

Runs by hand (it needs the two earlier seasons at both levels and grades a season twice; about ten minutes):

    python baseball/altitude/analysis.py

Writes baseball/altitude/output/results.json (park-level aggregates and pitcher-level grades; no pitch-level data).
Deterministic.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from sparvi.config import RAW, ROOT, baseball

PQ = ROOT / "baseball" / "pitch_quality"
sys.path.insert(0, str(PQ))
import develop_park as P  # noqa: E402
import train as T  # noqa: E402
from features import FEATURES, build_features  # noqa: E402

CFG = baseball()
OUT = ROOT / "baseball" / "altitude" / "output"
SEASON = CFG.season
FIT_SEASONS = [SEASON - 2, SEASON - 1]          # never the season graded
MIN_PITCHES = 100                               # a pitcher's adjusted grade is shown from this many Triple-A pitches
BOOT, SEED = 1000, 11


def factor_intervals(seasons: list[int]) -> dict:
    """90% interval of each park's factor and of the extension offset, resampling pitchers."""
    d = pd.concat([P._reference(y) for y in seasons], ignore_index=True)
    d["num"] = d["pfx_z"] * d["pfx_z_mlb"] + d["pfx_x"] * d["pfx_x_mlb"]
    d["den"] = d["pfx_z_mlb"] ** 2 + d["pfx_x_mlb"] ** 2
    d["ext"] = d["release_extension"] - d["release_extension_mlb"]
    rng = np.random.default_rng(SEED)
    out = {}
    for park, x in d.groupby("home_team"):
        g = x.groupby("pitcher")[["num", "den"]].sum().to_numpy()
        ix = rng.integers(0, len(g), size=(BOOT, len(g)))
        draws = g[ix, 0].sum(axis=1) / g[ix, 1].sum(axis=1)
        out[park] = {"ci90": [float(np.percentile(draws, 5)), float(np.percentile(draws, 95))], "pitchers": int(len(g))}
    g = d.groupby("pitcher")["ext"].agg(["sum", "size"]).to_numpy()
    ix = rng.integers(0, len(g), size=(BOOT, len(g)))
    draws = g[ix, 0].sum(axis=1) / g[ix, 1].sum(axis=1)
    return {"parks": out, "extension_ci90": [float(np.percentile(draws, 5)), float(np.percentile(draws, 95))]}


def published_models() -> dict:
    """The Stuff+ models the published grades were scored with: the bundle on record (pitch_quality/model_artifacts.json),
    from the local store or the private bucket, checked against its recorded sha256. A machine's own raw files can
    differ from the refresh's in a few rows, which is enough to key (and fit) a different model, so `train.fitted`
    is only right where it finds the recorded key."""
    import hashlib

    import artifacts as A
    rec = json.loads(A.RECORD.read_text(encoding="utf-8"))["production"]
    path = A.STORE / f"production-{rec['key'][:20]}.zip"
    if path.exists():
        data = path.read_bytes()
    else:
        from ingest.common.snapshots import store_from_env
        store = store_from_env()
        if store is None:
            raise SystemExit("the published models are not in the local store and the bucket is not configured")
        data = store.get(rec["object"])
        A.STORE.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if hashlib.sha256(data).hexdigest() != rec["sha256"]:
        raise SystemExit("the stored model bundle is not the one on record")
    print(f"published models: key {rec['key'][:12]}, sha256 {rec['sha256'][:12]}")
    return A.unbundle(data)["stuff"]


def main() -> None:
    ref = json.loads((PQ / "output" / "scale_reference.json").read_text(encoding="utf-8"))["stuff"]
    assert json.loads((PQ / "output" / "scale_reference.json").read_text(encoding="utf-8"))["applies_to_season"] == SEASON
    models = published_models()
    off = P.offsets(FIT_SEASONS)
    ci = factor_intervals(FIT_SEASONS)
    raw = pd.read_parquet(RAW / f"statcast_aaa_{SEASON}.parquet")
    graded = {}
    for form in ("none", "scale"):
        d = build_features(P.adjust(raw, off, form))
        d["xrv"] = T.predict(models, d, FEATURES).to_numpy()
        graded[form] = d
    a, b = graded["none"], graded["scale"]
    key = ["game_pk", "at_bat_number", "pitch_number"]
    assert a[key].equals(b[key]), "the two gradings must be of the same pitches in the same order"
    a = a.assign(xrv_adj=b["xrv"].to_numpy(), d_pts=1000 * (b["xrv"].to_numpy() - a["xrv"].to_numpy()) / ref["sd"],
                 at_altitude=a["home_team"].isin(P.ALTITUDE))

    def plus(x):
        return 100 + 10 * (100 * x - ref["mean"]) / ref["sd"]

    # Parks: the factor and what the adjustment is worth to the pitches thrown there.
    by_park = a.groupby("home_team").agg(pitches=("xrv", "size"), points=("d_pts", "mean"))
    parks = [{"park": p, "altitude": p in P.ALTITUDE, "factor": off["parks"][p]["scale"], "factor_ci90": ci["parks"][p]["ci90"],
              "fit_pitches": off["parks"][p]["n"], "fit_pitchers": ci["parks"][p]["pitchers"],
              "pitches": int(r.pitches), "points": float(r.points)} for p, r in by_park.iterrows()]
    # Pitchers: as recorded and adjusted, with the share of pitches thrown at altitude.
    g = a.groupby("pitcher")
    pit = pd.DataFrame({"name": g["player_name"].first(), "team": g["team"].agg(lambda s: s.mode().iloc[0]), "pitches": g["xrv"].size(),
                        "stuff_plus": plus(g["xrv"].mean()), "stuff_plus_adj": plus(g["xrv_adj"].mean()),
                        "altitude_share": g["at_altitude"].mean()}).reset_index()
    pit["delta"] = pit["stuff_plus_adj"] - pit["stuff_plus"]
    pit = pit[pit["pitches"] >= MIN_PITCHES].sort_values("pitcher").reset_index(drop=True)
    # The as-recorded grade must be the published one for nearly every pitcher (the Rule 5 board's file is the committed
    # export). Not for all: a machine's Triple-A file can differ from the refresh's in a few games. The site applies each
    # pitcher's CHANGE (adjusted less as-recorded, both graded here on the same pitches) to his published grade, so the
    # article always matches the board; this check guards the method, not the last decimal.
    board = pd.DataFrame(json.loads((ROOT / "site" / "data" / "baseball" / "rule5_pitchers.json").read_text(encoding="utf-8")))
    chk = board.merge(pit, left_on="player_id", right_on="pitcher", suffixes=("_pub", ""))
    off_by = (chk["stuff_plus_pub"] - chk["stuff_plus"]).abs()
    diff, share = float(off_by.max()), float((off_by < 0.06).mean())
    assert len(chk) >= 100 and share >= 0.98, f"the as-recorded grade reproduces the published one for only {share:.1%} of board pitchers"
    dev = json.loads((PQ / "output" / "development_park.json").read_text(encoding="utf-8"))["folds"][str(SEASON)]
    assert dev["train_seasons"] == FIT_SEASONS

    def val(form):
        f = dev["forms"][form]
        return {"gap": f["gap_mlb_minus_aaa"]["mean"], "se": f["gap_mlb_minus_aaa"]["se"], "altitude": f["parks"]["altitude_mean"],
                "others": f["parks"]["others_mean"], "largest_park": f["parks"]["max_abs"]}
    res = {"season": SEASON, "fit_seasons": FIT_SEASONS, "min_pitches": MIN_PITCHES, "bootstrap": BOOT, "seed": SEED,
           "altitude_parks": P.ALTITUDE, "extension_offset_ft": off["level"]["ext"], "extension_ci90": ci["extension_ci90"],
           "level_factor": off["level"]["scale"], "fit_reference_pitches": off["level"]["n"], "fit_reference_pitchers": off["level"]["pitchers"],
           "validation": {"pitchers": dev["pitchers"], "as_recorded": val("none"), "adjusted": val("scale"),
                          "additive_altitude": dev["forms"]["add"]["parks"]["altitude_mean"]},
           "published_check": {"pitchers": int(len(chk)), "share_within_rounding": share, "max_abs_difference": diff},
           "parks": sorted(parks, key=lambda r: r["factor"]),
           "pitchers": [{"player_id": int(r.pitcher), "name": r.name, "team": r.team, "pitches": int(r.pitches),
                         "stuff_plus": round(float(r.stuff_plus), 1), "stuff_plus_adj": round(float(r.stuff_plus_adj), 1),
                         "delta": round(float(r.delta), 1), "altitude_share": round(float(r.altitude_share), 3)} for r in pit.itertuples()]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    alt = pit[pit["team"].isin(P.ALTITUDE)]
    print(f"{len(pit)} pitchers with {MIN_PITCHES}+ pitches; published check on {len(chk)} board pitchers, {share:.1%} within rounding (max difference {diff:.2f})")
    print(f"altitude teams: {len(alt)} pitchers, mean change {alt['delta'].mean():+.2f}; others {pit[~pit['team'].isin(P.ALTITUDE)]['delta'].mean():+.2f}")
    print(pd.DataFrame(parks).sort_values("factor").head(8).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
