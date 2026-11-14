"""This winter's free-agent pitchers by Stuff+, Location+, and Pitching+, with intervals.

Inputs: data/raw/free_agents_SEASON.parquet (pull_free_agents.py; a dated snapshot of MLB's list),
the pitch-level model predictions for the season (pitch_quality/train.py, gitignored), the frozen grade
scale (pitch_quality/output/scale_reference.json), and site/data/baseball/stuff_pitchers.json (role,
games, and the published grades, which the point estimates here must reproduce).

Who's in: pitchers on the list who are actually free (not players whose club option was exercised or
who were claimed off waivers) with at least MIN_PITCHES graded MLB pitches this season. Signing status
is as of the pull date.

Grades: 100 + 10 * (100 * mean(xRV) - reference mean) / reference sd, the same formula and frozen scale
as the published numbers. Intervals: 90% from a bootstrap that resamples a pitcher's games (pitches in
the same game move together), BOOT draws, fixed seed.

Writes baseball/free_agents/output/results.json. Deterministic. ~10 seconds.

    python baseball/free_agents/analysis.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sparvi.config import RAW, ROOT, baseball

CFG = baseball()
PQ = ROOT / "baseball" / "pitch_quality" / "output"
OUT = ROOT / "baseball" / "free_agents" / "output"
MIN_PITCHES = 300
BOOT, SEED = 1000, 3
GAP = 15                 # "results trailed / ran ahead of the grades": RV+ at least this far from Pitching+ (1.5 SD)
NOT_FREE = {"club option exercised", "claimed off waivers", "claimed off waivers."}
KINDS = {"stuff": "xrv_stuff", "loc": "xrv_loc", "pitching": "xrv_pitching", "rv": "rv"}


def history(ref: dict, rng: np.random.Generator) -> dict:
    """Did results that trailed (or ran ahead of) Pitching+ move toward it the next season? Out-of-sample pairs from
    pitch_quality/robustness.py (models fit on the season before the predictor season; 500+ pitches both seasons),
    on the same frozen scale. Groups by last season's gap, RV+ minus Pitching+; outcome: the change in RV+."""
    d = pd.read_csv(PQ / "yoy_pairs.csv")
    p_prev = 100 + 10 * (d["xrv100_pitching_prev"] - ref["pitching"]["mean"]) / ref["pitching"]["sd"]
    rv = lambda x: 100 + 10 * (x - ref["rv"]["mean"]) / ref["rv"]["sd"]  # noqa: E731
    rv_prev, rv_cur = rv(d["rv100_prev"]), rv(d["rv100_cur"])
    gap, change = (rv_prev - p_prev).to_numpy(), (rv_cur - rv_prev).to_numpy()
    out = {"gap": GAP, "n": int(len(d)), "seasons": [int(CFG.season) - 1, int(CFG.season)]}
    for name, mask in (("trailed", gap <= -GAP), ("ahead", gap >= GAP), ("close", np.abs(gap) < GAP)):
        x = change[mask]
        boot = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(BOOT)]
        out[name] = {"n": int(mask.sum()), "change": round(float(x.mean()), 1), "gap": round(float(gap[mask].mean()), 1),
                     "ci90": [round(float(v), 1) for v in np.percentile(boot, [5, 95])]}
    # Next season's RV+ on last season's Pitching+ and RV+ together: how much each carried forward.
    X = np.column_stack([np.ones(len(d)), p_prev, rv_prev])
    b = np.linalg.lstsq(X, rv_cur.to_numpy(), rcond=None)[0]
    out["coef_pitching"], out["coef_results"] = round(float(b[1]), 2), round(float(b[2]), 2)
    return out


def main() -> None:
    fa = pd.read_parquet(RAW / f"free_agents_{CFG.season}.parquet")
    pulled = str(fa["pulled_on"].iloc[0])
    fa = fa[(fa["position_type"] == "Pitcher") & ~fa["notes"].fillna("").str.lower().isin(NOT_FREE)]
    ref = json.loads((PQ / "scale_reference.json").read_text(encoding="utf-8"))
    if ref["applies_to_season"] != CFG.season:
        raise SystemExit(f"scale_reference applies to {ref['applies_to_season']}, not {CFG.season}")
    preds = pd.read_parquet(PQ / f"pitch_predictions_mlb_{CFG.season}.parquet",
                            columns=["game_pk", "pitcher", *KINDS.values()])
    preds = preds[preds["pitcher"].isin(fa["player_id"])]
    pub = {r["id"]: r for r in json.loads((ROOT / "site" / "data" / "baseball" / "stuff_pitchers.json").read_text(encoding="utf-8"))
           if r["level"] == "MLB"}
    plus = lambda x, k: 100 + 10 * (100 * x - ref[k]["mean"]) / ref[k]["sd"]  # noqa: E731

    rng = np.random.default_rng(SEED)
    rows = []
    for r in fa.sort_values("player_id").itertuples():
        p = preds[preds["pitcher"] == r.player_id]
        if len(p) < MIN_PITCHES:
            continue
        g = p.groupby("game_pk", sort=True)
        sums, counts = g[list(KINDS.values())].sum().to_numpy(), g.size().to_numpy()
        pick = rng.integers(0, len(counts), (BOOT, len(counts)))
        boot = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)[:, None]      # BOOT x kinds, mean xRV
        txt = lambda v: None if pd.isna(v) else str(v)  # noqa: E731
        row = {"id": int(r.player_id), "name": r.name, "from_team": txt(r.from_team), "to_team": txt(r.to_team),
               "date_signed": txt(r.date_signed), "notes": txt(r.notes), "pitches": int(len(p)), "games": int(len(counts))}
        for j, (k, col) in enumerate(KINDS.items()):
            row[f"{k}_plus"] = round(float(plus(p[col].mean(), k)), 1)
            lo, hi = np.percentile(plus(boot[:, j], k), [5, 95])
            row[f"{k}_lo"], row[f"{k}_hi"] = round(float(lo), 1), round(float(hi), 1)
        pp = pub.get(int(r.player_id))
        if pp is None:
            raise SystemExit(f"{r.name} ({r.player_id}) has {len(p)} graded pitches but no published MLB row")
        for k in ("stuff", "loc", "pitching", "rv"):          # must reproduce the published grades
            if abs(pp[f"{k}_plus"] - row[f"{k}_plus"]) > 0.11:
                raise SystemExit(f"{r.name}: {k}_plus {row[f'{k}_plus']} here vs {pp[f'{k}_plus']} published")
        row.update({"role": pp["role"], "g": pp["g"], "gs": pp["gs"], "team": pp["team"]})
        rows.append(row)
    rows.sort(key=lambda x: -x["stuff_plus"])
    hist = history(ref, rng)
    res = {"season": CFG.season, "pulled_on": pulled, "min_pitches": MIN_PITCHES, "boot": BOOT, "seed": SEED,
           "reference_season": ref["reference_season"], "n_list_pitchers": int(len(fa)), "history": hist, "pitchers": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(rows)} free-agent pitchers with {MIN_PITCHES}+ MLB pitches in {CFG.season} (list as of {pulled})")
    for x in rows[:12]:
        print(f"  {x['name']:<24} {x['role']:<5} Stuff+ {x['stuff_plus']:6.1f} [{x['stuff_lo']}, {x['stuff_hi']}]  "
              f"Pitching+ {x['pitching_plus']:6.1f}  RV+ {x['rv_plus']:6.1f}  {x['to_team'] or 'unsigned'}")


if __name__ == "__main__":
    main()
