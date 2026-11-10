"""Do in-season pitch shape changes stick?

For every MLB pitcher-pitch type with 100+ pitches in each half of a season, measure
the change in velocity, induced vertical break (IVB), and horizontal break (HB) from
the first half to the second. Then look at the first half of the *next* season and
ask how much of that change was still there.

Persistence = (next-season H1 - this-season H1) / (this-season H2 - this-season H1)
  1.0 = the whole change held; 0 = fully reverted.
Pooled with a regression slope rather than averaging per-pitch ratios (which blow up
when the original change is small).

Season pairs: 2024->2025 and 2025->2026.
"""
import json
import sys
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import plotstyle as ps  # noqa: E402

RAW = HERE.parents[1] / "data" / "raw"
OUT, FIG = HERE / "output", HERE / "figures"
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
MIN_N = 100
SPLIT = "07-15"  # roughly the All-Star break
THRESH = {"velo": 1.0, "ivb": 1.5, "hb": 1.5}  # what counts as a real change
METRICS = list(THRESH)
LABEL = {"velo": "Velocity (mph)", "ivb": "Induced vertical break (in)", "hb": "Horizontal break (in)"}

con = duckdb.connect()
halves = con.sql(f"""
    select
        pitcher, player_name, pitch_type, game_year as season,
        case when strftime(cast(game_date as date), '%m-%d') < '{SPLIT}' then 1 else 2 end as half,
        count(*)                as n,
        avg(release_speed)      as velo,
        avg(pfx_z * 12)         as ivb,
        avg(case when p_throws = 'R' then -pfx_x else pfx_x end * 12) as hb
    from read_parquet('{RAW}/statcast_mlb_*.parquet')
    where game_type = 'R' and pitch_type not in ('PO', 'EP', 'KN') and pitch_type is not null
    group by all
""").df()

w = halves.pivot_table(index=["pitcher", "player_name", "pitch_type", "season"],
                       columns="half", values=["n"] + METRICS).reset_index()
w.columns = [f"{a}{b}" if b != "" else a for a, b in w.columns]

rows = []
for yr in [2024, 2025]:
    a = w[(w.season == yr) & (w.n1 >= MIN_N) & (w.n2 >= MIN_N)]
    b = w[(w.season == yr + 1) & (w.n1 >= MIN_N)][["pitcher", "pitch_type"] + [f"{m}1" for m in METRICS]]
    b = b.rename(columns={f"{m}1": f"{m}_next" for m in METRICS})
    rows.append(a.merge(b, on=["pitcher", "pitch_type"]).assign(pair=f"{yr}->{yr + 1}"))
d = pd.concat(rows, ignore_index=True)
league = {}
for m in METRICS:
    d[f"{m}_chg"] = d[f"{m}2"] - d[f"{m}1"]          # in-season change
    d[f"{m}_kept"] = d[f"{m}_next"] - d[f"{m}1"]     # how much is left next spring
    # Remove the league-wide component for each season pair: velocity climbs into the
    # summer for almost everyone, and tracking calibration can shift between seasons.
    # What's left is the change specific to this pitcher.
    for col in [f"{m}_chg", f"{m}_kept"]:
        means = d.groupby("pair")[col].transform("mean")
        league.setdefault(col, d.groupby("pair")[col].mean().round(3).to_dict())
        d[col] = d[col] - means

results = {"n_pitch_types": int(len(d)), "pairs": sorted(d.pair.unique().tolist()),
           "league_mean_adjustments": league, "metrics": {}}
for m in METRICS:
    big = d[d[f"{m}_chg"].abs() >= THRESH[m]]
    slope_all = np.polyfit(d[f"{m}_chg"], d[f"{m}_kept"], 1)[0]
    res = {"threshold": THRESH[m], "n_big": int(len(big)), "slope_all": float(slope_all)}
    for direction, sub in [("gain", big[big[f"{m}_chg"] > 0]), ("loss", big[big[f"{m}_chg"] < 0])]:
        if len(sub) < 10:
            continue
        slope = (sub[f"{m}_kept"] * sub[f"{m}_chg"]).sum() / (sub[f"{m}_chg"] ** 2).sum()
        boots = []
        rng = np.random.default_rng(11)
        for _ in range(2000):
            s = sub.sample(len(sub), replace=True, random_state=rng.integers(1e9))
            boots.append((s[f"{m}_kept"] * s[f"{m}_chg"]).sum() / (s[f"{m}_chg"] ** 2).sum())
        res[direction] = {
            "n": int(len(sub)),
            "mean_change": float(sub[f"{m}_chg"].mean()),
            "mean_kept": float(sub[f"{m}_kept"].mean()),
            "persistence": float(slope),
            "persistence_ci90": [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))],
            "share_kept_half_or_more": float((sub[f"{m}_kept"] / sub[f"{m}_chg"] >= 0.5).mean()),
        }
    results["metrics"][m] = res

# Velocity split: fastballs vs everything else. A breaking ball losing velocity is often
# a deliberate shape change; a fastball losing velocity usually isn't.
results["velo_by_pitch_class"] = {}
is_fb = d["pitch_type"].isin(["FF", "SI", "FC"])
for cls, sub in [("fastball", d[is_fb]), ("non_fastball", d[~is_fb])]:
    for direction, s in [("gain", sub[sub.velo_chg >= THRESH["velo"]]), ("loss", sub[sub.velo_chg <= -THRESH["velo"]])]:
        results["velo_by_pitch_class"][f"{cls}_{direction}"] = {
            "n": int(len(s)),
            "persistence": float((s.velo_kept * s.velo_chg).sum() / (s.velo_chg ** 2).sum()),
        }

# Baseline: with no in-season change, how much does shape drift over an offseason anyway?
stable = d[(d.velo_chg.abs() < 0.3) & (d.ivb_chg.abs() < 0.5) & (d.hb_chg.abs() < 0.5)]
results["baseline_offseason_drift_sd"] = {m: float(stable[f"{m}_kept"].std()) for m in METRICS}
results["baseline_n"] = int(len(stable))
(OUT / "results.json").write_text(json.dumps(results, indent=2))
d.to_csv(OUT / "pitch_type_changes.csv", index=False)

# Figure: persistence by metric and direction
ps.apply()
fig, ax = plt.subplots(figsize=(7.5, 3.6))
labels, vals, los, his, colors = [], [], [], [], []
for m in METRICS:
    for direction, color in [("gain", ps.ORANGE), ("loss", ps.BLUE)]:
        r = results["metrics"][m].get(direction)
        if not r:
            continue
        word = {"velo": {"gain": "Velo gained", "loss": "Velo lost"},
                "ivb": {"gain": "IVB gained", "loss": "IVB lost"},
                "hb": {"gain": "Arm-side run gained", "loss": "Arm-side run lost"}}[m][direction]
        labels.append(f"{word}  (n={r['n']})")
        vals.append(r["persistence"] * 100)
        los.append(r["persistence_ci90"][0] * 100); his.append(r["persistence_ci90"][1] * 100)
        colors.append(color)
y = np.arange(len(labels))[::-1]
ax.hlines(y, los, his, color=ps.MUTED, lw=2)
ax.scatter(vals, y, color=colors, s=70, zorder=3, edgecolor=ps.SURFACE, lw=1.5)
for yi, v in zip(y, vals):
    ax.text(v, yi + 0.28, f"{v:.0f}%", ha="center", fontsize=9, color=ps.INK)
ax.set_yticks(y, labels)
ax.axvline(0, color=ps.MUTED, lw=1); ax.axvline(100, color=ps.MUTED, lw=1, ls="--")
ax.set_xlim(-10, 130); ax.set_ylim(-0.6, len(labels) - 0.3)
ax.set_xlabel("Share of the in-season change still there the next spring (90% CI)")
ax.grid(axis="y", visible=False)
ax.set_title("How much of a midseason shape change carries into next year", pad=16)
ps.save(fig, FIG / "persistence.png")

# Figure: velocity change vs what was kept, scatter
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, m in zip(axes, METRICS):
    ax.scatter(d[f"{m}_chg"], d[f"{m}_kept"], s=6, alpha=0.25, color=ps.BLUE, edgecolor="none")
    lim = np.nanpercentile(d[f"{m}_chg"].abs(), 99.5)
    xs = np.linspace(-lim, lim, 10)
    ax.plot(xs, xs, color=ps.MUTED, ls="--", lw=1, label="Fully kept")
    ax.plot(xs, results["metrics"][m]["slope_all"] * xs, color=ps.ORANGE, lw=2, label="Fitted")
    ax.set_xlim(-lim, lim); ax.set_ylim(-1.6 * lim, 1.6 * lim)
    ax.set_title(LABEL[m], fontsize=10)
    ax.set_xlabel("H1 → H2 change, league-adjusted")
axes[0].set_ylabel("Next season H1 vs this season H1")
axes[0].legend(loc="upper left", fontsize=8)
fig.suptitle("Every pitch type, 100+ pitches in each half (all changes, not just large ones)", x=0.01, ha="left",
             fontsize=11, fontweight="bold")
fig.tight_layout()
ps.save(fig, FIG / "change_vs_kept.png")
print(json.dumps(results, indent=2))
