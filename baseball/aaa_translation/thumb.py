"""Listing thumbnail for the Triple-A -> MLB translation article: how well each hitter number carries.

Reads the exported results (site/data/baseball/translation_results.json), so run after export_site_data.py.
Matplotlib PNGs carry no timestamp, so unchanged data gives an identical file.
"""
import json

import matplotlib.pyplot as plt

from sparvi import plotstyle as ps
from sparvi.config import ROOT

DATA = ROOT / "site" / "data" / "baseball"
R = json.loads((DATA / "translation_results.json").read_text(encoding="utf-8"))
NAMES = {"chase_pct": "Chase rate", "zone_swing_pct": "Zone swing", "zone_contact_pct": "Zone contact", "whiff_pct": "Whiff rate",
         "k_pct": "Strikeout rate", "bb_pct": "Walk rate", "hard_hit_pct": "Hard-hit rate", "ev90": "EV90", "xwoba": "xwOBA",
         "woba": "wOBA"}
RESULTS = {"woba", "xwoba"}
m = sorted(R["hitter"]["cohorts"]["pooled"]["metrics"].items(), key=lambda kv: kv[1]["carryover_r"])

ps.apply()
plt.rcParams["savefig.bbox"] = "standard"  # fixed 1065x542, same as the other listing thumbnails
fig, ax = plt.subplots(figsize=(7.1, 3.613), dpi=150)
ax.axvline(0, color=ps.MUTED, lw=1)
for y, (k, v) in enumerate(m):
    c = ps.MUTED if k in RESULTS else "#0f5257"
    ax.plot(v["carryover_ci90"], [y, y], color=c, lw=2)
    ax.scatter([v["carryover_r"]], [y], color=c, s=36, zorder=3)
ax.set_yticks(range(len(m)), [NAMES[k] for k, _ in m])
ax.set_xlim(-0.2, 1)
ax.set_xlabel("Correlation, Triple-A to MLB")
ax.set_title("Hitters: what carries to the majors", loc="left")
fig.subplots_adjust(left=0.2, right=0.97, bottom=0.16, top=0.9)
fig.savefig(ROOT / "site" / "baseball" / "aaa-translation" / "thumb.png", dpi=150, metadata={"Software": None})
plt.close(fig)
print("thumb written")
