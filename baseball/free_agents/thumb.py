"""Listing thumbnail for "This winter's free-agent pitchers": results (RV+) against Pitching+ for every free-agent
pitcher in the article, colored by whether results trailed or ran ahead of the grades
(site/baseball/free-agent-pitchers/thumb.png). Reads site/data/baseball/free_agents.json."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sparvi import plotstyle as ps  # noqa: E402
from sparvi.config import ROOT  # noqa: E402

D = json.loads((ROOT / "site" / "data" / "baseball" / "free_agents.json").read_text(encoding="utf-8"))
P, gap = D["pitchers"], D["history"]["gap"]
lo = min(min(r["pitching_plus"], r["rv_plus"]) for r in P) - 5
hi = max(max(r["pitching_plus"], r["rv_plus"]) for r in P) + 5

ps.apply()
fig, ax = plt.subplots(figsize=(4.8, 3.2))
ax.plot([lo, hi], [lo, hi], color=ps.MUTED, lw=1)
for d in (-gap, gap):
    ax.plot([lo, hi], [lo + d, hi + d], color="#c9c7c0", lw=1, ls="--")
color = {"trailed": ps.ORANGE, "ahead": ps.BLUE, "": "#b9b8b3"}
for r in sorted(P, key=lambda r: r["flag"] != ""):
    ax.scatter(r["pitching_plus"], r["rv_plus"], s=26, color=color[r["flag"]], edgecolor="white", lw=0.5, zorder=3)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("Pitching+")
ax.set_ylabel("Results (RV+)")
ax.set_title("Free-agent pitchers: results vs. grades")
fig.savefig(ROOT / "site" / "baseball" / "free-agent-pitchers" / "thumb.png", metadata={"Software": None})
