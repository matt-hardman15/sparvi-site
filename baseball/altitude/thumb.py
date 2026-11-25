"""Listing thumbnail for the altitude article: the share of MLB movement each Triple-A park records, the five
high-altitude parks marked (site/baseball/triple-a-altitude/thumb.png). Reads site/data/baseball/altitude.json."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sparvi import plotstyle as ps  # noqa: E402
from sparvi.config import ROOT  # noqa: E402

D = json.loads((ROOT / "site" / "data" / "baseball" / "altitude.json").read_text(encoding="utf-8"))
P = sorted(D["parks"], key=lambda p: p["factor"])

ps.apply()
fig, ax = plt.subplots(figsize=(4.8, 3.2))
for i, p in enumerate(P):
    c = "#0f5257" if p["altitude"] else "#b9b8b3"
    ax.plot([100 * p["lo"], 100 * p["hi"]], [i, i], color=c, lw=2)
    ax.scatter(100 * p["factor"], i, s=22, color=c, edgecolor="white", lw=0.5, zorder=3)
ax.axvline(100, color=ps.MUTED, lw=1, ls="--")
ax.set_yticks([])
ax.set_xlabel("Share of MLB movement recorded (%)")
ax.set_title("Triple-A parks: five read less movement")
fig.savefig(ROOT / "site" / "baseball" / "triple-a-altitude" / "thumb.png", metadata={"Software": None})
