"""Listing thumbnail for "Good offense vs. good defense": the offense tier x defense tier grid of average
game EPA/play, all plays (site/nfl/offense-or-defense/thumb.png). Reads nfl/offense_defense/output/results.json."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from sparvi import plotstyle as ps  # noqa: E402
from sparvi.config import ROOT  # noqa: E402

R = json.loads((ROOT / "nfl" / "offense_defense" / "output" / "results.json").read_text(encoding="utf-8"))
T = ["weak", "middle", "strong"]
cell = {(c["off"], c["def"]): c["epa"] for c in R["splits"]["all"]["grid"]}
m = np.array([[cell[(o, d)] for d in T] for o in reversed(T)])
lim = float(np.abs(m).max())

ps.apply()
fig, ax = plt.subplots(figsize=(4.8, 3.2))
cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#efeee9", "#eb6834"])
ax.imshow(m, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{m[i, j]:+.2f}", ha="center", va="center", fontsize=12, fontweight="bold", color="#16181d")
ax.set_xticks(range(3), [f"{t} D" for t in T])
ax.set_yticks(range(3), [f"{t} O" for t in reversed(T)])
ax.grid(False)
ax.set_title("Offense EPA/play, by matchup tier")
fig.savefig(ROOT / "site" / "nfl" / "offense-or-defense" / "thumb.png", metadata={"Software": None})
