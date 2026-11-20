"""Listing thumbnail for the fourth-down article: the go / toss-up / kick chart by field position and yards to go
(site/nfl/fourth-downs/thumb.png). Reads the exported site/data/nfl/fourth_downs.json."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from sparvi import plotstyle as ps  # noqa: E402
from sparvi.config import ROOT  # noqa: E402

D = json.loads((ROOT / "site" / "data" / "nfl" / "fourth_downs.json").read_text(encoding="utf-8"))
CODE = {"kick": 0, "toss-up": 1, "go": 2, "none": np.nan}    # "none": the own 20 and in, where the chart makes no call
ys = sorted({g["y"] for g in D["grid"]}, reverse=True)      # own goal line at the left: the offense moves left to right
ts = sorted({g["t"] for g in D["grid"]})
m = np.full((len(ts), len(ys)), np.nan)
for g in D["grid"]:
    m[ts.index(g["t"]), ys.index(g["y"])] = CODE[g["call"]]

ps.apply()
fig, ax = plt.subplots(figsize=(4.8, 3.2))
cmap = ListedColormap(["#2a78d6", "#d9d8d3", "#eb6834"])
cmap.set_bad("#ffffff")
ax.imshow(m, cmap=cmap, vmin=0, vmax=2, aspect="auto", origin="lower", extent=(ys[0] + 0.5, ys[-1] - 0.5, ts[0] - 0.5, ts[-1] + 0.5))
ax.set_xticks([90, 70, 50, 30, 10], ["own 10", "own 30", "50", "opp. 30", "opp. 10"])
ax.set_xlabel("Field position")
ax.set_yticks([1, 4, 7, 10])
ax.set_ylabel("Yards to go")
ax.grid(False)
ax.set_title("4th down: go (orange), kick (blue), toss-up")
fig.savefig(ROOT / "site" / "nfl" / "fourth-downs" / "thumb.png", metadata={"Software": None})
