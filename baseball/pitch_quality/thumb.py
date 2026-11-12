"""Listing thumbnail for the stuff-or-command article: qualified MLB pitchers, Stuff+ vs Location+.

Reads the exported season grades (site/data/baseball/stuff_pitchers.json), so run after export_site_data.py.
Matplotlib PNGs carry no timestamp, so unchanged data gives an identical file.
"""
import json

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from sparvi import plotstyle as ps
from sparvi.config import ROOT

DATA = ROOT / "site" / "data" / "baseball"
Q = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))["qualifiers"]["MLB"]
d = pd.DataFrame(json.loads((DATA / "stuff_pitchers.json").read_text(encoding="utf-8")))
d = d[(d["level"] == "MLB") & d["loc_plus"].notna() & d["rv_plus"].notna() & (d["pitches"] >= d["role"].map(Q))]

ps.apply()
plt.rcParams["savefig.bbox"] = "standard"  # fixed 1065x542, same as the other listing thumbnails
fig, ax = plt.subplots(figsize=(7.1, 3.613), dpi=150)
ax.axvline(100, color=ps.MUTED, lw=1, ls="--")
ax.axhline(100, color=ps.MUTED, lw=1, ls="--")
cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#d9d8d3", "#eb6834"])
ax.scatter(d["stuff_plus"], d["loc_plus"], c=d["rv_plus"].clip(70, 130), cmap=cmap, norm=TwoSlopeNorm(100, 70, 130),
           s=14, edgecolor="none", alpha=0.85)
ax.set_xlabel("Stuff+")
ax.set_ylabel("Location+")
ax.set_title("MLB pitchers, colored by results", loc="left")
fig.savefig(ROOT / "site" / "baseball" / "stuff-or-command" / "thumb.png", dpi=150, metadata={"Software": None})
plt.close(fig)
print("thumb written")
