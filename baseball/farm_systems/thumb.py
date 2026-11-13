"""Listing thumbnail for the farm-systems article: org Stuff+ change, first season vs. second.

Reads the exported org scores (site/data/baseball/farm_orgs.json), so run after export_site_data.py.
Matplotlib PNGs carry no timestamp, so unchanged data gives an identical file.
"""
import json

import matplotlib.pyplot as plt
import pandas as pd

from sparvi import plotstyle as ps
from sparvi.config import ROOT

d = pd.DataFrame(json.loads((ROOT / "site" / "data" / "baseball" / "farm_orgs.json").read_text(encoding="utf-8")))
d = d[d["metric"] == "stuff_plus"].pivot(index="org", columns="season", values="raw").dropna()
y0, y1 = d.columns.min(), d.columns.max()
lim = 1.15 * d.abs().to_numpy().max()

ps.apply()
plt.rcParams["savefig.bbox"] = "standard"  # fixed 1065x542, same as the other listing thumbnails
fig, ax = plt.subplots(figsize=(7.1, 3.613), dpi=150)
ax.axhline(0, color=ps.MUTED, lw=1)
ax.axvline(0, color=ps.MUTED, lw=1)
ax.scatter(d[y0], d[y1], s=40, color="#0f5257", alpha=0.8, edgecolor="white", lw=1)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel(f"{y0} org Stuff+ change vs. league")
ax.set_ylabel(f"{y1}")
ax.set_title("Farm systems: in-season Stuff+ gains, year to year", loc="left")
fig.savefig(ROOT / "site" / "baseball" / "farm-systems" / "thumb.png", dpi=150, metadata={"Software": None})
plt.close(fig)
print("thumb written")
