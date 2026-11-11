"""Listing thumbnail for the Rule 5 article: board pitchers, Stuff+ vs K-BB%, top five highlighted.

Reads the exported board (site/data/baseball/rule5_pitchers.json), so run after export_site_data.py.
Matplotlib PNGs carry no timestamp, so unchanged data gives an identical file.
"""
import json

import matplotlib.pyplot as plt
import pandas as pd

from sparvi import plotstyle as ps
from sparvi.config import ROOT

QA = json.loads((ROOT / "site" / "data" / "baseball" / "meta.json").read_text(encoding="utf-8"))["qualifiers"]["AAA"]
d = pd.DataFrame(json.loads((ROOT / "site" / "data" / "baseball" / "rule5_pitchers.json").read_text(encoding="utf-8")))
d = d[d["stuff_plus"].notna() & (d["pitches"] >= d["role"].map(QA))]
top = d.nlargest(5, "stuff_plus")

ps.apply()
plt.rcParams["savefig.bbox"] = "standard"  # fixed 1065x542, same as the other listing thumbnails
fig, ax = plt.subplots(figsize=(7.1, 3.613), dpi=150)
ax.axvline(100, color=ps.MUTED, lw=1, ls="--")
ax.scatter(d["stuff_plus"], 100 * d["k_bb_pct"], s=14, color="#c9c7c0", edgecolor="none")
ax.scatter(top["stuff_plus"], 100 * top["k_bb_pct"], s=46, color=ps.ORANGE, edgecolor="white", lw=1.2, zorder=3)
ax.set_xlabel("Stuff+")
ax.set_ylabel("K−BB%")
ax.set_title("Unprotected Triple-A pitchers", loc="left")
fig.savefig(ROOT / "site" / "baseball" / "rule-5" / "thumb.png", dpi=150, metadata={"Software": None})
plt.close(fig)
print("thumb written")
