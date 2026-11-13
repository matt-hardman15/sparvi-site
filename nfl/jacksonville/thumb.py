"""Listing thumbnail for "Is Jacksonville the best team in football?": the top ten in opponent-adjusted
net EPA/play as of the article's week (site/nfl/is-jacksonville-the-best/thumb.png)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from sparvi import plotstyle as ps  # noqa: E402
from sparvi.config import ROOT  # noqa: E402

R = json.loads((ROOT / "nfl" / "jacksonville" / "output" / "results.json").read_text(encoding="utf-8"))
a = pd.read_csv(ROOT / "nfl" / "team_efficiency" / "output" / "adjusted_team_week.csv")
s = a[(a["season"] == R["season"]) & (a["week"] == R["week"]) & (a["gt"] == "no_gt") & (a["split"] == "all")].copy()
s["net"] = s["off_adj"] - s["def_adj"]
top = s.sort_values("net", ascending=False).head(10).iloc[::-1]
ps.apply()
fig, ax = plt.subplots(figsize=(4.8, 3.2))
ax.barh(top["team"], top["net"], color=["#0f5257" if t == R["team"] else "#c9c7c0" for t in top["team"]])
ax.set_xlabel("Net EPA/play, opponent-adjusted")
ax.set_title(f"NFL through week {R['week']}, {R['season']}")
ax.grid(axis="y", visible=False)
fig.savefig(ROOT / "site" / "nfl" / "is-jacksonville-the-best" / "thumb.png", metadata={"Software": None})
plt.close(fig)
print("thumb.png written")
