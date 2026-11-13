"""Listing thumbnail for "Ranked vs. actually good": AP rank vs. power rating rank for the ranked teams
as of the latest week of the current season (site/cfb/ranked-vs-good/thumb.png)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sparvi import plotstyle as ps  # noqa: E402
from sparvi.config import ROOT, cfb  # noqa: E402

CFG = cfb()
d = json.loads((ROOT / "site" / "data" / "cfb" / f"team_week_{CFG.season}.json").read_text(encoding="utf-8"))
k = len(d["weeks"]) - 1
rows = [(t["series"]["no_gt"]["ap"][k], t["series"]["no_gt"]["rank"][k]) for t in d["teams"]
        if t["series"]["no_gt"]["ap"][k] is not None and t["series"]["no_gt"]["rank"][k] is not None]

ps.apply()
fig, ax = plt.subplots(figsize=(4.8, 3.2))
ax.plot([0.5, 25.5], [0.5, 25.5], ls="--", color=ps.MUTED, lw=1)
for ap, rk in rows:
    gap = rk - ap
    ax.scatter(ap, min(rk, 80), s=36, color=ps.BLUE if gap >= 10 else ps.ORANGE if gap <= -10 else "#c9c7c0", edgecolor="white", lw=0.6, zorder=3)
ax.set_xlim(0.5, 25.5)
ax.set_ylim(81, 0)
ax.set_xlabel("AP rank")
ax.set_ylabel("Power rating rank")
ax.set_title(f"Ranked vs. actually good, {CFG.season} week {d['weeks'][k]}")
fig.savefig(ROOT / "site" / "cfb" / "ranked-vs-good" / "thumb.png", metadata={"Software": None})
plt.close(fig)
print("thumb.png written")
