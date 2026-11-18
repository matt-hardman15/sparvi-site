"""Listing thumbnail for "Did the committee pick the best 12?": the best-rated teams without an
automatic bid at selection, colored by whether the committee took them
(site/cfb/did-the-committee-pick-the-best-12/thumb.png)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sparvi import plotstyle as ps  # noqa: E402
from sparvi.config import ROOT  # noqa: E402

R = json.loads((ROOT / "cfb" / "cfp" / "output" / "results.json").read_text(encoding="utf-8"))
cs = R["season"]
auto = {r["team"] for r in cs["rows"] if r["bid"] == "auto"}
rows = [t for t in cs["top"] if t["team"] not in auto][: cs["at_large_spots"] + 3][::-1]
out = {x["team"] for x in cs["left_out"]}
ps.apply()
fig, ax = plt.subplots(figsize=(4.8, 3.2))
ax.barh([t["team"] for t in rows], [t["power"] for t in rows],
        color=["#eb6834" if t["team"] in out else "#0f5257" if t["in_field"] else "#c9c7c0" for t in rows])
ax.set_xlabel("Power rating at selection (points)")
ax.set_title(f"{cs['season']} playoff: at-large candidates")
ax.grid(axis="y", visible=False)
fig.savefig(ROOT / "site" / "cfb" / "did-the-committee-pick-the-best-12" / "thumb.png", metadata={"Software": None})
plt.close(fig)
print("thumb.png written")
