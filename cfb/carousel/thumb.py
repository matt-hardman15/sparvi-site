"""Listing thumbnail for the coaching-change article: year-1 change in power rating against expectation, by why the
coach changed, with 90% intervals (site/cfb/coaching-changes/thumb.png). Reads cfb/carousel/output/results.json."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sparvi import plotstyle as ps  # noqa: E402
from sparvi.config import ROOT  # noqa: E402

R = json.loads((ROOT / "cfb" / "carousel" / "output" / "results.json").read_text(encoding="utf-8"))["summary"]
rows = [("Kept coach", "kept"), ("Coach left\nfor a job", "left_for_job"), ("Coach fired /\nother", "did_not_leave_for_job"),
        ("After a bad\nseason", "bad_last_year")]

ps.apply()
fig, ax = plt.subplots(figsize=(4.8, 3.2))
for i, (_, k) in enumerate(rows):
    v = R[k]
    color = "#2a78d6" if v["ci90"][1] < 0 else ps.MUTED
    ax.plot([i, i], v["ci90"], color=color, lw=2)
    ax.plot(i, v["mean"], "o", color=color, ms=7)
ax.axhline(0, color=ps.MUTED, lw=1, ls="--")
ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
ax.set_ylabel("Points vs. expected")
ax.set_title("Year 1 after a coaching change")
(ROOT / "site" / "cfb" / "coaching-changes").mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / "site" / "cfb" / "coaching-changes" / "thumb.png", metadata={"Software": None})
