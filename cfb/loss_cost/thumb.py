"""Listing thumbnail for "The committee punishes the loss, not the margin": average spots dropped after a
loss by a top-12 team, by margin, in the committee's ranking and in the power rating
(site/cfb/what-a-loss-costs/thumb.png). Reads cfb/loss_cost/output/results.json."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sparvi import plotstyle as ps  # noqa: E402
from sparvi.config import ROOT  # noqa: E402

R = json.loads((ROOT / "cfb" / "loss_cost" / "output" / "results.json").read_text(encoding="utf-8"))
labels = [f"{b['lo']}–{b['hi']}" if b["hi"] < 50 else f"{b['lo']}+" for b in R["by_margin"]]
x = range(len(labels))

ps.apply()
fig, ax = plt.subplots(figsize=(4.8, 3.2))
ax.plot(x, [b["drop"] for b in R["by_margin"]], color="#0f5257", lw=2.4, marker="o", label="Committee ranking")
ax.plot(x, [b["power_drop"] for b in R["by_margin"]], color=ps.MUTED, lw=2.4, marker="o", ls="--", label="Power rating")
ax.set_xticks(list(x), labels)
ax.set_ylim(0, None)
ax.set_xlabel("Lost by (points)")
ax.set_ylabel("Spots dropped")
ax.legend(frameon=False, loc="upper left")
ax.set_title("What a top-12 loss costs")
fig.savefig(ROOT / "site" / "cfb" / "what-a-loss-costs" / "thumb.png", metadata={"Software": None})
