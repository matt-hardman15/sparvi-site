"""Generate the static NFL team pages: site/nfl/teams/<slug>/index.qmd and preview.png for each team.

One real page per team (rather than one page with ?team=) so each has its own <title>, description,
and social preview image, and appears in the sitemap. The page body is the shared template
site/nfl/teams/_team.qmd; this script writes only the front matter and the preview image.

Reads site/data/nfl (run after nfl/export_site_data.py). Matplotlib PNGs carry no timestamp, so
unchanged data gives identical files.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt

from sparvi import plotstyle as ps
from sparvi.config import ROOT, nfl

CFG = nfl()
DATA = ROOT / "site" / "data" / "nfl"
PAGES = ROOT / "site" / "nfl" / "teams"

teams = json.loads((DATA / "teams.json").read_text(encoding="utf-8"))
meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
wk = json.loads((DATA / f"team_week_{CFG.season}.json").read_text(encoding="utf-8"))
weeks = wk["weeks"]
series = {t["id"]: t["series"]["no_gt"] for t in wk["teams"]}

PAGE = """---
title: "{name}"
pagetitle: "{name}: offense and defense efficiency, week by week"
description: "{name} ({division}) offense and defense in EPA per play and success rate, week by week, with passing and rushing splits and every result, {first}–{last}."
image: preview.png
page-layout: full
---

```{{=html}}
<div id="team-root" data-team="{id}"></div>
```

{{{{< include ../_team.qmd >}}}}
"""


def rank_text(vals: dict[str, float], team: str, lower_better: bool) -> str:
    v = {k: x for k, x in vals.items() if x is not None}
    order = sorted(v, key=lambda k: v[k], reverse=not lower_better)
    r = order.index(team) + 1
    suf = "th" if 10 <= r % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(r % 10, "th")
    return f"{r}{suf} of {len(v)}"


ps.apply()
plt.rcParams["savefig.bbox"] = "standard"
last = len(weeks) - 1
off_now = {t: s["off_epa_adj"][last] for t, s in series.items()}
def_now = {t: s["def_epa_adj"][last] for t, s in series.items()}
for t in teams:
    d = PAGES / t["slug"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "index.qmd").write_text(PAGE.format(first=CFG.first_season, last=CFG.season, **t), encoding="utf-8", newline="\n")

    s = series[t["id"]]
    net = [None if o is None or x is None else o - x for o, x in zip(s["off_epa_adj"], s["def_epa_adj"], strict=True)]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    ax.axhline(0, color=ps.MUTED, lw=1, ls="--")
    pts = [(w, n) for w, n in zip(weeks, net, strict=True) if n is not None]
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#0f5257", lw=2.4, marker="o", ms=4)
    ax.set_xlabel("Week")
    ax.set_ylabel("Net EPA/play, adjusted")
    ax.set_xlim(0.5, max(weeks) + 0.5)
    ax.set_title(f"{t['name']} · {CFG.season} through week {weeks[-1]}", loc="left", fontsize=13, fontweight="bold")
    ax.text(0.0, -0.3, f"Offense {rank_text(off_now, t['id'], False)} · defense {rank_text(def_now, t['id'], True)} "
            "in opponent-adjusted EPA/play · sparvi.io", transform=ax.transAxes, fontsize=9, color=ps.MUTED)
    fig.subplots_adjust(bottom=0.27, top=0.88, left=0.11, right=0.97)
    fig.savefig(d / "preview.png", dpi=150, metadata={"Software": None})
    plt.close(fig)
print(f"{len(teams)} team pages written")
