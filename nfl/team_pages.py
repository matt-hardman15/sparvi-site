"""Generate the static NFL team pages: site/nfl/teams/<slug>/index.qmd and preview.png for each team.

One real page per team (rather than one page with ?team=) so each has its own <title>, description,
and social preview image, and appears in the sitemap. The page body is the shared template
site/nfl/teams/_team.qmd; this script writes the front matter, a rule-based summary, the season's numbers
as plain HTML (nfl/team_summary.py; the charts replace them), links to the division rivals, the preview image,
and the /nfl/teams/ index (PLAN-seo S2 standard, BACKLOG SEO-2b).

Reads site/data/nfl (run after nfl/export_site_data.py). Matplotlib PNGs carry no timestamp, so
unchanged data gives identical files.
"""
from __future__ import annotations

import html
import json

import matplotlib.pyplot as plt

from nfl.team_summary import League, ordinal, static_block, summary
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
games = json.loads((DATA / f"team_games_{CFG.season}.json").read_text(encoding="utf-8"))
L = League(wk, games, teams)
href = lambda t: f"/nfl/teams/{t['slug']}/"  # noqa: E731

PAGE = """---
title: "{name}"
pagetitle: "{name} efficiency: EPA by week"
description: "{name} ({division}) offense and defense in EPA per play and success rate, week by week, with pass and run splits and results, {first}–{last}."
image: preview.png
page-layout: full
---

```{{=html}}
<div id="team-root" data-team="{id}"></div>
<p class="page-summary">{summary}</p>
<p class="note team-sub" style="border-left:4px solid {color};padding-left:.6rem">{division}: {rivals} · <a href="/nfl/teams/">All 32 teams</a> · <a href="/nfl/tools/team-efficiency.html?team={id}">Compare with every team</a></p>
<!-- The season's numbers as plain HTML; the interactive page below removes this once it runs. -->
<div id="team-static">
{static}
</div>
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
    rivals = " · ".join(f'<a href="{href(r)}">{html.escape(r["name"])}</a>' for r in teams
                        if r["division"] == t["division"] and r["id"] != t["id"])
    (d / "index.qmd").write_text(PAGE.format(first=CFG.first_season, last=CFG.season, rivals=rivals,
                                             summary=html.escape(summary(L, t["id"], CFG.season)),
                                             static=static_block(L, t["id"], CFG.season, href), **t),
                                 encoding="utf-8", newline="\n")

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

# The /nfl/teams/ index: every team by division, with its record and net rank (the hub's listing skips teams/**).
divs: dict[str, list] = {}
for t in teams:
    divs.setdefault(t["division"], []).append(t)
rn = {t["id"]: L.rank(t["id"], L.net)[0] for t in teams}
blocks = []
for dv in sorted(divs):
    items = "".join(f'<li><a href="{href(t)}">{html.escape(t["name"])}</a> <span class="muted">{L.record(t["id"])} · '
                    f'{"—" if rn[t["id"]] is None else ordinal(rn[t["id"]])} in net EPA/play</span></li>'
                    for t in sorted(divs[dv], key=lambda t: t["name"]))
    blocks.append(f'<h2>{dv}</h2>\n<ul class="team-index">{items}</ul>')
INDEX = """---
no-chart-libs: true
title: "NFL teams"
pagetitle: "NFL team efficiency pages: all 32 teams"
description: "Every NFL team's offense and defense in opponent-adjusted EPA per play and success rate, week by week, {first}–{last}, with pass and run splits."
page-layout: article
---

Each team page shows its offense and defense week by week in EPA per play and success rate, opponent-adjusted, with pass and run splits and every game. Records and ranks below are through week {week} of {last}; net EPA per play is offense minus defense allowed, opponent-adjusted. To compare all 32 at once, use the [team efficiency tool](/nfl/tools/team-efficiency.html).

```{{=html}}
{blocks}
```
"""
(PAGES / "index.qmd").write_text(INDEX.format(first=CFG.first_season, last=CFG.season, week=L.week, blocks="\n".join(blocks)),
                                 encoding="utf-8", newline="\n")
print("NFL team index written")

