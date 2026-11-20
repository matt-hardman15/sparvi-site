"""Generate the static college football team pages: site/cfb/teams/<slug>/index.qmd and preview.png for
every team in site/data/cfb/teams.json (each FBS team since the first season, including teams that have
since left FBS, whose pages open on their last FBS season).

One real page per team (rather than one page with ?team=) so each has its own <title>, description,
social preview image, and sitemap entry. The page body is the shared template site/cfb/teams/_team.qmd;
this script writes the front matter, a summary from the data, the season's numbers as plain HTML
(cfb/team_summary.py; the charts replace them), links to the conference's other teams, the preview image,
and the /cfb/teams/ index (PLAN-seo S2 standard, BACKLOG SEO-2b).

Reads site/data/cfb (run after cfb/export_site_data.py). Matplotlib PNGs carry no timestamp, so
unchanged data gives identical files.
"""
from __future__ import annotations

import html
import io
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from cfb.team_summary import League, power_rank, power_str, static_block, summary  # noqa: E402
from sparvi import plotstyle as ps  # noqa: E402
from sparvi.config import ROOT, cfb  # noqa: E402

CFG = cfb()
DATA = ROOT / "site" / "data" / "cfb"
PAGES = ROOT / "site" / "cfb" / "teams"
TITLE_MAX = 60 - len(" – Sparvi")     # site/build/seo.py checks the full <title>
DESC = (120, 155)

teams = json.loads((DATA / "teams.json").read_text(encoding="utf-8"))
meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
week_files, game_files = {}, {}


def load(season: int):
    if season not in week_files:
        week_files[season] = json.loads((DATA / f"team_week_{season}.json").read_text(encoding="utf-8"))
        game_files[season] = json.loads((DATA / f"team_games_{season}.json").read_text(encoding="utf-8"))
    return week_files[season], game_files[season]


def fit(options: list[str], lo: int, hi: int, what: str) -> str:
    for s in options:
        if lo <= len(s) <= hi:
            return s
    raise ValueError(f"no {what} fits {lo}-{hi} characters: {options}")


def yaml_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


leagues: dict[int, League] = {}


def league(season: int) -> League:
    if season not in leagues:
        wf, gf = load(season)
        leagues[season] = League(wf, gf, teams)
    return leagues[season]


def href(t: dict) -> str:
    return f"/cfb/teams/{t['slug']}/"


PAGE = """---
title: {title}
pagetitle: {pagetitle}
description: {description}
image: preview.png
page-layout: full
---

```{{=html}}
<div id="team-root" data-team="{id}"></div>
<p class="page-summary">{summary}</p>
<p class="note team-sub" style="border-left:4px solid {color};padding-left:.6rem">{links}</p>
<!-- The season's numbers as plain HTML; the interactive page below removes this once it runs. -->
<div id="team-static">
{static}
</div>
```

{{{{< include ../_team.qmd >}}}}
"""

ps.apply()
plt.rcParams["savefig.bbox"] = "standard"
for t in teams:
    seasons = [s for s in t["seasons"] if s in meta["seasons"]]
    last, first = seasons[-1], seasons[0]
    span = f"{first}–{last}" if first != last else str(first)
    name, conf = t["name"], t["conf"]
    pagetitle = fit([f"{name} football: power rating by week", f"{name}: power rating by week", f"{name} football ratings"],
                    1, TITLE_MAX, f"title for {name}")
    description = fit([
        f"{name} ({conf}) college football power rating and opponent-adjusted offense and defense, week by week, with AP and CFP ranks and every game, {span}.",
        f"{name} ({conf}) power rating and opponent-adjusted offense and defense, week by week, with AP and CFP ranks and every game, {span}.",
        f"{name} power rating and opponent-adjusted offense and defense, week by week, with AP and CFP ranks and every game, {span}.",
        f"{name} football: power rating, adjusted offense and defense, AP and CFP ranks, and every game result by season, {span}.",
    ], *DESC, f"description for {name}")
    L = league(last)
    current = last == CFG.season
    text, num = summary(L, t, last, CFG.season, meta["seasons"][-1])
    mates = [x for x in teams if x["conf"] == t["conf"] and x["id"] != t["id"] and CFG.season in x["seasons"]]
    links = (f"{html.escape(t['conf'])}: " + " · ".join(f'<a href="{href(x)}">{html.escape(x["name"])}</a>' for x in mates) + " · "
             if mates and current else "")
    links += (f'<a href="/cfb/teams/">All FBS teams</a> · '
              f'<a href="/cfb/tools/team-ratings.html?team={html.escape(t["id"], quote=True)}">Compare with every FBS team</a>')
    d = PAGES / t["slug"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "index.qmd").write_text(PAGE.format(
        title=yaml_str(f"{name} {t['nick']}" if t.get("nick") else name), pagetitle=yaml_str(pagetitle),
        description=yaml_str(description), summary=html.escape(text), id=html.escape(t["id"], quote=True),
        color=t.get("color") or "#0f5257", links=links, static=static_block(L, t, last, current, href)),
        encoding="utf-8", newline="\n")

    wf, _ = load(last)
    me = next(x for x in wf["teams"] if x["id"] == t["id"])["series"]["no_gt"]
    pts = [(w, p) for w, p in zip(wf["weeks"], me["power"], strict=True) if p is not None]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    ax.axhline(0, color=ps.MUTED, lw=1, ls="--")
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#0f5257", lw=2.4, marker="o", ms=4)
    ax.set_xlabel("Week")
    ax.set_ylabel("Power rating (points vs. average FBS)")
    ax.set_xlim(0.5, max(wf["weeks"]) + 0.5)
    through = f"through week {num['week']}" if num["current"] else "final"
    ax.set_title(f"{name} · {last} {through}", loc="left", fontsize=13, fontweight="bold")
    foot = f"Record {num['record']}"
    if num["power"] is not None:
        foot += f" · power rating {power_str(num['power'])}, {num['rank_str']} of {num['n']} FBS teams"
    ax.text(0.0, -0.3, foot + " · sparvi.io", transform=ax.transAxes, fontsize=9, color=ps.MUTED)
    fig.subplots_adjust(bottom=0.27, top=0.88, left=0.11, right=0.97)
    # A 64-color palette PNG: a quarter of the size, and the weekly refresh commits all of these.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, metadata={"Software": None})
    plt.close(fig)
    Image.open(buf).convert("RGB").quantize(64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(
        d / "preview.png", optimize=True)
print(f"{len(teams)} CFB team pages written")

# The /cfb/teams/ index: this season's FBS teams by conference (record and power rank), then teams that left FBS.
L = league(CFG.season)
now = [t for t in teams if CFG.season in t["seasons"]]
confs: dict[str, list] = {}
for t in now:
    confs.setdefault(t["conf"], []).append(t)
item = lambda t: (f'<li><a href="{href(t)}">{html.escape(t["name"])}</a> <span class="muted">{L.record(t["id"])}'  # noqa: E731
                  + (f' · {power_rank(L, t["id"], short=True)} in power rating' if L.val(t["id"], "rank") else "") + "</span></li>")
blocks = [f'<h2>{html.escape(c)}</h2>\n<ul class="team-index">{"".join(item(t) for t in sorted(confs[c], key=lambda t: t["name"]))}</ul>'
          for c in sorted(confs)]
gone = sorted((t for t in teams if CFG.season not in t["seasons"]), key=lambda t: t["name"])
if gone:
    blocks.append('<h2>Former FBS teams</h2>\n<ul class="team-index">' + "".join(
        f'<li><a href="{href(t)}">{html.escape(t["name"])}</a> <span class="muted">FBS {t["seasons"][0]}–{t["seasons"][-1]}</span></li>'
        for t in gone) + "</ul>")
INDEX = """---
no-chart-libs: true
title: "College football teams"
pagetitle: "College football team ratings: every FBS team"
description: "Every FBS team's power rating and opponent-adjusted offense and defense, week by week, with AP and CFP ranks and every game, {first}–{last}."
page-layout: article
---

Each team page shows its power rating and opponent-adjusted offense and defense week by week, with AP and CFP ranks and every game since {first}. Records and power-rating ranks below are through week {week} of {last}. To compare every team at once, use the [team ratings tool](/cfb/tools/team-ratings.html).

```{{=html}}
{blocks}
```
"""
(PAGES / "index.qmd").write_text(INDEX.format(first=meta["seasons"][0], last=CFG.season, week=L.week, blocks="\n".join(blocks)),
                                 encoding="utf-8", newline="\n")
print("CFB team index written")

