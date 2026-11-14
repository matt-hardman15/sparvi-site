"""Generate the static college football team pages: site/cfb/teams/<slug>/index.qmd and preview.png for
every team in site/data/cfb/teams.json (each FBS team since the first season, including teams that have
since left FBS, whose pages open on their last FBS season).

One real page per team (rather than one page with ?team=) so each has its own <title>, description,
social preview image, and sitemap entry. The page body is the shared template site/cfb/teams/_team.qmd;
this script writes the front matter, a one-paragraph summary from the data (so the page says something
without JavaScript), and the preview image.

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


def ordinal(r: int) -> str:
    return f"{r}{'th' if 10 <= r % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(r % 10, 'th')}"


def signed(x: float) -> str:
    return f"{x:+.1f}".replace("-", "−")


def fit(options: list[str], lo: int, hi: int, what: str) -> str:
    for s in options:
        if lo <= len(s) <= hi:
            return s
    raise ValueError(f"no {what} fits {lo}-{hi} characters: {options}")


def yaml_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def summary(t: dict, season: int) -> tuple[str, dict]:
    """One paragraph from the team's latest season, plus the numbers the preview image uses."""
    wf, gf = load(season)
    k = len(wf["weeks"]) - 1
    me = next(x for x in wf["teams"] if x["id"] == t["id"])["series"]["no_gt"]
    played = [g for g in gf["teams"].get(t["id"], []) if g["res"]]
    rec = [sum(g["res"] == r for g in played) for r in "WLT"]
    record = f"{rec[0]}–{rec[1]}" + (f"–{rec[2]}" if rec[2] else "")
    n = sum(x["series"]["no_gt"]["power"][k] is not None for x in wf["teams"])
    power, rank, ap = me["power"][k], me["rank"][k], me["ap"][k]
    current = season == CFG.season
    when = (f"Through week {wf['weeks'][k]} of {season}," if current
            else f"In {season}" + ("" if season == meta["seasons"][-1] else ", its last FBS season,"))
    verb = "is" if current else "went"
    s = f"{when} {t['name']} {verb} {record}."
    if power is not None:
        s += f" {'Its' if current else 'Its final'} power rating, {signed(power)} points against an average FBS team, " \
             f"{'ranks' if current else 'ranked'} {ordinal(rank)} of {n}"
        s += (f"; the AP poll {'has' if current else 'had'} it {ordinal(ap)}." if ap else
              f"; {'it is' if current else 'it was'} unranked in the AP poll.")
    return s, {"record": record, "power": power, "rank": rank, "n": n, "week": wf["weeks"][k], "current": current}


PAGE = """---
title: {title}
pagetitle: {pagetitle}
description: {description}
image: preview.png
page-layout: full
---

{summary}

```{{=html}}
<div id="team-root" data-team="{id}"></div>
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
    text, num = summary(t, last)
    d = PAGES / t["slug"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "index.qmd").write_text(PAGE.format(
        title=yaml_str(f"{name} {t['nick']}" if t.get("nick") else name), pagetitle=yaml_str(pagetitle),
        description=yaml_str(description), summary=text, id=html.escape(t["id"], quote=True)), encoding="utf-8", newline="\n")

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
        foot += f" · power rating {signed(num['power'])}, {ordinal(num['rank'])} of {num['n']} FBS teams"
    ax.text(0.0, -0.3, foot + " · sparvi.io", transform=ax.transAxes, fontsize=9, color=ps.MUTED)
    fig.subplots_adjust(bottom=0.27, top=0.88, left=0.11, right=0.97)
    # A 64-color palette PNG: a quarter of the size, and the weekly refresh commits all of these.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, metadata={"Software": None})
    plt.close(fig)
    Image.open(buf).convert("RGB").quantize(64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(
        d / "preview.png", optimize=True)
print(f"{len(teams)} CFB team pages written")
