"""Data-driven summary sentences and static tables for player pages (PLAN-seo S2).

Everything here is built by rule from the player's exported numbers: no free text, no model at build
time. The page's static HTML (what a crawler or a reader without JavaScript sees) and the summary read
the same values, the final season-to-date snapshot, which is also what the interactive page opens on.
tests/test_player_pages.py checks every number in a summary appears in that page's tables.

Standard library only: this runs as a Quarto post-render step (see player_pages.py).
"""
from __future__ import annotations

import html
import zlib

LEVEL = {"AAA": "Triple-A", "MLB": "MLB"}
PITCH = {"FF": "four-seam fastball", "SI": "sinker", "FC": "cutter", "SL": "slider", "ST": "sweeper",
         "CU": "curveball", "KC": "knuckle curve", "CH": "changeup", "FS": "splitter", "SV": "slurve",
         "KN": "knuckleball", "FO": "forkball", "SC": "screwball", "CS": "slow curve", "EP": "eephus", "FA": "fastball"}
ROLE = {"SP": "as a starter", "RP": "in relief", "Swing": "as a swingman"}
NOTABLE_HI, NOTABLE_LO, BEST_PITCH_MIN, BEST_PITCH_N = 90, 10, 110, 100

# Thresholds for an indexed page (PLAN-seo S2). Below them a player keeps a page (tools link to every
# player) but it is noindex and left out of the sitemap.
MIN_PA, MIN_BF, MIN_MLB_PITCHES = 25, 25, 100


# ---- formatting: the same output as Sparvi.fmt in sparvi.js ------------------------------------
def pct(v, d=1):
    return "—" if v is None else f"{100 * v:.{d}f}%"


def num(v, d=1):
    return "—" if v is None else f"{v:.{d}f}"


def intc(v):
    return "—" if v is None else f"{round(v):,}"


def woba(v):
    if v is None:
        return "—"
    s = f"{v:.3f}"
    return s[1:] if s.startswith("0") else s


def ordinal(n: int) -> str:
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


# (label, field, formatter, noun used in a sentence)
HIT_METRICS = [("xwOBA", "xwoba", woba, "xwOBA"), ("EV90", "ev90", lambda v: num(v) + " mph", "EV90"),
               ("Hard-hit rate", "hard_hit_pct", pct, "hard-hit rate"), ("Chase rate", "chase_pct", pct, "chase rate"),
               ("Zone swing", "zone_swing_pct", pct, "zone-swing rate"), ("Zone contact", "zone_contact_pct", pct, "zone-contact rate"),
               ("Whiff rate", "whiff_pct", pct, "whiff rate"), ("Strikeout rate", "k_pct", pct, "strikeout rate"),
               ("Walk rate", "bb_pct", pct, "walk rate")]
PIT_METRICS = [("Strikeout rate", "k_pct", pct, "strikeout rate"), ("Walk rate", "bb_pct", pct, "walk rate"),
               ("CSW rate", "csw_pct", pct, "CSW rate"), ("Whiff rate", "whiff_pct", pct, "whiff rate"),
               ("Chase induced", "chase_pct", pct, "induced chase rate"), ("Hard-hit allowed", "hard_hit_pct", pct, "hard-hit rate allowed"),
               ("xwOBA allowed", "xwoba", woba, "xwOBA allowed")]


def at_end(series: dict | None, field: str):
    """Last non-missing value of a snapshot series (the final season-to-date snapshot)."""
    vals = (series or {}).get(field) or []
    return vals[-1] if vals else None


def final(player: dict) -> dict:
    """The final std snapshot per section: {'hitter': {...}, 'AAA': {...}, 'MLB': {...}} plus arsenals."""
    tl, out = player.get("timeline") or {}, {}
    hs = ((tl.get("AAA") or {}).get("hitter") or {}).get("std")
    if hs and at_end(hs, "pa") is not None:
        out["hitter"] = {k: at_end(hs, k) for k in hs}
    for lvl in ("AAA", "MLB"):
        t = tl.get(lvl) or {}
        ps = (t.get("pitcher") or {}).get("std")
        if ps and at_end(ps, "pitches") is not None:
            row = {k: at_end(ps, k) for k in ps}
            ars = [{"pt": p["pt"], **{k: at_end(p["series"], k) for k in p["series"]}} for p in t.get("pitch_types") or []]
            row["arsenal"] = sorted((a for a in ars if a.get("pitches") is not None),
                                    key=lambda a: (-(a.get("usage_pct") or 0), a["pt"]))
            out[lvl] = row
    return out


def roles(fin: dict) -> list[str]:
    """The sections with a real sample, in page order: 'hitter', 'AAA' (pitching), 'MLB' (pitching).
    A position player's mop-up inning doesn't make him a pitcher here."""
    return [k for k, f, n in (("hitter", "pa", MIN_PA), ("AAA", "bf", MIN_BF), ("MLB", "pitches", MIN_MLB_PITCHES))
            if ((fin.get(k) or {}).get(f) or 0) >= n]


def indexable(fin: dict) -> bool:
    return bool(roles(fin))


def teams(player: dict, fin: dict) -> list[str]:
    out = []
    if "hitter" in fin and fin["hitter"].get("team"):
        out.append(f"{fin['hitter']['team']} (Triple-A)")
    for lvl in ("AAA", "MLB"):
        if lvl in fin and fin[lvl].get("team"):
            out.append(f"{fin[lvl]['team']} ({LEVEL[lvl]})")
    return list(dict.fromkeys(out))


# ---- summary -------------------------------------------------------------------------------------
def _pick(pid: int, key: str, options: list[str]) -> str:
    """Deterministic phrasing variant, so pages don't all read alike and reruns don't change them."""
    return options[zlib.crc32(f"{pid}:{key}".encode()) % len(options)]


def summary(player: dict, season: int) -> str:
    pid, name, fin = player["id"], player["name"], final(player)
    s: list[str] = []
    h, a, m = fin.get("hitter"), fin.get("AAA"), fin.get("MLB")
    ha = player.get("hitter_aaa") or {}
    # 1. What he did and where (always). The name leads the first sentence; later ones say "he".
    def who(cap: bool) -> str:
        return name if not s else ("He" if cap else "he")

    if h and h["pa"] >= MIN_PA:
        pos = f", mostly at {ha['position']}" if ha.get("position") else ""
        s.append(_pick(pid, "base", [f"{who(True)} had {h['pa']} plate appearances for {h['team']} in Triple-A in {season}{pos}.",
                                     f"In {season}, {who(False)} batted {h['pa']} times for {h['team']} in Triple-A{pos}."]))
    for lvl, r in (("AAA", a), ("MLB", m)):
        if not r or (lvl == "AAA" and (r.get("bf") or 0) < MIN_BF) or (lvl == "MLB" and r["pitches"] < MIN_MLB_PITCHES):
            continue
        role = ROLE.get(r.get("role") or "", "")
        role = f", {role}" if role else ""
        if lvl == "AAA":
            s.append(_pick(pid, "base" + lvl, [
                f"{who(True)} faced {r['bf']} batters in {r['games']} Triple-A games for {r['team']} in {season}{role}.",
                f"In {season}, {who(False)} pitched in {r['games']} games for {r['team']} in Triple-A{role}, facing {r['bf']} batters."]))
        else:
            s.append(_pick(pid, "base" + lvl, [
                f"{who(True)} threw {intc(r['pitches'])} MLB pitches for {r['team']} in {season}{role}.",
                f"In the majors, {who(False)} threw {intc(r['pitches'])} pitches for {r['team']} in {season}{role}."]))
    if not s:
        lv = " and ".join(LEVEL[k] for k in ("AAA", "MLB") if k in fin) or "Triple-A or MLB"
        s.append(f"{name} has a small {season} sample in {lv}: below the {MIN_PA} plate appearances, {MIN_BF} batters faced, "
                 f"or {MIN_MLB_PITCHES} MLB pitches this site uses for a full profile.")
        return " ".join(s)

    # Candidate facts as (score, kind, render(variant)); the two highest-scoring are used.
    notes: list[tuple[float, str, object]] = []
    # 2. Best pitch by Stuff+ (MLB first, where the grade is most meaningful).
    for lvl in ("MLB", "AAA"):
        r = fin.get(lvl)
        if not r:
            continue
        best = max((p for p in r["arsenal"] if (p.get("pitches") or 0) >= BEST_PITCH_N and (p.get("stuff_plus") or 0) >= BEST_PITCH_MIN),
                   key=lambda p: (p["stuff_plus"], p["pt"]), default=None)
        if best:
            nm, g, lv = PITCH.get(best["pt"], best["pt"]), num(best["stuff_plus"], 0), LEVEL[lvl]
            txt = _pick(pid, "best", [f"His {nm} grades {g} Stuff+ in {lv} (100 is average).",
                                      f"The standout pitch is his {nm}: {g} Stuff+ in {lv}, where 100 is average.",
                                      f"By Stuff+, his best pitch is the {nm} at {g} in {lv} (100 is average)."])
            notes.append((best["stuff_plus"] - 100 + 0.5, "pitch", lambda k, t=txt: t))
            break
    # 3. Triple-A percentiles at the extremes (qualified players only; percentiles are oriented higher = better).
    for key, metrics, pool in (("hitter", HIT_METRICS, "hitters"), ("AAA", PIT_METRICS, "pitchers")):
        r = fin.get(key)
        if not r or not r.get("qualified"):
            continue
        for _label, f, fm, noun in metrics:
            p = r.get(f + "_pctl")
            if p is None or NOTABLE_LO < p < NOTABLE_HI:
                continue
            p, v = round(p), fm(r.get(f))
            rank = {0: f"the lowest percentile (0th) among qualified Triple-A {pool}",
                    100: f"the top percentile (100th) among qualified Triple-A {pool}"}.get(
                p, f"the {ordinal(p)} percentile of qualified Triple-A {pool}")
            notes.append((abs(p - 50) / 5, "pctl", lambda k, noun=noun, v=v, rank=rank: (
                f"His {noun} ({v}) ranks in {rank}." if k == 0 else f"His {noun}, {v}, also sits in {rank}.")))
    # 4. A change flag from the season's final 14 days.
    for fl in player.get("flags") or []:
        txt = "; ".join(x for x in (fl.get("flag_discipline"), fl.get("flag_contact")) if x)
        if txt:
            notes.append((1.0, "flag", lambda k, t=txt: f"Over the season's final 14 days: {t[0].lower() + t[1:]}."))
    notes.sort(key=lambda t: -t[0])
    k = 0
    for _, kind, render in notes[:2]:
        s.append(render(k if kind == "pctl" else 0))
        k += kind == "pctl"
    if k:
        s.append("Percentiles run so higher is better.")
    return " ".join(s)


# ---- static tables -------------------------------------------------------------------------------
def _e(x) -> str:
    return html.escape(str(x), quote=True)


def tiles(pairs) -> str:
    return '<div class="stat-row">' + "".join(
        f'<div class="stat"><div class="v">{_e(v)}</div><div class="k">{_e(k)}</div></div>' for k, v in pairs) + "</div>"


def table(head: list[str], rows: list[list], first_left: bool = True) -> str:
    th = "".join(f'<th{" class=l" if i == 0 and first_left else ""}>{_e(h)}</th>' for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td{" class=l" if i == 0 and first_left else ""}>{_e(c)}</td>' for i, c in enumerate(r)) + "</tr>"
                 for r in rows)
    return f'<div class="s-table-wrap"><table class="s-table static"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def pct_table(r: dict, metrics) -> str:
    rows = [[label, fm(r.get(f)), "—" if r.get(f + "_pctl") is None else str(round(r[f + "_pctl"]))] for label, f, fm, _ in metrics]
    return table(["", "Value", "Percentile"], rows)


def arsenal_table(r: dict) -> str:
    rows = [[PITCH.get(p["pt"], p["pt"]).capitalize() + f" ({p['pt']})", intc(p.get("pitches")), pct(p.get("usage_pct"), 0),
             num(p.get("velo")), num(p.get("ivb")), num(p.get("hb")), pct(p.get("whiff_pct"), 0),
             num(p.get("stuff_plus"), 0), num(p.get("loc_plus"), 0), num(p.get("pitching_plus"), 0)] for p in r["arsenal"]]
    return table(["Pitch", "Thrown", "Usage", "Velo", "IVB", "HB", "Whiff%", "Stuff+", "Location+", "Pitching+"], rows)


def static_body(player: dict, through: dict[str, str]) -> str:
    """Server-rendered sections, the same headings the interactive page draws over them."""
    fin, out = final(player), []
    h = fin.get("hitter")
    if h and h["pa"] >= MIN_PA:
        out += ["<h2>Hitting · Triple-A</h2>",
                tiles([("PA", h["pa"]), ("xwOBA", woba(h.get("xwoba"))), ("Chase", pct(h.get("chase_pct"))),
                       ("Hard-hit", pct(h.get("hard_hit_pct"))), ("EV90", num(h.get("ev90")) + " mph")]),
                pct_table(h, HIT_METRICS),
                f'<p class="note">Season to date through {_e(through["AAA"])}. Percentiles vs. qualified Triple-A hitters; '
                "higher is always better (chase, whiff, and strikeout rate are inverted).</p>"]
    for lvl in ("AAA", "MLB"):
        r = fin.get(lvl)
        if not r or (lvl == "AAA" and (r.get("bf") or 0) < MIN_BF):
            continue
        out.append(f"<h2>Pitching · {LEVEL[lvl]}</h2>")
        if r.get("role"):
            out.append(f'<p class="role-tag">{_e(r["role"])} · {r.get("games") or 0} G, {r.get("starts") or 0} GS</p>')
        if lvl == "AAA":
            out += [tiles([("BF", r.get("bf")), ("K%", pct(r.get("k_pct"))), ("BB%", pct(r.get("bb_pct"))),
                           ("Stuff+", num(r.get("stuff_plus"), 0)), ("Location+", num(r.get("loc_plus"), 0)),
                           ("FB velo", num(r.get("fb_velo")) + " mph")]),
                    pct_table(r, PIT_METRICS)]
        else:
            out.append(tiles([("Pitches", intc(r.get("pitches"))), ("Stuff+", num(r.get("stuff_plus"), 0)),
                              ("Location+", num(r.get("loc_plus"), 0)), ("Pitching+", num(r.get("pitching_plus"), 0)),
                              ("RV+ (results)", num(r.get("rv_plus"), 0)), ("K%", pct(r.get("k_pct"))), ("BB%", pct(r.get("bb_pct")))]))
        out += ["<h3>Arsenal</h3>", arsenal_table(r),
                f'<p class="note">Season to date through {_e(through[lvl])}. Movement in inches. Stuff+, Location+, Pitching+: '
                "100 is an average qualified MLB pitcher (or pitch, within its group).</p>"]
    flags = ["; ".join(x for x in (f.get("flag_discipline"), f.get("flag_contact")) if x) for f in player.get("flags") or []]
    flags = [f for f in flags if f]
    if flags:
        out.append('<div class="flag-list"><strong>Final 14 days</strong><ul>' + "".join(f"<li>{_e(f)}</li>" for f in flags) + "</ul></div>")
    return "\n".join(out)
