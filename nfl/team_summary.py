"""Summary and static tables for the NFL team pages (PLAN-seo S2 standard, BACKLOG SEO-2b).

Everything is built by rule from site/data/nfl, for the view a team page opens on: the current
season through its last week, opponent-adjusted, garbage time removed, all plays. The interactive page
(site/nfl/teams/_team.qmd) draws over the static block. tests/test_football.py checks every number in a
summary appears in that page's tables.
"""
from __future__ import annotations

import html
import sys
import zlib

from sparvi.config import ROOT

sys.path.insert(0, str(ROOT / "site" / "build"))
from summarize import ordinal, table, tiles  # noqa: E402  (shared with the player pages)

GT = "no_gt"


def signed(v, d=3) -> str:
    return "—" if v is None else ("+" if v > 0 else "") + f"{v:.{d}f}"


def pct(v) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def _pick(key: str, options: list[str]) -> str:
    return options[zlib.crc32(key.encode()) % len(options)]


class League:
    """One season's end-of-data snapshot for all teams: values and ranks (1 = best)."""

    def __init__(self, week_file: dict, game_file: dict, teams: list[dict]):
        self.week = week_file["weeks"][-1]
        self.k = len(week_file["weeks"]) - 1
        self.series = {t["id"]: t["series"][GT] for t in week_file["teams"]}
        self.games = game_file["teams"]
        self.teams = {t["id"]: t for t in teams}

    def val(self, team: str, field: str):
        s = self.series.get(team, {}).get(field)
        return s[self.k] if s else None

    def net(self, team: str):
        o, d = self.val(team, "off_epa_adj"), self.val(team, "def_epa_adj")
        return None if o is None or d is None else o - d

    def rank(self, team: str, fn, lower_better: bool = False) -> tuple[int | None, int]:
        vals = {t: fn(t) for t in self.series}
        vals = {t: v for t, v in vals.items() if v is not None}
        v = vals.get(team)
        if v is None:
            return None, len(vals)
        return 1 + sum((x < v) if lower_better else (x > v) for x in vals.values()), len(vals)

    def record(self, team: str) -> str:
        g = [x for x in self.games.get(team, []) if x.get("res") and x["week"] <= self.week]
        w, losses, t = (sum(x["res"] == r for x in g) for r in "WLT")
        return f"{w}–{losses}" + (f"–{t}" if t else "")


def splits(L: League, team: str) -> list[list[str]]:
    """Offense/defense x all/pass/run: value and rank (defense is EPA allowed, lower is better)."""
    rows = []
    for side, lower in (("off", False), ("def", True)):
        for split, f in (("All plays", f"{side}_epa_adj"), ("Passes", f"{side}_pass_epa_adj"), ("Runs", f"{side}_run_epa_adj")):
            r, n = L.rank(team, lambda t, f=f: L.val(t, f), lower)
            rows.append([f"{'Offense' if side == 'off' else 'Defense (allowed)'} · {split.lower()}", signed(L.val(team, f)),
                         "—" if r is None else f"{ordinal(r)} of {n}"])
    return rows


def summary(L: League, team: str, season: int) -> str:
    T = L.teams[team]
    name = T["name"]
    o, d, net = L.val(team, "off_epa_adj"), L.val(team, "def_epa_adj"), L.net(team)
    if o is None or d is None:
        return f"The {name} have no {season} games in the data yet."
    ro, n = L.rank(team, lambda t: L.val(t, "off_epa_adj"))
    rd, _ = L.rank(team, lambda t: L.val(t, "def_epa_adj"), True)
    s = [_pick(f"{team}base", [f"Through week {L.week} of {season}, the {name} are {L.record(team)}.",
                               f"The {name} are {L.record(team)} through week {L.week} of {season}."]),
         f"Opponent-adjusted, the offense ranks {ordinal(ro)} of {n} in EPA per play ({signed(o)}) "
         f"and the defense {ordinal(rd)} ({signed(d)} allowed)."]
    # One more fact, by rule: the division's best or worst net EPA, else a pass/run split in the top or bottom 3.
    div = [t for t, x in L.teams.items() if x["division"] == T["division"] and L.net(t) is not None]
    nets = sorted(div, key=lambda t: -L.net(t))
    if len(nets) > 1 and team in (nets[0], nets[-1]):
        best = team == nets[0]
        s.append(f"The team's net EPA per play ({signed(net)}) is the {'best' if best else 'worst'} in the {T['division']}.")
    else:
        for side, lower, label in (("off", False, "offense"), ("def", True, "defense")):
            hit = None
            for split in ("pass", "run"):
                f = f"{side}_{split}_epa_adj"
                r, n = L.rank(team, lambda t, f=f: L.val(t, f), lower)
                if r is not None and (r <= 3 or r > n - 3):
                    hit = (split, r, n, L.val(team, f))
                    break
            if hit:
                split, r, n, v = hit
                word = "passing" if split == "pass" else "running"
                s.append(f"The {word} {label} ranks {ordinal(r)} of {n} ({signed(v)}{' allowed' if side == 'def' else ''}).")
                break
    return " ".join(s)


def static_block(L: League, team: str, season: int, team_href) -> str:
    """Tiles, splits, and the game log as plain HTML: what a crawler or a reader without JavaScript sees."""
    net = L.net(team)
    rn, n = L.rank(team, L.net)
    ro, _ = L.rank(team, lambda t: L.val(t, "off_epa_adj"))
    rd, _ = L.rank(team, lambda t: L.val(t, "def_epa_adj"), True)
    ordn = lambda r: "—" if r is None else ordinal(r)  # noqa: E731
    out = [f"<h3>{season} through week {L.week}</h3>",
           tiles([("Record", L.record(team)), (f"Offense EPA/play · {ordn(ro)} of {n}", signed(L.val(team, "off_epa_adj"))),
                  (f"Defense EPA/play · {ordn(rd)} of {n} (allowed)", signed(L.val(team, "def_epa_adj"))),
                  (f"Net EPA/play · {ordn(rn)} of {n}", signed(net))]),
           table(["Opponent-adjusted EPA/play", "Value", "Rank"], splits(L, team))]
    rows = []
    for g in sorted(L.games.get(team, []), key=lambda g: g["week"]):
        opp = L.teams.get(g["opp"])
        who = f'{"vs" if g.get("home") else "@"} {g["opp"]}'
        rows.append((g["week"], who, opp, f'{g["res"]} {g["pf"]}–{g["pa"]}' if g.get("res") else "—",
                     signed(g.get("off_ng_epa")), signed(g.get("def_ng_epa")), pct(g.get("off_ng_sr"))))
    head = "".join(f"<th{' class=l' if i in (1, 2) else ''}>{html.escape(h)}</th>" for i, h in
                   enumerate(["Wk", "Opponent", "Result", "Off EPA/play", "Def EPA/play", "Off success"]))
    body = "".join(
        f"<tr><td>{w}</td><td class=l>{html.escape(who.split(' ')[0])} "
        + (f'<a href="{team_href(opp)}">{html.escape(opp["id"])}</a>' if opp else html.escape(who.split(" ")[1]))
        + f"</td><td class=l>{html.escape(res)}</td><td>{o}</td><td>{d}</td><td>{sr}</td></tr>"
        for w, who, opp, res, o, d, sr in rows)
    out += ["<h3>Results</h3>",
            f'<div class="s-table-wrap"><table class="s-table static"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>',
            '<p class="note">Single games are raw (not opponent-adjusted), garbage time removed.</p>']
    return "\n".join(out)
