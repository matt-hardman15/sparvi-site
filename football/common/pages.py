"""Shared pieces of the NFL and CFB team pages at the S2 standard (PLAN-seo; BACKLOG SEO-2b).

Both sports' week files (`team_week_<season>.json`) and game files (`team_games_<season>.json`) share field
names, so ranking, the offense/defense split table, and the game log are built here; each sport's
team_summary.py adds its own tiles and summary sentences. Formats match Sparvi.fmt in sparvi.js, so the
static numbers equal what the interactive page shows on load.
"""
from __future__ import annotations

import html
import sys
import zlib
from collections.abc import Callable

from sparvi.config import ROOT

sys.path.insert(0, str(ROOT / "site" / "build"))
from summarize import ordinal, table, tiles  # noqa: E402,F401  (shared with the player pages; re-exported)

GT = "no_gt"   # the view the pages open on: garbage time removed


def signed(v, d=3) -> str:
    return "—" if v is None else ("+" if v > 0 else "") + f"{v:.{d}f}"


def pct(v) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def pick(key: str, options: list[str]) -> str:
    """Deterministic phrasing variant (crc32, not Python's salted hash), so reruns don't change pages."""
    return options[zlib.crc32(key.encode()) % len(options)]


def ordn(r: int | None) -> str:
    return "—" if r is None else ordinal(r)


class League:
    """One season's final snapshot for every team in the week file: values and ranks (1 = best)."""

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

    def rank(self, team: str, fn: Callable, lower_better: bool = False) -> tuple[int | None, int]:
        vals = {t: fn(t) for t in self.series}
        vals = {t: v for t, v in vals.items() if v is not None}
        v = vals.get(team)
        if v is None:
            return None, len(vals)
        return 1 + sum((x < v) if lower_better else (x > v) for x in vals.values()), len(vals)

    def field_rank(self, team: str, field: str, lower_better: bool = False) -> tuple[int | None, int]:
        return self.rank(team, lambda t: self.val(t, field), lower_better)

    def record(self, team: str) -> str:
        g = [x for x in self.games.get(team, []) if x.get("res") and x["week"] <= self.week]
        w, losses, t = (sum(x["res"] == r for x in g) for r in "WLT")
        return f"{w}–{losses}" + (f"–{t}" if t else "")


def splits(L: League, team: str) -> str:
    """Offense/defense x all/pass/run, opponent-adjusted EPA/play with ranks (defense is allowed: lower is better)."""
    rows = []
    for side, lower in (("off", False), ("def", True)):
        for split, f in (("all plays", f"{side}_epa_adj"), ("passes", f"{side}_pass_epa_adj"), ("runs", f"{side}_run_epa_adj")):
            r, n = L.field_rank(team, f, lower)
            rows.append([f"{'Offense' if side == 'off' else 'Defense (allowed)'} · {split}", signed(L.val(team, f)),
                         "—" if r is None else f"{ordinal(r)} of {n}"])
    return table(["Opponent-adjusted EPA/play", "Value", "Rank"], rows)


def split_fact(L: League, team: str) -> str | None:
    """A pass or run split in the top or bottom 3, offense first, as a sentence; None if there isn't one."""
    for side, lower, label in (("off", False, "offense"), ("def", True, "defense")):
        for split, word in (("pass", "passing"), ("run", "running")):
            f = f"{side}_{split}_epa_adj"
            r, n = L.field_rank(team, f, lower)
            if r is not None and (r <= 3 or r > n - 3):
                return f"The {word} {label} ranks {ordinal(r)} of {n} ({signed(L.val(team, f))}{' allowed' if side == 'def' else ''})."
    return None


def game_log(L: League, team: str, href: Callable[[dict], str], label: Callable[[dict], str]) -> str:
    """Every game this season; opponents with a page are linked. Single games are raw, garbage time removed."""
    body = []
    for g in sorted(L.games.get(team, []), key=lambda g: g["week"]):
        opp = L.teams.get(g["opp"])
        who = f'<a href="{href(opp)}">{html.escape(label(opp))}</a>' if opp else html.escape(g["opp"])
        res = f'{g["res"]} {g["pf"]}–{g["pa"]}' if g.get("res") else "—"
        body.append(f'<tr><td>{g["week"]}</td><td class=l>{"vs" if g.get("home") else "@"} {who}</td>'
                    f"<td class=l>{html.escape(res)}</td><td>{signed(g.get('off_ng_epa'))}</td>"
                    f"<td>{signed(g.get('def_ng_epa'))}</td><td>{pct(g.get('off_ng_sr'))}</td></tr>")
    head = "".join(f"<th{' class=l' if i in (1, 2) else ''}>{h}</th>" for i, h in
                   enumerate(["Wk", "Opponent", "Result", "Off EPA/play", "Def EPA/play", "Off success"]))
    return ("<h3>Results</h3>\n"
            f'<div class="s-table-wrap"><table class="s-table static"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>\n'
            '<p class="note">Single games are raw (not opponent-adjusted), garbage time removed.</p>')
