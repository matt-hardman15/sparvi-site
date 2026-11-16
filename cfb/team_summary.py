"""Summary and static tables for the CFB team pages (PLAN-seo S2 standard, BACKLOG SEO-2b).

Built by rule from site/data/cfb for the view a team page opens on: the team's latest FBS season through its
last week, opponent-adjusted, garbage time removed, all plays. The interactive page
(site/cfb/teams/_team.qmd) draws over the static block. tests/test_team_pages.py checks every number in a
summary is in the page's tables. Ranking, splits, and the game log are shared with the NFL
(football/common/pages.py).
"""
from __future__ import annotations

from football.common.pages import League, game_log, ordinal, ordn, signed, split_fact, splits, tiles

__all__ = ["League", "power_str", "static_block", "summary"]


def power_str(x: float | None) -> str:
    """Power rating in points, with a true minus sign (as the page and preview images print it)."""
    return "—" if x is None else f"{x:+.1f}".replace("-", "−")


def power_n(L: League) -> int:
    return sum(L.val(t, "power") is not None for t in L.series)


def summary(L: League, t: dict, season: int, current_season: int, last_listed: int) -> tuple[str, dict]:
    """One paragraph from the team's latest season, plus the numbers the preview image uses."""
    tid = t["id"]
    record = L.record(tid)
    n = power_n(L)
    power, rank, ap = L.val(tid, "power"), L.val(tid, "rank"), L.val(tid, "ap")
    current = season == current_season
    when = (f"Through week {L.week} of {season}," if current
            else f"In {season}" + ("" if season == last_listed else ", its last FBS season,"))
    s = f"{when} {t['name']} {'is' if current else 'went'} {record}."
    if power is not None:
        s += f" {'Its' if current else 'Its final'} power rating, {power_str(power)} points against an average FBS team, " \
             f"{'ranks' if current else 'ranked'} {ordinal(rank)} of {n}"
        s += (f"; the AP poll {'has' if current else 'had'} it {ordinal(ap)}." if ap else
              f"; {'it is' if current else 'it was'} unranked in the AP poll.")
        # One more fact, by rule: best or worst power rating in its conference this season, else a top/bottom-3 split.
        conf = sorted((x for x, tt in L.teams.items() if tt["conf"] == t["conf"] and L.val(x, "power") is not None
                       and season in tt["seasons"]), key=lambda x: -L.val(x, "power"))
        if current and len(conf) > 1 and tid in (conf[0], conf[-1]):
            s += f" That is the {'highest' if tid == conf[0] else 'lowest'} power rating in the {t['conf']}."
        elif fact := split_fact(L, tid):
            s += " " + fact
    return s, {"record": record, "power": power, "rank": rank, "n": n, "week": L.week, "current": current}


def static_block(L: League, t: dict, season: int, current: bool, team_href) -> str:
    """Tiles, splits, and the game log as plain HTML: what a crawler or a reader without JavaScript sees."""
    tid = t["id"]
    ro, n = L.field_rank(tid, "off_epa_adj")
    rd, _ = L.field_rank(tid, "def_epa_adj", True)
    ap, cfp = L.val(tid, "ap"), L.val(tid, "cfp")
    return "\n".join([
        f"<h3>{season} {'through week ' + str(L.week) if current else 'final'}</h3>",
        tiles([("Record", L.record(tid)), (f"Power rating · {ordn(L.val(tid, 'rank'))} of {power_n(L)}", power_str(L.val(tid, "power"))),
               ("AP / CFP rank · latest polls", f"{ap or 'NR'} / {cfp or 'NR'}"),
               (f"Offense EPA/play · {ordn(ro)} of {n}", signed(L.val(tid, "off_epa_adj"))),
               (f"Defense EPA/play · {ordn(rd)} of {n} (allowed)", signed(L.val(tid, "def_epa_adj")))]),
        splits(L, tid),
        game_log(L, tid, team_href, lambda o: o["name"])])
