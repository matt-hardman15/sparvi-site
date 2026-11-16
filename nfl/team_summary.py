"""Summary and static tables for the NFL team pages (PLAN-seo S2 standard, BACKLOG SEO-2b).

Built by rule from site/data/nfl for the view a team page opens on: the current season through its last
week, opponent-adjusted, garbage time removed, all plays. The interactive page (site/nfl/teams/_team.qmd)
draws over the static block. tests/test_team_pages.py checks every number in a summary is in the page's
tables. Ranking, splits, and the game log are shared with CFB (football/common/pages.py).
"""
from __future__ import annotations

from football.common.pages import League, game_log, ordinal, ordn, pick, signed, split_fact, splits, tiles

__all__ = ["League", "ordinal", "static_block", "summary"]


def summary(L: League, team: str, season: int) -> str:
    T = L.teams[team]
    name = T["name"]
    o, d, net = L.val(team, "off_epa_adj"), L.val(team, "def_epa_adj"), L.net(team)
    if o is None or d is None:
        return f"The {name} have no {season} games in the data yet."
    ro, n = L.field_rank(team, "off_epa_adj")
    rd, _ = L.field_rank(team, "def_epa_adj", True)
    s = [pick(f"{team}base", [f"Through week {L.week} of {season}, the {name} are {L.record(team)}.",
                              f"The {name} are {L.record(team)} through week {L.week} of {season}."]),
         f"Opponent-adjusted, the offense ranks {ordinal(ro)} of {n} in EPA per play ({signed(o)}) "
         f"and the defense {ordinal(rd)} ({signed(d)} allowed)."]
    # One more fact, by rule: the division's best or worst net EPA, else a pass/run split in the top or bottom 3.
    nets = sorted((t for t, x in L.teams.items() if x["division"] == T["division"] and L.net(t) is not None),
                  key=lambda t: -L.net(t))
    if len(nets) > 1 and team in (nets[0], nets[-1]):
        s.append(f"The team's net EPA per play ({signed(net)}) is the {'best' if team == nets[0] else 'worst'} in the {T['division']}.")
    elif fact := split_fact(L, team):
        s.append(fact)
    return " ".join(s)


def static_block(L: League, team: str, season: int, team_href) -> str:
    """Tiles, splits, and the game log as plain HTML: what a crawler or a reader without JavaScript sees."""
    rn, n = L.rank(team, L.net)
    ro, _ = L.field_rank(team, "off_epa_adj")
    rd, _ = L.field_rank(team, "def_epa_adj", True)
    return "\n".join([
        f"<h3>{season} through week {L.week}</h3>",
        tiles([("Record", L.record(team)), (f"Offense EPA/play · {ordn(ro)} of {n}", signed(L.val(team, "off_epa_adj"))),
               (f"Defense EPA/play · {ordn(rd)} of {n} (allowed)", signed(L.val(team, "def_epa_adj"))),
               (f"Net EPA/play · {ordn(rn)} of {n}", signed(L.net(team)))]),
        splits(L, team),
        game_log(L, team, team_href, lambda t: t["id"])])
