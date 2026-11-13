"""NFL team abbreviations: one code per franchise across every season.

nflverse play-by-play already uses the current codes (LA, LAC, LV) for every season; schedules and
some other tables keep the historical ones. scripts/pull_nfl.py maps everything through ALIASES so
every raw file shares one set of 32 codes.
"""
NFL_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}


def nfl_team(code: str) -> str:
    return NFL_ALIASES.get(code, code)
