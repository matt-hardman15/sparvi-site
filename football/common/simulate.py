"""A season with known team strengths and an unbalanced schedule, for testing the opponent adjustment.

The schedule correction can only be checked against truth where truth is known. `season()` draws team offense and
defense strengths, builds a schedule in which teams in the same division play each other twice and most other pairs
not at all (so who you played matters), and draws play EPA as

    epa = mean + off[offense] + def[defense] + home * is_home + noise.

`recovery()` then asks how much of each team's true schedule effect (the average true strength of the opponents it
faced, plus home field) the adjustment's correction accounts for:

    recovered = 1 - sum((estimated - true)^2) / sum(true^2)      over teams, offense and defense, and seasons

1 = the correction removes the schedule exactly; 0 = no better than not adjusting; < 0 = worse than not adjusting.
Defaults are NFL-sized (2023-2025: play EPA sd 1.16; true team sd 0.075 on offense, 0.045 on defense; about 60
plays a side per game), so the number says what the published adjustment can and cannot do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from football.common.adjust import ridge_ratings


def schedule(rng: np.random.Generator, n_teams: int = 32, division: int = 4, games: int = 17) -> list[tuple[int, int, int]]:
    """(week, home, away) rows: each team plays its division rivals home and away, then others at random, one game
    a week. Division games are spread through the season; a team can sit out a week if no partner is free."""
    teams = np.arange(n_teams)
    div = teams // division
    pairs = [(a, b) for a in teams for b in teams if a != b and div[a] == div[b]]       # (home, away), both directions
    rng.shuffle(pairs)
    played = {t: 0 for t in teams}
    out: list[tuple[int, int, int]] = []
    week = 0
    while min(played.values()) < games and week < games + 4:
        week += 1
        free = set(teams[[played[t] < games for t in teams]])
        this_week = []
        for hm, aw in list(pairs):                       # division games first, about a third of the weeks
            if hm in free and aw in free and rng.random() < 0.45:
                this_week.append((hm, aw)); pairs.remove((hm, aw)); free -= {hm, aw}
        rest = list(free)
        rng.shuffle(rest)
        while len(rest) >= 2:
            a = rest.pop()
            j = next((i for i, b in enumerate(rest) if div[a] != div[b]), None)
            if j is None:
                continue
            this_week.append((a, rest.pop(j)))
        for hm, aw in this_week:
            out.append((week, int(hm), int(aw)))
            played[hm] += 1; played[aw] += 1
    return out


def season(rng: np.random.Generator, n_teams: int = 32, sd_off: float = 0.075, sd_def: float = 0.045,
           sd_play: float = 1.16, plays_per_side: int = 60, home: float = 0.01, mean: float = 0.0,
           games: int = 17, division: int = 4, strength_by_division: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(plays, truth). plays: week, game_id, offense, defense, is_home, epa. truth: team, off, def (true strengths:
    a better defense has a lower `def`). `strength_by_division` (0-1) is the share of strength variance shared by a
    division, which makes schedules more unequal (a strong division's teams all face strong opponents)."""
    names = [f"T{i:02d}" for i in range(n_teams)]
    d_eff = rng.normal(size=(2, n_teams // division)).repeat(division, axis=1)
    own = rng.normal(size=(2, n_teams))
    z = np.sqrt(strength_by_division) * d_eff + np.sqrt(1 - strength_by_division) * own
    off, deff = sd_off * z[0], sd_def * z[1]
    rows = []
    for gid, (week, hm, aw) in enumerate(schedule(rng, n_teams, division, games)):
        for o, d, h in ((hm, aw, 1), (aw, hm, 0)):
            n = int(rng.poisson(plays_per_side))
            rows.append(pd.DataFrame({"week": week, "game_id": gid, "offense": names[o], "defense": names[d],
                                      "is_home": h, "epa": mean + off[o] + deff[d] + home * h + rng.normal(0, sd_play, n)}))
    truth = pd.DataFrame({"team": names, "off": off, "def": deff})
    return pd.concat(rows, ignore_index=True), truth.assign(home=home)


def true_schedule(plays: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Each team's true schedule effect on its offense and defense EPA/play, centered on the league (the part of its
    raw number that is its opponents and home field, not itself)."""
    t = truth.set_index("team")
    home = float(truth["home"].iloc[0])
    p = plays.assign(vs_off=plays["defense"].map(t["def"]) + home * plays["is_home"],
                     vs_def=plays["offense"].map(t["off"]) + home * plays["is_home"])
    return pd.DataFrame({"off_sched": p.groupby("offense")["vs_off"].mean(),
                         "def_sched": p.groupby("defense")["vs_def"].mean()}).rename_axis("team").reset_index()


def recovery(lam: float, lam_schedule=None, seasons: int = 40, seed: int = 0, through_week: int | None = None,
             **kw) -> dict:
    """Share of the true schedule effect the correction recovers (see module docstring), with the pieces to read it:
    the true effect's sd, the estimate's sd, and the slope of estimate on truth (1 = full size, 0 = none)."""
    rng = np.random.default_rng(seed)
    est, tru, err_raw, err_adj = [], [], [], []
    for _ in range(seasons):
        plays, truth = season(rng, **kw)
        if through_week:
            plays = plays[plays["week"] <= through_week]
        r = ridge_ratings(plays, lam, lam_schedule=lam_schedule).set_index("team")
        ts = true_schedule(plays, truth).set_index("team").reindex(r.index)
        t = truth.set_index("team").reindex(r.index)
        for side, true_strength in (("off", t["off"]), ("def", t["def"])):
            e, s = r[f"{side}_sched"] - r[f"{side}_sched"].mean(), ts[f"{side}_sched"] - ts[f"{side}_sched"].mean()
            est.append(e.to_numpy()); tru.append(s.to_numpy())
            # Error against the team's true strength, centered: raw vs. adjusted (the noise in both is the same plays).
            target = true_strength - true_strength.mean()
            err_raw.append((r[f"{side}_raw"] - r[f"{side}_raw"].mean() - target).to_numpy())
            err_adj.append((r[f"{side}_adj"] - r[f"{side}_adj"].mean() - target).to_numpy())
    est, tru, err_raw, err_adj = (np.concatenate(x) for x in (est, tru, err_raw, err_adj))
    return {
        "recovered": float(1 - ((est - tru) ** 2).sum() / (tru ** 2).sum()),
        "slope": float((est * tru).sum() / (tru ** 2).sum()),
        "corr": float(np.corrcoef(est, tru)[0, 1]),
        "sd_true": float(tru.std()), "sd_estimated": float(est.std()),
        "rmse_raw": float(np.sqrt((err_raw ** 2).mean())), "rmse_adjusted": float(np.sqrt((err_adj ** 2).mean())),
        "seasons": seasons,
    }
