"""Opponent adjustment: ridge regression of play-level EPA on offense, defense, and home field.

    epa_play = mean + off[offense] + def[defense] + home * is_home + noise

fit with an L2 penalty `lam` on every coefficient (the mean is not penalized). The coefficients
estimate each team's strength, shrunk toward average in proportion to how little data there is
(`lam` is in units of plays: a coefficient is shrunk as if the team had `lam` extra plays at the
league average).

What the site shows as "opponent-adjusted" is a schedule correction on the team's own numbers:

    off_adj = off_raw - average over the team's offensive plays of (def[opponent] + home * is_home)
    def_adj = def_raw - average over plays against the team of (off[opponent] + home * is_home)

so a team keeps its own (unshrunk) production and only the strength of the opponents it faced is
removed. The shrunk coefficients are used for that opponent strength, where shrinkage is what you
want, and are also returned as off_ridge/def_ridge. `lam` is chosen by out-of-sample prediction of
the next week's game margins from off_adj/def_adj (nfl/team_efficiency/adjusted.py).

Plays are first collapsed to (offense, defense, home) cells with a count and a sum. Weighted
least squares on the cells gives exactly the play-level ridge solution, and it is fast enough to
refit for every season x week x variant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ridge_ratings(plays: pd.DataFrame, lam: float, teams: list[str] | None = None,
                  off="offense", deff="defense", home="is_home", y="epa") -> pd.DataFrame:
    """Adjusted offense/defense per team from play rows (columns offense, defense, is_home, epa).

    Returns one row per team: team, off_plays, def_plays, off_raw, def_raw, off_adj, def_adj
    (schedule-corrected), off_ridge, def_ridge (shrunk strength estimates), plus the fitted mean and
    home effect as attributes `mean` and `home` on the frame.
    """
    if lam <= 0:
        raise ValueError("lam must be positive: offense and defense dummies are collinear without it")
    teams = sorted(set(plays[off]) | set(plays[deff])) if teams is None else list(teams)
    idx = {t: i for i, t in enumerate(teams)}
    k = len(teams)

    cells = (plays.assign(_h=plays[home].astype(float))
             .groupby([off, deff, "_h"], sort=True)[y].agg(["size", "sum"]).reset_index())
    n = cells["size"].to_numpy(float)
    s = cells["sum"].to_numpy(float)
    mean = s.sum() / n.sum()

    # Design per cell: off one-hot, def one-hot, home. Target = cell mean minus league mean, weight n.
    X = np.zeros((len(cells), 2 * k + 1))
    oi = cells[off].map(idx).to_numpy()
    di = cells[deff].map(idx).to_numpy()
    X[np.arange(len(cells)), oi] = 1.0
    X[np.arange(len(cells)), k + di] = 1.0
    X[:, 2 * k] = cells["_h"].to_numpy()
    r = s / n - mean
    XtW = X.T * n
    beta = np.linalg.solve(XtW @ X + lam * np.eye(2 * k + 1), XtW @ r)

    g_off = plays.groupby(off)[y].agg(["size", "mean"])
    g_def = plays.groupby(deff)[y].agg(["size", "mean"])
    out = pd.DataFrame({"team": teams})
    out["off_plays"] = out["team"].map(g_off["size"]).fillna(0).astype(int)
    out["def_plays"] = out["team"].map(g_def["size"]).fillna(0).astype(int)
    out["off_raw"] = out["team"].map(g_off["mean"])
    out["def_raw"] = out["team"].map(g_def["mean"])
    b_off, b_def, b_home = beta[:k], beta[k:2 * k], beta[2 * k]
    out["off_ridge"] = mean + b_off
    out["def_ridge"] = mean + b_def
    # Opponent effect each team faced, weighted by plays: on offense, the defenses (and home field);
    # on defense, the offenses (and home field for those offenses).
    h = cells["_h"].to_numpy()
    opp_vs_off = np.bincount(oi, weights=n * (b_def[di] + b_home * h), minlength=k) / np.maximum(np.bincount(oi, n, k), 1)
    opp_vs_def = np.bincount(di, weights=n * (b_off[oi] + b_home * h), minlength=k) / np.maximum(np.bincount(di, n, k), 1)
    out["off_adj"] = out["off_raw"] - opp_vs_off
    out["def_adj"] = out["def_raw"] - opp_vs_def
    out.attrs["mean"] = float(mean)
    out.attrs["home"] = float(beta[2 * k])
    return out
