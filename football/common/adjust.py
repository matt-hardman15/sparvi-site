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
want, and are also returned as off_ridge/def_ridge.

The two uses want different penalties. `lam` is chosen by out-of-sample prediction of the next
week's game margins (nfl/team_efficiency/adjusted.py): a forecast, where a free slope on the rating
gap absorbs any uniform shrinkage, so that search barely constrains the size of the coefficients and
settles on a heavy penalty. Used for the schedule correction, those heavily shrunk strengths remove
almost none of the schedule (NFL, lambda 3,200: the largest correction in 2026 was 0.008 EPA/play).
`lam_schedule`, when given, is a second penalty used only for the opponent strengths the correction
subtracts; it is chosen by how well the coefficients predict a held-out game's EPA/play
(`cell_cv` here). off_ridge/def_ridge always come from `lam`. With `lam_schedule=None` (college
football today) one fit serves both, exactly as before.

Plays are first collapsed to (offense, defense, home) cells with a count and a sum. Weighted
least squares on the cells gives exactly the play-level ridge solution, and it is fast enough to
refit for every season x week x variant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _penalty(lam: float | tuple[float, float], k: int) -> np.ndarray:
    """Diagonal of the ridge penalty: one value for every coefficient, or (offense, defense) with the
    home coefficient taking the smaller of the two."""
    lo, ld = (lam, lam) if np.isscalar(lam) else lam
    if min(lo, ld) <= 0:
        raise ValueError("lam must be positive: offense and defense dummies are collinear without it")
    return np.r_[np.full(k, float(lo)), np.full(k, float(ld)), float(min(lo, ld))]


def _solve(oi: np.ndarray, di: np.ndarray, h: np.ndarray, n: np.ndarray, s: np.ndarray, k: int,
           lam: float | tuple[float, float]) -> tuple[float, np.ndarray]:
    """Weighted ridge on cells. Design per cell: off one-hot, def one-hot, home; target = cell mean
    minus league mean, weight n. Returns (mean, coefficients [off k, def k, home])."""
    mean = s.sum() / n.sum()
    m = len(n)
    X = np.zeros((m, 2 * k + 1))
    X[np.arange(m), oi] = 1.0
    X[np.arange(m), k + di] = 1.0
    X[:, 2 * k] = h
    XtW = X.T * n
    return float(mean), np.linalg.solve(XtW @ X + np.diag(_penalty(lam, k)), XtW @ (s / n - mean))


def cell_cv(plays: pd.DataFrame, lams: list, group: str = "game_id", forward_from: int | None = None,
            off="offense", deff="defense", home="is_home", y="epa") -> list[dict]:
    """Out-of-sample test of the strength coefficients themselves, on one season's plays: hold out each
    `group` (a game), fit on the rest, and predict the held-out game's EPA/play for each side as
    mean + off + def + home.

    By default the fit uses every other game (leave-one-game-out: the correction describes a season
    to date, so later games are fair evidence about an opponent). With `forward_from=w` it uses only
    games from earlier weeks, for games in week w+1 on (needs a `week` column): the stricter,
    forecast-style version of the same question. Returns, per penalty in `lams` (a value, or an
    (offense, defense) pair): sse, sse_mean (predicting the training mean), plays. Pool seasons with
    `pool_cv`.
    """
    teams = sorted(set(plays[off]) | set(plays[deff]))
    idx = {t: i for i, t in enumerate(teams)}
    k = len(teams)
    cols = [group, off, deff, "_h"] + (["week"] if forward_from is not None else [])
    cells = plays.assign(_h=plays[home].astype(float)).groupby(cols, sort=True)[y].agg(["size", "sum"]).reset_index()
    oi, di = cells[off].map(idx).to_numpy(), cells[deff].map(idx).to_numpy()
    h, n, s = cells["_h"].to_numpy(), cells["size"].to_numpy(float), cells["sum"].to_numpy(float)
    g = cells[group].to_numpy()
    wk = cells["week"].to_numpy() if forward_from is not None else None
    out = []
    for lam in lams:
        sse = sse0 = tot = 0.0
        for gi in pd.unique(g):
            te = g == gi
            if wk is None:
                tr = ~te
            else:
                w = wk[te][0]
                if w <= forward_from:
                    continue
                tr = wk < w
            mean, b = _solve(oi[tr], di[tr], h[tr], n[tr], s[tr], k, lam)
            obs = s[te] / n[te]
            e = obs - (mean + b[oi[te]] + b[k + di[te]] + b[2 * k] * h[te])
            sse += float((n[te] * e * e).sum())
            sse0 += float((n[te] * (obs - mean) ** 2).sum())
            tot += float(n[te].sum())
        out.append({"lambda": lam, "sse": sse, "sse_mean": sse0, "plays": tot})
    return out


def pool_cv(parts: list[list[dict]]) -> list[dict]:
    """Pool `cell_cv` results from several seasons: per penalty, mse per play and skill."""
    out = []
    for rows in zip(*parts, strict=True):
        sse, sse0, tot = (sum(r[key] for r in rows) for key in ("sse", "sse_mean", "plays"))
        out.append({"lambda": rows[0]["lambda"], "mse": sse / tot, "skill": 1 - sse / sse0, "plays": int(tot)})
    return out


def ridge_ratings(plays: pd.DataFrame, lam: float, teams: list[str] | None = None,
                  off="offense", deff="defense", home="is_home", y="epa",
                  lam_schedule: float | tuple[float, float] | None = None) -> pd.DataFrame:
    """Adjusted offense/defense per team from play rows (columns offense, defense, is_home, epa).

    Returns one row per team: team, off_plays, def_plays, off_raw, def_raw, off_adj, def_adj
    (schedule-corrected), off_ridge, def_ridge (shrunk strength estimates at `lam`), plus the fitted
    mean and home effect as attributes `mean` and `home` on the frame. `lam_schedule` (one value, or
    (offense, defense)) is the penalty for the opponent strengths off_adj/def_adj subtract; None = `lam`.
    The `home` attribute is the home effect of the fit the correction uses.
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
    oi = cells[off].map(idx).to_numpy()
    di = cells[deff].map(idx).to_numpy()
    h = cells["_h"].to_numpy()
    mean, beta = _solve(oi, di, h, n, s, k, lam)
    # The strengths the schedule correction subtracts: the same fit, or one with its own penalty.
    beta_s = beta if lam_schedule is None else _solve(oi, di, h, n, s, k, lam_schedule)[1]

    g_off = plays.groupby(off)[y].agg(["size", "mean"])
    g_def = plays.groupby(deff)[y].agg(["size", "mean"])
    out = pd.DataFrame({"team": teams})
    out["off_plays"] = out["team"].map(g_off["size"]).fillna(0).astype(int)
    out["def_plays"] = out["team"].map(g_def["size"]).fillna(0).astype(int)
    out["off_raw"] = out["team"].map(g_off["mean"])
    out["def_raw"] = out["team"].map(g_def["mean"])
    out["off_ridge"] = mean + beta[:k]
    out["def_ridge"] = mean + beta[k:2 * k]
    b_off, b_def, b_home = beta_s[:k], beta_s[k:2 * k], beta_s[2 * k]
    # Opponent effect each team faced, weighted by plays: on offense, the defenses (and home field);
    # on defense, the offenses (and home field for those offenses).
    opp_vs_off = np.bincount(oi, weights=n * (b_def[di] + b_home * h), minlength=k) / np.maximum(np.bincount(oi, n, k), 1)
    opp_vs_def = np.bincount(di, weights=n * (b_off[oi] + b_home * h), minlength=k) / np.maximum(np.bincount(di, n, k), 1)
    out["off_adj"] = out["off_raw"] - opp_vs_off
    out["def_adj"] = out["def_raw"] - opp_vs_def
    out.attrs["mean"] = float(mean)
    out["off_sched"] = opp_vs_off     # what was subtracted: > 0 = easier opponents than average
    out["def_sched"] = opp_vs_def
    out.attrs["home"] = float(b_home)
    return out
