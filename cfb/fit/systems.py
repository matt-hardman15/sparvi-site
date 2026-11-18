"""Empirical team systems (SYSTEM-FIT F1): each team-season's offense as a style vector, summarized two ways:

  dimensions   principal components of the standardized style vector, fit on every team-season. The fit model uses
               these: styles turned out continuous rather than discrete (F1 report: split-half agreement of hard systems
               0.38-0.63), while the first dimensions are stable (D1 pass-heaviness, D2 spread vs. TE/RB passing game)
               and persist year to year under the same head coach far more than after a change.
  systems      a Gaussian mixture with LABEL_K components, as readable labels only (the triple option separates at
               every K). Style means *how* an offense plays (play-calling, pace, aggression, how the ball is
spread across positions), not *how well*; efficiency stays out.

STYLE (every season from 2014) is what the clustering uses, so systems are comparable across seasons. STYLE_TEXT (from
the richer 2025+ play text) describes current teams but isn't clustered on. Add a style feature: add a function to
STYLE; it's standardized and clustered with the rest.

Output of fit(): the mixture (number of systems chosen by BIC among K_GRID), each team-season's probabilities over
systems, and per-system feature means (z-scores) for naming them afterwards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from cfb.profile import engine as G
from football.common.filters import cfb_garbage_time

K_GRID = range(3, 11)
LABEL_K = 4            # labels: the most systems whose even/odd-season refits agree at ARI >= 0.55 (F1 report)
N_DIMS = 5
SEED = 11
RUN_PASS = {"Rush", "Rushing Touchdown", "Pass Reception", "Pass Incompletion", "Passing Touchdown", "Sack", "Interception",
            "Pass Interception Return", "Pass Interception", "Interception Return Touchdown"}
PASS = RUN_PASS - {"Rush", "Rushing Touchdown"}


def _offense_plays(plays: pd.DataFrame) -> pd.DataFrame:
    p = plays[(plays["seasonType"] == "regular")].copy()
    margin = (p["offenseScore"] - p["defenseScore"]).abs()
    p = p[~cfb_garbage_time(p["period"].to_numpy(), margin.to_numpy())]
    p["close"] = margin.loc[p.index] <= 8
    p["is_pass"] = p["playType"].isin(PASS)
    p["is_rp"] = p["playType"].isin(RUN_PASS)
    return p


def style(plays: pd.DataFrame, ev: pd.DataFrame, pos: pd.Series, min_plays: int = 300) -> pd.DataFrame:
    """One row per offense (team) for one season: the STYLE features."""
    p = _offense_plays(plays)
    rp = p[p["is_rp"]]
    g = rp.groupby("offense")
    out = pd.DataFrame({"plays": g.size()})
    out["pass_rate"] = g["is_pass"].mean()
    early = rp[rp["down"] <= 2]
    out["early_pass_rate"] = early.groupby("offense")["is_pass"].mean()
    neutral = early[early["close"] & early["yardsToGoal"].between(20, 80)]
    out["neutral_pass_rate"] = neutral.groupby("offense")["is_pass"].mean()
    short3 = rp[(rp["down"] == 3) & (rp["distance"] <= 3)]
    out["short_yardage_pass_rate"] = short3.groupby("offense")["is_pass"].mean()
    out["plays_per_game"] = g.size() / g["gameId"].nunique()
    fourth = p[(p["down"] == 4) & (p["distance"] <= 5) & (p["yardsToGoal"] <= 60)]
    fourth = fourth[fourth["is_rp"] | fourth["playType"].str.contains("Punt|Field Goal", regex=True, na=False)]
    out["fourth_down_go_rate"] = fourth.groupby("offense")["is_rp"].mean()
    # How the ball is spread: target shares by position, concentration of targets and carries, QB runs.
    grp = ev["player_id"].map(pos).map(G.POSITION_GROUP)
    tg = ev[ev["opp"] == "target"].assign(g=grp)
    share = tg.groupby(["offense", "g"]).size().unstack(fill_value=0)
    share = share.div(share.sum(axis=1), axis=0)
    for k in ("WR", "TE", "RB"):
        out[f"{k.lower()}_target_share"] = share.get(k, pd.Series(dtype=float))
    hhi = lambda s: float(((s / s.sum()) ** 2).sum())  # noqa: E731
    out["target_concentration"] = tg.groupby("offense")["player_id"].agg(lambda s: hhi(s.value_counts()))
    ca = ev[ev["opp"] == "carry"].assign(g=grp)
    out["carry_concentration"] = ca[ca["g"] == "RB"].groupby("offense")["player_id"].agg(lambda s: hhi(s.value_counts()))
    out["qb_run_share"] = ca.groupby("offense")["g"].apply(lambda s: float((s == "QB").mean()))
    comp = ev[(ev["opp"] == "dropback") & (ev["complete"] == 1)]
    out["yards_per_completion"] = comp.groupby("offense")["yards"].mean()
    out = out[out["plays"] >= min_plays].drop(columns="plays")
    return out.dropna()


STYLE = ["pass_rate", "early_pass_rate", "neutral_pass_rate", "short_yardage_pass_rate", "plays_per_game", "fourth_down_go_rate",
         "wr_target_share", "te_target_share", "rb_target_share", "target_concentration", "carry_concentration", "qb_run_share",
         "yards_per_completion"]
STYLE_TEXT = ["shotgun", "no_huddle", "deep", "dir_middle"]


CLIP = 3.0     # z-scores capped: a few team-seasons have extreme values from thin or oddly coded data


def standardize(x: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    return ((x[STYLE] - ref[STYLE].mean()) / ref[STYLE].std()).clip(-CLIP, CLIP)


def fit(styles: pd.DataFrame, k: int | None = None) -> dict:
    """styles: team-seasons (index (season, team)) x STYLE. Chooses k by BIC unless given; returns the model."""
    Z = standardize(styles, styles).to_numpy()
    bic = {}
    models = {}
    for kk in ([k] if k else K_GRID):
        m = GaussianMixture(kk, covariance_type="diag", n_init=10, random_state=SEED).fit(Z)
        bic[kk], models[kk] = float(m.bic(Z)), m
    best = k or min(bic, key=bic.get)
    m = models[best]
    order = np.argsort(m.means_[:, STYLE.index("pass_rate")])       # systems numbered from most run-heavy to most pass-heavy
    probs = pd.DataFrame(m.predict_proba(Z)[:, order], index=styles.index, columns=[f"S{i + 1}" for i in range(best)])
    means = pd.DataFrame(m.means_[order], columns=STYLE, index=probs.columns)
    return {"k": best, "bic": bic, "model": m, "order": order, "probs": probs, "means": means, "ref": styles[STYLE]}


def predict(fitted: dict, styles: pd.DataFrame) -> pd.DataFrame:
    Z = standardize(styles, fitted["ref"]).to_numpy()
    p = fitted["model"].predict_proba(Z)[:, fitted["order"]]
    return pd.DataFrame(p, index=styles.index, columns=fitted["probs"].columns)


def dimensions(styles: pd.DataFrame, n: int = N_DIMS) -> dict:
    """Principal components of the standardized styles. Signs fixed so D1 rises with pass rate and D2 with WR target share."""
    from sklearn.decomposition import PCA
    Z = standardize(styles, styles)
    pca = PCA(n, random_state=0).fit(Z)
    comps = pca.components_.copy()
    for i, (feat, sign) in enumerate((("pass_rate", 1), ("wr_target_share", 1))):
        if i < n and np.sign(comps[i, STYLE.index(feat)]) != sign:
            comps[i] *= -1
    cols = [f"D{i + 1}" for i in range(n)]
    coords = pd.DataFrame((Z.to_numpy() - pca.mean_) @ comps.T, index=styles.index, columns=cols)
    return {"coords": coords, "loadings": pd.DataFrame(comps.T, index=STYLE, columns=cols),
            "explained": pca.explained_variance_ratio_.tolist(), "pca_mean": pca.mean_, "components": comps, "ref": styles[STYLE]}


def project(dims: dict, styles: pd.DataFrame) -> pd.DataFrame:
    Z = standardize(styles, dims["ref"]).to_numpy()
    return pd.DataFrame((Z - dims["pca_mean"]) @ dims["components"].T, index=styles.index, columns=dims["loadings"].columns)
