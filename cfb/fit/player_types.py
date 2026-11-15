"""Empirical player types (SYSTEM-FIT F2): per position group, each player-season as a style vector (how he is used and
how he plays, plus size), summarized like team systems (systems.py): continuous dimensions for the fit model and a
mixture as readable labels. No type is defined by hand; names come afterwards from what distinguishes each type.

STYLE lists the profile features (cfb/profile/features.yml ids, shrunk values) per group; add one by adding its id.
A player-season qualifies with the group's minimum opportunities (MIN_N on the group's main feature).

Groups as used by the fit model (F2 report): QB and RB each on their own; receivers pooled (WR and TE together, with a
TE flag), because tight ends alone (about 900 qualified seasons) gave unstable dimensions. Discrete labels only where
they reproduce across even/odd-season refits: LABEL_K (None = continuous dimensions only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

STYLE = {
    "QB": ["qb_rush_share", "qb_rush_explosive", "qb_sack", "qb_comp", "qb_explosive", "qb_int"],
    "RB": ["rb_carry_share", "rb_target_share", "rb_run10", "rb_run20", "rb_stuff", "rb_goal_line", "rb_short"],
    "WR": ["rec_target_share", "rec_ypr", "rec_catch", "rec_explosive", "rec_redzone_share", "rec_third_share"],
    "TE": ["rec_target_share", "rec_ypr", "rec_catch", "rec_explosive", "rec_redzone_share", "rec_third_share"],
}
SIZE = ["height", "weight"]
MAIN = {"QB": ("qb_dropback_share", 100), "RB": ("rb_carry_share", 40), "WR": ("rec_target_share", 25), "TE": ("rec_target_share", 25)}
CLIP, SEED, N_DIMS = 3.0, 11, 4
FIT_GROUPS = {"QB": ["QB"], "RB": ["RB"], "REC": ["WR", "TE"]}
LABEL_K = {"QB": 2, "RB": 4, "REC": None}
K_GRID = range(2, 9)


def matrix(profile: pd.DataFrame, rosters: dict[int, pd.DataFrame], group: str) -> pd.DataFrame:
    """Player-seasons (index season, player_id) x STYLE[group] + SIZE, qualified players only."""
    p = profile[profile["group"] == group]
    feat, need = MAIN[group]
    ok = p[(p["feature"] == feat) & (p["n"] >= need)][["season", "player_id"]]
    wide = p.pivot_table(index=["season", "player_id"], columns="feature", values="shrunk")
    wide = wide.reindex(columns=STYLE[group]).loc[wide.index.isin(pd.MultiIndex.from_frame(ok))]
    size = pd.concat([r.assign(player_id=r["id"].astype(str), season=s)[["season", "player_id", *SIZE]].drop_duplicates(["season", "player_id"])
                      for s, r in rosters.items()]).set_index(["season", "player_id"])
    out = wide.join(size, how="left")
    out["height"] = out["height"].where(out["height"].between(62, 84))        # inches; outside: a data error
    out["weight"] = out["weight"].where(out["weight"].between(140, 400))      # pounds
    for c in [*STYLE[group], *SIZE]:
        out[c] = out[c].fillna(out[c].median())
    return out


def standardize(x: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    sd = ref.std().replace(0, 1.0)          # a constant feature standardizes to zero rather than dividing by zero
    return ((x - ref.mean()) / sd).clip(-CLIP, CLIP)


def fit_labels(m: pd.DataFrame, k: int) -> dict:
    Z = standardize(m, m).to_numpy()
    g = GaussianMixture(k, covariance_type="diag", n_init=10, random_state=SEED).fit(Z)
    order = np.argsort(g.means_[:, 0])
    probs = pd.DataFrame(g.predict_proba(Z)[:, order], index=m.index, columns=[f"T{i + 1}" for i in range(k)])
    return {"k": k, "model": g, "order": order, "probs": probs, "means": pd.DataFrame(g.means_[order], columns=m.columns, index=probs.columns),
            "ref": m, "bic": float(g.bic(Z))}


def predict_labels(fitted: dict, m: pd.DataFrame) -> pd.DataFrame:
    p = fitted["model"].predict_proba(standardize(m, fitted["ref"]).to_numpy())[:, fitted["order"]]
    return pd.DataFrame(p, index=m.index, columns=fitted["probs"].columns)


def dimensions(m: pd.DataFrame, n: int = N_DIMS) -> dict:
    """Principal components; sign fixed so each dimension's largest loading is positive."""
    Z = standardize(m, m)
    pca = PCA(n, random_state=0).fit(Z)
    comps = pca.components_ * np.sign(pca.components_[np.arange(n), np.abs(pca.components_).argmax(1)])[:, None]
    cols = [f"D{i + 1}" for i in range(n)]
    coords = pd.DataFrame((Z.to_numpy() - pca.mean_) @ comps.T, index=m.index, columns=cols)
    return {"coords": coords, "loadings": pd.DataFrame(comps.T, index=m.columns, columns=cols),
            "explained": pca.explained_variance_ratio_.tolist(), "components": comps, "pca_mean": pca.mean_, "ref": m}


def project(dims: dict, m: pd.DataFrame) -> pd.DataFrame:
    Z = standardize(m, dims["ref"]).to_numpy()
    return pd.DataFrame((Z - dims["pca_mean"]) @ dims["components"].T, index=m.index, columns=dims["loadings"].columns)


def group_matrix(profile: pd.DataFrame, rosters: dict[int, pd.DataFrame], fit_group: str) -> pd.DataFrame:
    """The matrix for a fit group: one position, or receivers pooled with an is_te column."""
    parts = FIT_GROUPS[fit_group]
    if len(parts) == 1:
        return matrix(profile, rosters, parts[0])
    return pd.concat([matrix(profile, rosters, g).assign(is_te=float(g == "TE")) for g in parts]).sort_index()
