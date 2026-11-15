"""The player-profile engine (PLAYER-PROFILE): every feature in features.yml, computed the same way for one season.

For each feature and player:
  raw        the plain average (or share) over his qualifying opportunities
  adj        opponent-adjusted (adjust: opponent): each play's value minus what that defense allowed in this same stat,
             from a team-level additive fit value ~ offense + defense over every play in the season (ridge, so small
             samples are pulled to average). Because the schedule links every team, a defense's number already reflects
             the offenses it faced, which is what translates production across conferences and levels.
  shrunk     empirical Bayes toward his position group's average: shrunk = mean + (adj - mean) * reliability_i, where
             reliability_i = tau^2 / (tau^2 + sigma^2 / n_i). tau^2 is the real spread between players, sigma^2 the
             play-to-play noise, both estimated per feature, position, and season. Unreliable stats move a lot; stable
             ones barely move.
  pct        percentile within the position group, among players with min_n opportunities; flipped for better: lower so
             that higher always means better, left as is for style stats.

Output: one row per season x player x feature (long), so new features and sources add rows, not code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from scipy.sparse.linalg import spsolve

FEATURES = Path(__file__).with_name("features.yml")
REQUIRED = {"id", "label", "group", "opp", "kind", "adjust", "better", "min_n"}
POSITION_GROUP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}
TEAM_RIDGE = 30.0          # team-level fit: each team's effect is shrunk as if it had this many average plays


def load_features(path: Path = FEATURES) -> list[dict]:
    feats = yaml.safe_load(path.read_text(encoding="utf-8"))["features"]
    ids = set()
    for f in feats:
        missing = REQUIRED - set(f)
        if missing:
            raise ValueError(f"feature {f.get('id')}: missing {sorted(missing)}")
        if f["id"] in ids:
            raise ValueError(f"duplicate feature id {f['id']}")
        ids.add(f["id"])
        if f["kind"] == "mean" and "value" not in f:
            raise ValueError(f"feature {f['id']}: kind mean needs a value")
        if f["kind"] == "share" and "of" not in f:
            raise ValueError(f"feature {f['id']}: kind share needs `of`")
        if f["adjust"] not in ("opponent", "none") or f["better"] not in ("higher", "lower", "style"):
            raise ValueError(f"feature {f['id']}: bad adjust/better")
    return feats


def opponent_effects(rows: pd.DataFrame, value: np.ndarray) -> dict[str, float]:
    """Defense effects from value ~ mean + offense + defense (ridge on team-game cells, weighted by plays)."""
    cells = pd.DataFrame({"o": rows["offense"].to_numpy(), "d": rows["defense"].to_numpy(), "v": value})
    cells = cells.groupby(["o", "d"]).agg(n=("v", "size"), s=("v", "sum")).reset_index()
    mean = cells["s"].sum() / cells["n"].sum()
    offs, defs = np.sort(cells["o"].unique()), np.sort(cells["d"].unique())
    oi = pd.Categorical(cells["o"], categories=offs).codes
    di = pd.Categorical(cells["d"], categories=defs).codes + len(offs)
    k = len(offs) + len(defs)
    rows_i = np.repeat(np.arange(len(cells)), 2)
    X = sp.csr_matrix((np.ones(2 * len(cells)), (rows_i, np.ravel(np.column_stack([oi, di])))), shape=(len(cells), k))
    w = cells["n"].to_numpy(float)
    y = cells["s"].to_numpy(float) / w - mean
    W = sp.diags(w)
    b = spsolve((X.T @ W @ X + TEAM_RIDGE * sp.identity(k)).tocsc(), X.T @ (w * y))
    d = b[len(offs):]
    plays_vs = cells.groupby("d")["n"].sum().reindex(defs).to_numpy(float)
    d = d - (plays_vs * d).sum() / plays_vs.sum()      # centered: the average play faces a zero adjustment
    return dict(zip(defs, d, strict=True))


def _feature_rows(ev: pd.DataFrame, f: dict, season: int) -> pd.DataFrame | None:
    if season < f.get("from", 0):
        return None
    r = ev[ev["opp"] == f["opp"]]
    if f.get("filter"):
        cols = set(r.columns)
        if not all(c in cols for c in _names(f["filter"])):
            return None
        r = r.query(f["filter"])
    return r if len(r) else None


def _names(expr: str) -> set[str]:
    import re
    return {t for t in re.findall(r"[A-Za-z_]\w*", expr) if t not in {"and", "or", "not", "True", "False"}}


def _shrink(df: pd.DataFrame, noise_var: float) -> pd.DataFrame:
    """Empirical Bayes within one position group: adds shrunk and reliability."""
    x, n = df["adj"].to_numpy(float), df["n"].to_numpy(float)
    w = n / n.sum()
    mu = float((w * x).sum())
    tau2 = max(float((w * (x - mu) ** 2).sum() - (w * noise_var / n).sum()), 1e-9)
    rel = tau2 / (tau2 + noise_var / n)
    return df.assign(shrunk=mu + (x - mu) * rel, reliability=rel)


def season_profile(ev: pd.DataFrame, positions: pd.Series, season: int, feats: list[dict] | None = None) -> pd.DataFrame:
    """Long profile table for one season's events. positions: player_id -> roster position."""
    feats = feats or load_features()
    group = ev["player_id"].map(positions).map(POSITION_GROUP)
    ev = ev.assign(group=group.to_numpy())
    games = ev.groupby("player_id")["game_id"].agg(lambda s: set(s))
    team_of = ev.groupby("player_id")["offense"].agg(lambda s: s.mode().iloc[0])
    out = []
    for f in feats:
        r = _feature_rows(ev, f, season)
        if r is None:
            continue
        r = r[r["group"].isin(f["group"])]
        if f["kind"] == "mean":
            if f["value"] not in r.columns and not _names(f["value"]) <= set(r.columns):
                continue
            v = r.eval(f["value"]).astype(float).to_numpy() if f["value"] not in r.columns else r[f["value"]].to_numpy(float)
            ok = ~np.isnan(v)
            r, v = r[ok], v[ok]
            if not len(r):
                continue
            if f["adjust"] == "opponent":
                full = _feature_rows(ev, f, season)                       # every team's plays, not just this group
                fv = full.eval(f["value"]).astype(float).to_numpy() if f["value"] not in full.columns else full[f["value"]].to_numpy(float)
                keep = ~np.isnan(fv)
                eff = opponent_effects(full[keep], fv[keep])
                adj = v - r["defense"].map(eff).fillna(0.0).to_numpy()
            else:
                adj = v
            g = pd.DataFrame({"player_id": r["player_id"].to_numpy(), "group": r["group"].to_numpy(), "v": v, "a": adj})
            agg = g.groupby(["player_id", "group"]).agg(n=("v", "size"), raw=("v", "mean"), adj=("a", "mean")).reset_index()
            noise = float(g.groupby("player_id")["a"].var().dropna().mean()) if len(g) > 2 else 0.0
            agg = pd.concat([_shrink(x, noise) for _, x in agg.groupby("group")], ignore_index=True)
        else:   # share of the team's opportunities of type `of`, in the games he played
            # The team's plays of that kind (pass plays for dropbacks/targets, runs for carries), counted by play id
            # whether or not every player on them was matched: a dropback count would undercount teams whose passers
            # aren't matched (v1.0.1 fix; shares above 1 appeared).
            kind = "run" if f["of"] == "carry" else "pass"
            team_game = ev[ev["kind"] == kind].drop_duplicates("play_id").groupby(["offense", "game_id"]).size()
            num = r.groupby(["player_id", "group"]).size().rename("n").reset_index()
            den = [sum(team_game.get((team_of[p], gid), 0) for gid in games[p]) for p in num["player_id"]]
            num["raw"] = np.minimum(num["n"] / np.maximum(den, 1), 1.0)
            num["adj"] = num["shrunk"] = num["raw"]
            num["reliability"] = np.nan
            agg = num
        agg["feature"], agg["season"] = f["id"], season
        q = agg["n"] >= f["min_n"]
        pct = agg.loc[q].groupby("group")["shrunk"].rank(pct=True) * 100
        agg["pct"] = np.nan
        agg.loc[q, "pct"] = (100 - pct) if f["better"] == "lower" else pct
        out.append(agg)
    prof = pd.concat(out, ignore_index=True)
    prof["team"] = prof["player_id"].map(team_of)
    return prof[["season", "player_id", "group", "team", "feature", "n", "raw", "adj", "shrunk", "reliability", "pct"]]
