"""PLAYER-PROFILE validation file (cfb/profile/output/validation.json): the pre-registered primary result and the
secondary results in the model card, from the same folds as models/cfb/player_profile/backtest.py, run locally on the
raw files. The backtest workflow's run is the official one for the gate; this is what the card and registry read.

    movers       players who changed teams: r(season t-1 value, season t adjusted value), adjusted+shrunk vs raw,
                 pooled over opponent-adjusted features (standardized within feature), with a player-block bootstrap
    all          the same for every player
    raw_target   the movers test against the season t raw value instead
    by_feature   per feature: n, r adjusted, r raw (movers and all)
    split_half   odd vs. even games within a season: mean over features of r(half A, half B), adjusted vs raw
    opponent     player-game values vs. the opponent's allowed level in that stat: mean |r| over features, raw vs adjusted

    python cfb/profile/validate.py
"""
from __future__ import annotations

import importlib.util
import json

import numpy as np
import pandas as pd

from backtest.asof import Loader
from backtest.splits import season_folds
from cfb.profile import engine as G
from sparvi.config import ROOT

OUT = ROOT / "cfb" / "profile" / "output"
BOOT, SEED = 2000, 7


def _mod():
    spec = importlib.util.spec_from_file_location("pp_bt", ROOT / "models" / "cfb" / "player_profile" / "backtest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def r(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")


def compare(df: pd.DataFrame, y: str, rng) -> dict:
    """Adjusted (point) vs raw (last_season) against y, with a 90% interval from resampling players."""
    players = df["player"].unique()
    idx = {p: np.flatnonzero(df["player"].to_numpy() == p) for p in players}
    draws = []
    for _ in range(BOOT):
        pick = np.concatenate([idx[p] for p in rng.choice(players, len(players))])
        s = df.iloc[pick]
        draws.append(r(s["point"], s[y]) - r(s["last_season"], s[y]))
    return {"n": int(len(df)), "n_players": int(len(players)), "adjusted": {"r": round(r(df["point"], df[y]), 4)},
            "raw": {"r": round(r(df["last_season"], df[y]), 4)},
            "edge": {"est": round(r(df["point"], df[y]) - r(df["last_season"], df[y]), 4),
                     "ci90": [round(float(v), 4) for v in np.percentile(draws, [5, 95])]}}


def by_feature(df: pd.DataFrame, y: str) -> dict:
    return {f: {"n": int(len(g)), "adjusted": round(r(g["point"], g[y]), 3), "raw": round(r(g["last_season"], g[y]), 3)}
            for f, g in df.groupby("feature")}


def split_half(mod, loader, s: int, as_of) -> dict:
    ev, pos = mod._events(loader, s, as_of)
    games = np.sort(ev["game_id"].unique())
    half = ev["game_id"].map(dict(zip(games, np.arange(len(games)) % 2, strict=True)))
    adjusted = {f["id"]: f["min_n"] for f in G.load_features() if f["adjust"] == "opponent"}
    halves = [G.season_profile(ev[half == h], pos, s) for h in (0, 1)]
    d = halves[0].merge(halves[1], on=["player_id", "feature"], suffixes=("_a", "_b"))
    d = d[d["feature"].isin(adjusted) & (d["n_a"] >= d["feature"].map(adjusted) / 2) & (d["n_b"] >= d["feature"].map(adjusted) / 2)]
    out = {}
    for name, c in (("adjusted", "shrunk"), ("raw", "raw")):
        out[name] = round(float(np.nanmean([r(g[f"{c}_a"], g[f"{c}_b"]) for _, g in d.groupby("feature")])), 4)
    return out | {"season": s, "n": int(len(d))}


def opponent(mod, loader, s: int, as_of) -> dict:
    """Player-game averages vs. the opponent's allowed level in the same stat (its defense effect), per feature."""
    ev, pos = mod._events(loader, s, as_of)
    ev = ev.assign(group=ev["player_id"].map(pos).map(G.POSITION_GROUP))
    res = {"raw": [], "adjusted": []}
    for f in G.load_features():
        if f["adjust"] != "opponent" or f["kind"] != "mean":
            continue
        rows = G._feature_rows(ev, f, s)
        if rows is None:
            continue
        v = rows.eval(f["value"]).astype(float).to_numpy() if f["value"] not in rows.columns else rows[f["value"]].to_numpy(float)
        ok = ~np.isnan(v)
        rows, v = rows[ok], v[ok]
        eff = rows["defense"].map(G.opponent_effects(rows, v)).fillna(0.0).to_numpy()
        g = pd.DataFrame({"pg": rows["player_id"] + "|" + rows["game_id"].astype(str), "raw": v, "adjusted": v - eff, "eff": eff,
                          "grp": rows["group"].isin(f["group"]).to_numpy()})
        g = g[g["grp"]].groupby("pg").agg(raw=("raw", "mean"), adjusted=("adjusted", "mean"), eff=("eff", "mean"), n=("raw", "size"))
        g = g[g["n"] >= 3]
        for k in ("raw", "adjusted"):
            res[k].append(abs(r(g[k], g["eff"])))
    return {k: round(float(np.nanmean(v)), 4) for k, v in res.items()} | {"season": s}


def main() -> None:
    mod = _mod()
    rng = np.random.default_rng(SEED)
    folds = season_folds(2021, 2025, 1, "02-01", 1)
    rows = []
    for fold in folds:
        loader = Loader()
        pred = mod.predict(mod.fit(loader, fold), loader, fold)
        pr = mod.pairs(loader, fold)
        d = pred.merge(pr[["entity_id", "adj", "raw", "mover"]].rename(columns={"adj": "y", "raw": "y_raw"}), on="entity_id")
        rows.append(d.assign(season=fold.test_season))
        m = d[d["mover"]]
        print(f"{fold.test_season}: movers {len(m)} rows ({m.player.nunique()} players) adj r {r(m.point, m.y):.3f} raw r {r(m.last_season, m.y):.3f}"
              f" | all {len(d)} adj {r(d.point, d.y):.3f} raw {r(d.last_season, d.y):.3f}")
    d = pd.concat(rows, ignore_index=True)
    m = d[d["mover"]]
    res = {"seasons": {"first": folds[0].test_season, "last": folds[-1].test_season},
           "movers": compare(m, "y", rng), "all": compare(d, "y", rng), "raw_target": compare(m, "y_raw", rng),
           "by_feature": {"movers": by_feature(m, "y"), "all": by_feature(d, "y")}}
    loader, last = Loader(), folds[-1]
    res["split_half"] = split_half(mod, loader, last.test_season - 1, last.decision_date)
    res["opponent"] = opponent(mod, loader, last.test_season - 1, last.decision_date)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in res.items() if k != "by_feature"}, indent=1))


if __name__ == "__main__":
    main()
