"""CFB-PROJ P1: team volume. Projects each FBS offense's pass plays and run plays per game, and its games, in season t,
as of May 1 of t (PLAN-fit, CFB-PROJ). DEVELOPMENT ONLY: this module evaluates on seasons 2016-2020; the backtest's
test seasons (2021-2025) are never scored here (never tune on a test fold), and 2026 is the vault.

Per team-season (from the play-by-play through the point-in-time loader; non-garbage-time plays, labels.team_plays):
  pass_pg, run_pg   pass and run plays per game; games = games with at least one play
Inputs for season t: pass_pg / run_pg / pass rate in t-1 and t-2, offensive and defensive EPA per play in t-1, kept head
coach (CFBD coaches as of May 1), kept coordinator (Wikipedia, FIT-OC; not point-in-time, flagged), and for a new head
coach who was a head coach in t-1, his old team's pass rate (a coach's pass-heaviness travels: r 0.53, Phase 1).
Models compared, leave-one-season-out over the development seasons:
  carry      season t-1 carried forward
  shrink     t-1 pulled toward the league mean by the development seasons' year-to-year slope
  ridge      ridge on all inputs

    python cfb/proj/volume.py            # --cached reuses the private team-season table
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from backtest.asof import Loader
from cfb.fit import data as D
from cfb.fit import labels as LB
from sparvi.config import RAW, ROOT

OUT = ROOT / "cfb" / "proj" / "output"
CACHE = RAW / "cfb" / "proj_team_seasons.parquet"      # private (from CFBD plays)
SEASONS = range(2014, 2021)                            # inputs from 2014; development outcomes 2016-2020
DEV = range(2016, 2021)
TARGETS = ("pass_pg", "run_pg", "games")
SEED, N_BOOT = 7, 1000


def team_seasons(L: Loader) -> pd.DataFrame:
    rows = []
    for s in SEASONS:
        ev, _ = D.season_events(L, s, pd.Timestamp(f"{s + 1}-02-01"))
        tp = LB.team_plays(ev).unstack(fill_value=0)
        plays = ev[ev["kind"].isin(["pass", "run"])].drop_duplicates("play_id")
        g = plays.groupby("offense")["game_id"].nunique()
        off = plays.groupby("offense")["epa"].mean()
        dfn = plays.groupby("defense")["epa"].mean()
        d = pd.DataFrame({"season": s, "games": g, "pass": tp.get("pass"), "run": tp.get("run"), "off_epa": off, "def_epa": dfn})
        rows.append(d.rename_axis("team").reset_index())
        print(f"  team volume {s}", flush=True)
    d = pd.concat(rows, ignore_index=True)
    d["pass_pg"], d["run_pg"] = d["pass"] / d["games"], d["run"] / d["games"]
    d["pass_rate"] = d["pass"] / (d["pass"] + d["run"])
    return d


def frame(ts: pd.DataFrame, L: Loader) -> pd.DataFrame:
    """One row per FBS team-season t in DEV with its inputs (as of May 1 of t) and outcomes."""
    k = ts.set_index(["season", "team"])
    co = L.load("coaches", pd.Timestamp(f"{max(DEV)}-05-01"))
    head = co.sort_values("games").groupby(["season", "school"])["coach_id"].last()
    where = {(s, c): sc for (s, sc), c in head.items()}
    oc = pd.read_parquet(RAW / "cfb" / "coordinators.parquet").dropna(subset=["oc"]).set_index(["season", "school"])["oc_all"]
    names = oc.map(lambda v: {n.strip().lower() for n in v.split(";")})
    rows = []
    for t in DEV:
        fbs = set(pd.read_parquet(RAW / "cfb" / f"teams_{t}.parquet")["school"])
        for team in sorted(fbs):
            if (t, team) not in k.index or (t - 1, team) not in k.index:
                continue
            r = {"season": t, "team": team, **{f"y_{c}": k.loc[(t, team), c] for c in TARGETS}}
            for lag in (1, 2):
                prev = k.loc[(t - lag, team)] if (t - lag, team) in k.index else None
                for c in ("pass_pg", "run_pg", "pass_rate", "games"):
                    r[f"{c}_{lag}"] = np.nan if prev is None else prev[c]
            r["off_epa_1"], r["def_epa_1"] = k.loc[(t - 1, team), "off_epa"], k.loc[(t - 1, team), "def_epa"]
            h, hp = head.get((t, team)), head.get((t - 1, team))
            r["new_hc"] = float(h is not None and hp is not None and h != hp)
            old = where.get((t - 1, h)) if r["new_hc"] else None
            r["hc_old_pass_rate"] = k.loc[(t - 1, old), "pass_rate"] if old is not None and (t - 1, old) in k.index else np.nan
            a, b = names.get((t, team)), names.get((t - 1, team))
            r["new_oc"] = np.nan if a is None or b is None else float(not (a & b))
            rows.append(r)
    d = pd.DataFrame(rows)
    for c in ("pass_pg", "run_pg", "pass_rate", "games"):
        d[f"{c}_2"] = d[f"{c}_2"].fillna(d[f"{c}_1"])
    d["has_hc_old"] = d["hc_old_pass_rate"].notna().astype(float)
    d["hc_old_shift"] = (d["hc_old_pass_rate"] - d["pass_rate_1"]).fillna(0.0)
    d["oc_known"] = d["new_oc"].notna().astype(float)
    d["new_oc"] = d["new_oc"].fillna(0.0)
    return d


FEATS = ["pass_pg_1", "run_pg_1", "pass_rate_1", "games_1", "pass_pg_2", "run_pg_2", "pass_rate_2", "off_epa_1", "def_epa_1",
         "new_hc", "has_hc_old", "hc_old_shift", "new_oc", "oc_known"]
LAGGED = {"pass_pg": "pass_pg_1", "run_pg": "run_pg_1", "games": "games_1"}


GAMES_FEATS = ["off_epa_1", "def_epa_1", "new_hc"]       # games: schedule length + postseason, which follows strength
COVID = 2020                                             # short, irregular schedules: not a games outcome or input


def predict(d: pd.DataFrame, target: str) -> dict[str, np.ndarray]:
    """Leave-one-season-out predictions over DEV for each model ("shrink" is the league mean for games)."""
    out = {m: np.full(len(d), np.nan) for m in ("carry", "shrink", "ridge")}
    x1 = d[LAGGED[target]].to_numpy(float)
    y = d[f"y_{target}"].to_numpy(float)
    X = d[GAMES_FEATS if target == "games" else FEATS].to_numpy(float)
    for s in DEV:
        if target == "games":
            if s == COVID:
                continue
            tr = ((d["season"] != s) & (d["season"] != COVID)).to_numpy()
            te = (d["season"] == s).to_numpy()
            out["carry"][te] = np.where(d.loc[te, "season"].to_numpy() - 1 == COVID, y[tr].mean(), x1[te])
            out["shrink"][te] = y[tr].mean()
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd[sd == 0] = 1
            out["ridge"][te] = Ridge(alpha=10.0).fit((X[tr] - mu) / sd, y[tr]).predict((X[te] - mu) / sd)
            continue
        tr, te = (d["season"] != s).to_numpy(), (d["season"] == s).to_numpy()
        out["carry"][te] = x1[te]
        slope = np.polyfit(x1[tr], y[tr], 1)
        out["shrink"][te] = np.polyval(slope, x1[te])
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1
        out["ridge"][te] = Ridge(alpha=10.0).fit((X[tr] - mu) / sd, y[tr]).predict((X[te] - mu) / sd)
    return out


def _boot(d: pd.DataFrame, y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    """MAE(a) - MAE(b) with a 90% interval, resampling teams."""
    rng = np.random.default_rng(SEED)
    teams = d["team"].to_numpy()
    uniq = np.unique(teams)
    pos = {t: np.flatnonzero(teams == t) for t in uniq}
    ea, eb = np.abs(a - y), np.abs(b - y)
    draws = []
    for _ in range(N_BOOT):
        i = np.concatenate([pos[t] for t in rng.choice(uniq, len(uniq))])
        draws.append(ea[i].mean() - eb[i].mean())
    return {"est": round(float(ea.mean() - eb.mean()), 4), "ci90": [round(float(np.percentile(draws, 5)), 4), round(float(np.percentile(draws, 95)), 4)]}


def main() -> None:
    L = Loader()
    ts = pd.read_parquet(CACHE) if "--cached" in sys.argv and CACHE.exists() else team_seasons(L)
    ts.to_parquet(CACHE, index=False)
    d = frame(ts, L)
    assert d["season"].max() <= 2020, "development seasons only"
    res = {"status": "DEVELOPMENT (CFB-PROJ P1): seasons 2016-2020 only; test seasons 2021-2025 not scored", "team_seasons": int(len(d)),
           "new_hc_share": round(float(d["new_hc"].mean()), 4), "oc_known_share": round(float(d["oc_known"].mean()), 4), "targets": {}}
    for target in TARGETS:
        p = predict(d, target)
        y = d[f"y_{target}"].to_numpy(float)
        if target == "games":                                   # 2020 is not scored for games
            keep = (d["season"] != COVID).to_numpy()
            p, y, dd = {m: v[keep] for m, v in p.items()}, y[keep], d[keep]
        else:
            dd = d
        res["targets"][target] = {
            "mean": round(float(y.mean()), 3),
            **{m: {"mae": round(float(np.abs(v - y).mean()), 4),
                  "r": None if np.std(v) == 0 else round(float(np.corrcoef(v, y)[0, 1]), 4)} for m, v in p.items()},
            "ridge_minus_carry_mae": _boot(dd, y, p["ridge"], p["carry"]),
            "ridge_minus_shrink_mae": _boot(dd, y, p["ridge"], p["shrink"]),
        }
        if target != "games":
            Xs = d[FEATS]
            m = Ridge(alpha=10.0).fit((Xs - Xs.mean()) / Xs.std().replace(0, 1), y)
            res["targets"][target]["ridge_coef_std"] = {f: round(float(c), 3) for f, c in zip(FEATS, m.coef_, strict=True)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "volume_dev.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
