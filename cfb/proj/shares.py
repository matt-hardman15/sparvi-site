"""CFB-PROJ P2: projected rosters and shares. For each FBS team, position group, and season t, as of May 1 of t: who
competes for the position's work, and each one's share of it (PLAN-fit, CFB-PROJ). DEVELOPMENT ONLY: evaluated on
seasons 2016-2020; the test seasons (2021-2025) are never scored here, and 2026 is the vault.

Work per group (season t, non-garbage-time plays): QB dropbacks, RB carries, WR/TE receiving yards (completions only:
2021-2024 incompletions mostly lack the receiver, CFB-RECV-FIX).
Candidates (the projected roster):
  returner   on the team's t-1 roster at the position, weighted by P(stays | experience) estimated on earlier seasons
  transfer   development seasons: moves found from the September rosters (hindsight; no portal before 2021, decided by
             Matt Nov 15); test seasons will use portal entries filed by May 1
  freshman   the team's high-school signing class at the position (linked to a roster id)
Players who take work but weren't projected (walk-ons, late adds, unlinked recruits) are the "unprojected" outside
option, so each team-group's projected shares plus the outside option sum to 1.

Model (per group): conditional logit with an outside option, fit by soft-label cross-entropy on actual shares.
Baselines (pre-registered in PLAN-fit): carry = last season's share at this team (0 for newcomers); share_only = last
season's share wherever he played (transfers carry theirs), no competition.

    python cfb/proj/shares.py            # --cached reuses the private candidate table
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from backtest.asof import Loader
from cfb.fit import data as D
from cfb.fit import labels as LB
from cfb.profile import engine as G
from sparvi.config import RAW, ROOT

OUT = ROOT / "cfb" / "proj" / "output"
P = RAW / "cfb" / "players"
CACHE = RAW / "cfb" / "proj_candidates.parquet"         # private (CFBD rosters and plays)
DEV = range(2016, 2021)
GROUPS = ("QB", "RB", "REC")                            # pools: QB dropbacks, RB carries, receiving yards (WR + TE + RB)
POOLS_OF = {"QB": ("QB",), "RB": ("RB", "REC"), "WR": ("REC",), "TE": ("REC",)}
RECRUIT_POS = {"PRO": "QB", "DUAL": "QB", "QB": "QB", "RB": "RB", "APB": "RB", "WR": "WR", "TE": "TE"}
FEATS = ["sq_share", "had_role", "exp", "exp2", "transfer", "freshman", "log_stay", "hs", "no_hs", "dlvl", "eff", "no_eff",
         "sq_x_transfer", "sq_x_dlvl", "is_te", "is_rb"]
TRANSFER_FEATS = ["open_x_transfer", "open_x_freshman", "tr_dlvl", "tr_eff", "tr_no_eff", "tr_hs", "tr_exp"]
L2, SEED, N_BOOT = 1.0, 7, 1000


def work(ev: pd.DataFrame, pos: pd.Series) -> pd.DataFrame:
    """Per (team, player, pool): his amount of the pool's work and its share of the team's total. Pools: QB dropbacks,
    RB carries (by RBs), REC receiving yards on completions (by WRs, TEs, and RBs, of the team's completion yards)."""
    grp = ev["player_id"].map(pos).map(G.POSITION_GROUP)
    comp = (ev["opp"] == "target") & (ev["complete"] == 1)
    out = []
    for g in GROUPS:
        if g == "QB":
            r, amt = ev[(ev["opp"] == "dropback") & (grp == "QB")], None
        elif g == "RB":
            r, amt = ev[(ev["opp"] == "carry") & (grp == "RB")], None
        else:
            r, amt = ev[comp & grp.isin(["WR", "TE", "RB"])], "yards"
        tot_src = {"QB": ev[ev["opp"] == "dropback"], "RB": ev[(ev["opp"] == "carry") & (grp == "RB")]}.get(g, ev[comp])
        if amt:
            n = r.groupby(["offense", "player_id"])["yards"].sum().clip(lower=0)
            tot = tot_src.groupby("offense")["yards"].sum().clip(lower=1)
        else:
            n = r.groupby(["offense", "player_id"]).size()
            tot = tot_src.groupby("offense").size()
        d = n.rename("amount").reset_index().rename(columns={"offense": "team"})
        d["share"] = (d["amount"] / d["team"].map(tot)).clip(0, 1)
        out.append(d.assign(group=g))
    return pd.concat(out, ignore_index=True)


def efficiency(ev: pd.DataFrame, pos: pd.Series) -> pd.Series:
    """(player, group) -> his EPA per opportunity last season (dropbacks / carries / completions), min 20."""
    grp = ev["player_id"].map(pos).map(G.POSITION_GROUP)
    out = {}
    comp = (ev["opp"] == "target") & (ev["complete"] == 1)
    for g, m in (("QB", (ev["opp"] == "dropback") & (grp == "QB")), ("RB", (ev["opp"] == "carry") & (grp == "RB")),
                 ("REC", comp & grp.isin(["WR", "TE", "RB"]))):
        r = ev[m].groupby("player_id")["epa"].agg(["mean", "size"])
        r = r[r["size"] >= 20]["mean"]
        z = (r - r.mean()) / r.std()
        out.update({(p, g): v for p, v in z.items()})
    return pd.Series(out)


def candidates() -> pd.DataFrame:
    L = Loader()
    seasons = range(min(DEV) - 2, max(DEV) + 1)
    rosters = {s: pd.read_parquet(P / f"roster_{s}.parquet") for s in seasons}
    recruits = pd.concat([pd.read_parquet(P / f"recruits_{s}.parquet") for s in range(2010, max(DEV) + 1)
                          if (P / f"recruits_{s}.parquet").exists()], ignore_index=True)
    evs, wk, played, eff = {}, {}, {}, {}
    for s in seasons:
        ev, pos = D.season_events(L, s, pd.Timestamp(f"{s + 1}-02-01"))
        evs[s] = (ev, pos)
        wk[s] = work(ev, pos)
        played[s] = ev.groupby("player_id")["offense"].agg(lambda x: x.mode().iloc[0]).to_dict()
        eff[s] = efficiency(ev, pos)
        print(f"  events {s}", flush=True)
    hs = (recruits[recruits["athleteId"].notna()].assign(pid=lambda d: d["athleteId"].astype(str))
          .sort_values(["year", "rating"], ascending=[True, False]).drop_duplicates("pid").set_index("pid")["rating"])
    rows = []
    for t in DEV:
        conf_prev = dict(pd.read_parquet(RAW / "cfb" / f"teams_{t - 1}.parquet")[["school", "conference"]].values)
        conf_t = dict(pd.read_parquet(RAW / "cfb" / f"teams_{t}.parquet")[["school", "conference"]].values)
        exp = D.experience(t - 1, {s: rosters[s] for s in seasons if s <= t - 1}, recruits)
        # P(not on the same team next season | experience), from the two seasons before t-1 -> t.
        hist = []
        for h in [h for h in (t - 2, t - 1) if h - 1 in rosters]:
            r0 = rosters[h - 1].assign(pid=lambda d: d["id"].astype(str))
            r1 = set(zip(rosters[h]["team"], rosters[h]["id"].astype(str), strict=True))
            e0 = D.experience(h - 1, {s: rosters[s] for s in rosters if s <= h - 1}, recruits)
            r0 = r0[r0["position"].map(G.POSITION_GROUP).notna()]
            hist.append(pd.DataFrame({"exp": r0["pid"].map(e0).fillna(1).clip(1, 6).astype(int),
                                      "gone": [(tm, p) not in r1 for tm, p in zip(r0["team"], r0["pid"], strict=True)]}))
        p_leave = pd.concat(hist).groupby("exp")["gone"].mean().to_dict()
        prev_w = wk[t - 1].set_index(["team", "player_id", "group"])["share"]
        any_w = wk[t - 1].sort_values("share", ascending=False).drop_duplicates(["player_id", "group"]).set_index(["player_id", "group"])["share"]
        actual = wk[t].set_index(["team", "player_id", "group"])["share"]
        tr = LB.transfers(rosters, t, None, played)
        tr = tr[tr["team_to"].isin(conf_t)]
        ros_prev = rosters[t - 1].assign(pid=lambda d: d["id"].astype(str), group=lambda d: d["position"].map(G.POSITION_GROUP))
        ros_prev = ros_prev[ros_prev["group"].notna() & ros_prev["team"].isin(conf_t)].drop_duplicates(["team", "pid"])
        moved_out = set(tr["prev_id"].dropna().astype(str))
        cand = []
        for r in ros_prev.itertuples():
            e = int(np.clip(exp.get(r.pid, 1), 1, 6))
            cand.append({"team": r.team, "player_id": r.pid, "out_id": r.pid, "pos": r.group, "status": "returner", "exp": e,
                         "log_stay": float(np.log(max(1 - p_leave.get(e, 0.3), 0.02))), "dlvl": 0.0, "moved_out": r.pid in moved_out})
        lvl = {"FCS": 0, "G5": 1, "Power": 2}
        for r in tr.itertuples():
            pid = str(r.prev_id) if pd.notna(r.prev_id) else None
            cand.append({"team": r.team_to, "player_id": pid, "out_id": str(r.player_id), "pos": r.group, "status": "transfer",
                         "exp": int(np.clip(exp.get(pid, 2) if pid else 2, 1, 6)), "log_stay": 0.0,
                         "dlvl": float(lvl[LB.tier(r.team_to, t, conf_t)] - lvl[LB.tier(r.team_from, t - 1, conf_prev)]), "moved_out": False})
        fr = recruits[(recruits["year"] == t) & recruits["committedTo"].isin(conf_t) & recruits["athleteId"].notna()]
        for r in fr.itertuples():
            g = RECRUIT_POS.get(r.position)
            if g:
                pid = str(int(r.athleteId))
                cand.append({"team": r.committedTo, "player_id": pid, "out_id": pid, "pos": g, "status": "freshman",
                             "exp": 0, "log_stay": 0.0, "dlvl": 0.0, "moved_out": False})
        c = pd.DataFrame(cand)
        # Development: a returner who is known (from the September roster) to have transferred out is not on the roster.
        c = c[~(c["moved_out"] & (c["status"] == "returner"))].drop(columns="moved_out")
        c = c.drop_duplicates(["team", "pos", "out_id"])
        c = c.assign(group=c["pos"].map(POOLS_OF)).explode("group").reset_index(drop=True)
        c["last_share"] = [0.0 if st == "freshman" or p is None else
                           (prev_w.get((tm, p, g), 0.0) if st == "returner" else any_w.get((p, g), 0.0))
                           for tm, p, g, st in zip(c["team"], c["player_id"], c["group"], c["status"], strict=True)]
        c["eff"] = [eff[t - 1].get((p, g), np.nan) if p else np.nan for p, g in zip(c["player_id"], c["group"], strict=True)]
        c["actual"] = [actual.get((tm, o, g), 0.0) for tm, o, g in zip(c["team"], c["out_id"], c["group"], strict=True)]
        c["any_last"] = [any_w.get((p, g), 0.0) if p else 0.0 for p, g in zip(c["player_id"], c["group"], strict=True)]
        c["hs"] = c["player_id"].map(hs)
        c["season"] = t
        # Work done by players nobody projected (per team-group): the outside option's actual share.
        tot = c.groupby(["team", "group"])["actual"].sum()
        c["unprojected"] = [max(0.0, 1 - tot[(tm, g)]) for tm, g in zip(c["team"], c["group"], strict=True)]
        rows.append(c)
        print(f"  candidates {t}: {len(c):,}", flush=True)
    return pd.concat(rows, ignore_index=True)


def design(c: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=c.index)
    x["sq_share"] = np.sqrt(c["last_share"].clip(lower=0))
    x["had_role"] = (c["last_share"] >= 0.1).astype(float)
    x["exp"], x["exp2"] = c["exp"].astype(float), c["exp"].astype(float) ** 2
    x["transfer"], x["freshman"] = (c["status"] == "transfer").astype(float), (c["status"] == "freshman").astype(float)
    x["log_stay"], x["dlvl"] = c["log_stay"], c["dlvl"]
    x["no_hs"] = c["hs"].isna().astype(float)
    x["hs"] = c["hs"].fillna(c["hs"].min())
    x["no_eff"] = c["eff"].isna().astype(float)
    x["eff"] = c["eff"].fillna(0.0)
    x["sq_x_transfer"] = x["sq_share"] * x["transfer"]
    x["sq_x_dlvl"] = x["sq_share"] * x["dlvl"]
    x["is_te"], x["is_rb"] = (c["pos"] == "TE").astype(float), (c["pos"] == "RB").astype(float)
    # Transfers get their own terms (returners are 85%+ of rows and would otherwise set every coefficient), and the
    # destination's opening: 1 - the returners' expected share (last share x P(stays)), the same for a whole team, so
    # it acts only through newcomers vs. returners.
    ret = c["status"] == "returner"
    kept = (c["last_share"] * np.exp(c["log_stay"])).where(ret, 0.0).groupby([c["season"], c["team"], c["group"]]).transform("sum")
    x["open"] = (1 - kept).clip(0, 1)
    x["open_x_transfer"], x["open_x_freshman"] = x["open"] * x["transfer"], x["open"] * x["freshman"]
    for f in ("dlvl", "eff", "no_eff", "hs", "exp"):
        x[f"tr_{f}"] = x[f] * x["transfer"]
    return x[FEATS + TRANSFER_FEATS]


def fit(X: np.ndarray, keys: np.ndarray, y: np.ndarray, y0: dict) -> np.ndarray:
    """Conditional logit with an outside option (utility = the last coefficient), soft labels: minimize
    -sum_team [sum_i y_i log p_i + y_0 log p_0] + L2 |beta|^2."""
    uniq, inv = np.unique(keys, return_inverse=True)
    y0v = np.array([y0[k] for k in uniq])

    def loss(b):
        u = X @ b[:-1]
        m = np.maximum(np.full(len(uniq), b[-1]), np.maximum.reduceat(u[np.argsort(inv, kind="stable")], _starts(inv)))
        e = np.exp(u - m[inv])
        den = np.bincount(inv, e, minlength=len(uniq)) + np.exp(b[-1] - m)
        logp = (u - m[inv]) - np.log(den[inv])
        logp0 = (b[-1] - m) - np.log(den)
        val = -(y * logp).sum() - (y0v * logp0).sum() + L2 * (b[:-1] ** 2).sum()
        p, p0 = np.exp(logp), np.exp(logp0)
        tot = np.bincount(inv, y, minlength=len(uniq)) + y0v
        g_u = -(y - tot[inv] * p)
        grad = np.append(X.T @ g_u + 2 * L2 * b[:-1], -(y0v - tot * p0).sum())
        return val, grad
    return minimize(loss, np.zeros(X.shape[1] + 1), jac=True, method="L-BFGS-B").x


def _starts(inv: np.ndarray) -> np.ndarray:
    s = np.sort(inv, kind="stable")
    return np.flatnonzero(np.r_[True, s[1:] != s[:-1]])


def predict(X: np.ndarray, keys: np.ndarray, b: np.ndarray) -> np.ndarray:
    u = X @ b[:-1]
    df = pd.DataFrame({"k": keys, "u": u})
    m = np.maximum(df.groupby("k")["u"].transform("max").to_numpy(), b[-1])
    e = np.exp(u - m)
    den = pd.Series(e).groupby(keys).transform("sum").to_numpy() + np.exp(b[-1] - m)
    return e / den


def _boot(d: pd.DataFrame, a: str, b: str) -> dict:
    """r(a, actual) - r(b, actual), 90% interval resampling players."""
    rng = np.random.default_rng(SEED)
    keys = d["out_id"].to_numpy()
    uniq = np.unique(keys)
    pos = {k: np.flatnonzero(keys == k) for k in uniq}
    r = lambda z, c: np.corrcoef(z[c], z["actual"])[0, 1]  # noqa: E731
    draws = []
    for _ in range(N_BOOT):
        i = np.concatenate([pos[k] for k in rng.choice(uniq, len(uniq))])
        z = d.iloc[i]
        draws.append(r(z, a) - r(z, b))
    return {"est": round(float(r(d, a) - r(d, b)), 4), "ci90": [round(float(np.percentile(draws, 5)), 4), round(float(np.percentile(draws, 95)), 4)]}


def main() -> None:
    c = pd.read_parquet(CACHE) if "--cached" in sys.argv and CACHE.exists() else candidates()
    c.to_parquet(CACHE, index=False)
    assert c["season"].max() <= 2020, "development seasons only"
    c = c.reset_index(drop=True)
    X = design(c)
    c["pred"] = np.nan
    coefs = {}
    for g in GROUPS:
        for s in DEV:
            tr, te = ((c["group"] == g) & (c["season"] != s)).to_numpy(), ((c["group"] == g) & (c["season"] == s)).to_numpy()
            mu, sd = X[tr].mean(), X[tr].std().replace(0, 1)
            Z = ((X - mu) / sd).to_numpy()
            keys = (c["season"].astype(str) + "|" + c["team"]).to_numpy()
            y0 = c[tr].groupby(keys[tr])["unprojected"].first().to_dict()
            b = fit(Z[tr], keys[tr], c.loc[tr, "actual"].to_numpy(), y0)
            c.loc[te, "pred"] = predict(Z[te], keys[te], b)
            coefs.setdefault(g, []).append(b)
    c["carry"] = np.where(c["status"] == "returner", c["last_share"], 0.0)
    c["share_only"] = np.where(c["status"] == "freshman", 0.0, c["any_last"])
    res = {"status": "DEVELOPMENT (CFB-PROJ P2): seasons 2016-2020 only; test seasons 2021-2025 not scored", "groups": {}}
    for g in GROUPS:
        d = c[c["group"] == g]
        out = {"rows": int(len(d)), "unprojected_share_mean": round(float(d.groupby(["season", "team"])["unprojected"].first().mean()), 4),
               "coef_std": dict(zip([*FEATS, *TRANSFER_FEATS, "outside"], np.round(np.mean(coefs[g], axis=0), 3).tolist(), strict=True))}
        for m in ("pred", "carry", "share_only"):
            out[m] = {"r": round(float(np.corrcoef(d[m], d["actual"])[0, 1]), 4), "mae": round(float((d[m] - d["actual"]).abs().mean()), 5)}
        out["model_minus_carry_r"] = _boot(d, "pred", "carry")
        out["model_minus_share_only_r"] = _boot(d, "pred", "share_only")
        if g == "REC":
            out["by_position"] = {pp: {"n": int(len(q)), **{m: round(float(np.corrcoef(q[m], q["actual"])[0, 1]), 4)
                                                             for m in ("pred", "carry", "share_only")}} for pp, q in d.groupby("pos")}
        q = d[d["status"] == "transfer"]
        out["transfers_model_minus_share_only_r"] = _boot(q, "pred", "share_only")
        out["by_status"] = {}
        for st, q in d.groupby("status"):
            out["by_status"][st] = {"n": int(len(q)), "actual_mean": round(float(q["actual"].mean()), 4),
                                    **{m: round(float(np.corrcoef(q[m], q["actual"])[0, 1]), 4) if q[m].std() > 0 else None
                                       for m in ("pred", "carry", "share_only")}}
        res["groups"][g] = out
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shares_dev.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
