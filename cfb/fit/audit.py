"""SYSTEM-FIT Phase 1b, FIT-AUDIT (PLAN-fit, Matt Nov 15: the training benchmark may be flawed; coaches chase stats or
settle). EXPLORATORY: every transfer here has been seen; the 2026 vault is not read. Definitions were fixed in PLAN-fit
before this ran. Writes cfb/fit/output/audit.json (aggregates only).

  D-A  Do coaches sign for fit? For each actual transfer, the fit rank of his actual destination among the FBS programs
       at the destination's level (0 = his best fit there, 1 = his worst). If fit played no part, the mean is 0.5.
  D-B  The agreement test (Matt). Fit rank is within the player, so his talent is held fixed. Outcome: value over
       replacement in year 1 (A) and the best of years 2-3 (B), against a quality-only expectation (walk-forward ridge
       on earlier cohorts, explore.BASE). Categories: beat / fell short / didn't play (A: below the backup minimum in
       year 1; B: no value in years 2-3, left or didn't play). The model agrees when a good-fit move (best third) beat
       its expectation or a poor-fit move (worst third) fell short or didn't play.
  D-C  How high is the bar? Label rates, and how well open share alone (and last season's efficiency, share) separates
       success (AUC).
  D-D  Teammate context. EPA per opportunity raw, opponent-adjusted, teammate-relative (receivers vs. the same passers'
       other targets, backs vs. the team's other backs), and both; year-to-year stability for players who stayed, and
       for transfers how well each at the old school predicts opponent-adjusted EPA at the new one.

Fit measures:
  f4      the F4 model's probability (cfb_system_fit, walk-forward fold t: cohorts through t-2), windows 2021-2025
  f4_sys  the F4 model's system terms only (system dimensions and type x system products: the "fit" part of its score)
  style   Phase 1's style distance between his old and new team's systems (closer = better fit), cohorts 2019-2025

    python cfb/fit/audit.py            # --cached reuses the private intermediate tables
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from backtest.asof import Loader
from cfb.fit import data as D
from cfb.fit import explore as X
from cfb.fit import f3_report
from cfb.fit import labels as LB
from cfb.fit import model as M
from cfb.profile import engine as G
from sparvi.config import RAW, ROOT

OUT = ROOT / "cfb" / "fit" / "output"
CACHE = RAW / "cfb"                                     # private intermediates (built from CFBD rosters and plays)
F4_WINDOWS = range(2021, 2026)                          # 2025 on year 1 only; no 2026 outcome is read
SYS_COLS = [f"sys{j + 1}" for j in range(M.N_SYS)] + [f"ty{i + 1}_x_sys{j + 1}" for i in range(M.N_TYPE) for j in range(M.N_SYS)]
GROUPS = ("QB", "RB", "WR", "TE")
OPP = {"QB": "dropback", "RB": "carry", "WR": "target", "TE": "target"}


def _decision(t: int) -> pd.Timestamp:
    return pd.Timestamp(f"{t}-05-01")


def _conf(s: int) -> dict[str, str]:
    return dict(pd.read_parquet(RAW / "cfb" / f"teams_{s}.parquet")[["school", "conference"]].values)


# ---- fit scores for every (transfer, FBS program) pair ---------------------------------------------------------------
def f4_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Per F4 window: every actual transfer (by his previous id) x every FBS program, the model's probability and its
    system-term contribution, from fold t's walk-forward fit as of May 1."""
    L = Loader()
    rows = []
    for t in F4_WINDOWS:
        as_of = _decision(t)
        fitted = M.fit_fold(L, t, as_of)
        teams_t = L.load(f"teams_{t}", as_of, columns=["school", "conference"])
        conf_t = dict(zip(teams_t["school"], teams_t["conference"], strict=True))
        fbs = sorted(conf_t)
        tr = df[df["transfer_season"] == t].drop_duplicates("prev_id")
        pid = pd.Series(np.repeat(tr["prev_id"].astype(str).to_numpy(), len(fbs)))
        team = pd.Series(np.tile(fbs, len(tr)))
        x = M.pairs(M.player_frame(L, t, as_of, fitted["dims"]), M.team_frame(L, t, as_of, fitted["dims"], conf_t), pid, team)
        grp = np.repeat(tr["group"].to_numpy(), len(fbs))
        x["is_te"] = (grp == "TE").astype(float)
        x["fit_group"] = pd.Series(grp).map(M.FIT_GROUP).to_numpy()
        p, sysc = M.score(fitted, x), np.full(len(x), np.nan)
        for g, m in fitted["models"].items():
            sel = (x["fit_group"] == g).to_numpy()
            if sel.any():
                Z = M._standardize(x.loc[sel, m["cols"]], m["ref"])
                k = [m["cols"].index(c) for c in SYS_COLS]
                sysc[sel] = Z[:, k] @ m["model"].coef_[0][k]
        rows.append(pd.DataFrame({"season": t, "prev_id": pid, "team": team, "f4": p, "f4_sys": sysc}))
        print(f"  F4 window {t}: {len(tr)} transfers x {len(fbs)} programs", flush=True)
    return pd.concat(rows, ignore_index=True)


def style_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Every transfer x every FBS program with a system in season t-1: minus the style distance (higher = closer)."""
    st = pd.read_csv(OUT / "team_styles.csv").set_index(["season", "team"])
    rows = []
    for t, tr in df[df["transfer_season"] >= 2019].drop_duplicates(["transfer_season", "prev_id"]).groupby("transfer_season"):
        prog = st.loc[t - 1].reindex(sorted(_conf(t)))[["D1", "D2"]].dropna()
        tr = tr[tr["from_D1"].notna()]
        d = np.hypot(prog["D1"].to_numpy()[None, :] - tr["from_D1"].to_numpy()[:, None],
                     prog["D2"].to_numpy()[None, :] - tr["from_D2"].to_numpy()[:, None])
        rows.append(pd.DataFrame({"season": t, "prev_id": np.repeat(tr["prev_id"].astype(str).to_numpy(), len(prog)),
                                  "team": np.tile(prog.index.to_numpy(), len(tr)), "style": -d.ravel()}))
    return pd.concat(rows, ignore_index=True)


def destination_rank(df: pd.DataFrame, sc: pd.DataFrame, col: str) -> pd.Series:
    """Per transfer row: the percentile rank (0 best .. 1 worst) of his actual destination by `col` among FBS programs
    at the destination's level that season. NaN when the destination isn't FBS or has no score."""
    lvl = {}
    for t in sc["season"].unique():
        c = _conf(int(t))
        lvl.update({(t, team): LB.tier(team, int(t), c) for team in c})
    s = sc[["season", "prev_id", "team", col]].dropna().copy()
    s["lvl"] = [lvl.get(k) for k in zip(s["season"], s["team"], strict=True)]
    s["r"] = s.groupby(["season", "prev_id", "lvl"])[col].rank(ascending=False, method="average")
    s["n"] = s.groupby(["season", "prev_id", "lvl"])[col].transform("size")
    s["pct"] = (s["r"] - 1) / (s["n"] - 1).clip(lower=1)
    key = s.set_index(["season", "prev_id", "team"])["pct"]
    k = list(zip(df["transfer_season"], df["prev_id"].astype(str), df["team_to"], strict=True))
    return pd.Series(key.reindex(k).to_numpy(), index=df.index)


# ---- expectations and outcome categories -----------------------------------------------------------------------------
def expectation(df: pd.DataFrame, out: str, lag: int) -> np.ndarray:
    """Walk-forward quality-only expectation per position: cohort c is predicted by a ridge fit on cohorts <= c - lag
    (outcomes complete by then). NaN where no earlier cohort exists."""
    pred = np.full(len(df), np.nan)
    for g in GROUPS:
        gi = np.flatnonzero((df["group"] == g).to_numpy())
        d = df.iloc[gi]
        Xa, y, coh = d[X.BASE].to_numpy(float), d[out].to_numpy(float), d["transfer_season"].to_numpy()
        for c in np.unique(coh):
            tr = (coh <= c - lag) & ~np.isnan(y)
            if tr.sum() < 30:
                continue
            mu, sd = Xa[tr].mean(0), Xa[tr].std(0)
            sd[sd == 0] = 1
            Z = (Xa - mu) / sd
            te = coh == c
            pred[gi[te]] = Ridge(alpha=1.0).fit(Z[tr], y[tr]).predict(Z[te])
    return pred


def outcomes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["A_exp"] = expectation(df, "A_vor", 1)
    df["B_exp"] = expectation(df, "B_vor", 3)
    for m in ("A", "B"):
        df[f"{m}_resid"] = df[f"{m}_vor"] - df[f"{m}_exp"]
    didnt_a = df["volume"].fillna(0) < df["group"].map(LB.BACKUP_MIN)
    didnt_b = df["B_vor"] == 0
    for m, didnt in (("A", didnt_a), ("B", didnt_b)):
        cat = np.where(didnt, "didnt_play", np.where(df[f"{m}_resid"] > 0, "beat", "fell_short"))
        df[f"{m}_cat"] = pd.Series(cat, index=df.index).where(df[f"{m}_resid"].notna() & df[f"{m}_vor"].notna())
    return df


# ---- statistics --------------------------------------------------------------------------------------------------------
def _third(p: pd.Series) -> pd.Series:
    return pd.Series(np.where(p <= 1 / 3, "good", np.where(p >= 2 / 3, "poor", "middle")), index=p.index).where(p.notna())


def _agree(d: pd.DataFrame, cat: str) -> float:
    e = d[d["third"].isin(["good", "poor"])]
    if e.empty:
        return np.nan
    good, beat = (e["third"] == "good").to_numpy(), (e[cat] == "beat").to_numpy()
    agree = (good & beat) | (~good & ~beat)
    null = good.mean() * beat.mean() + (1 - good.mean()) * (1 - beat.mean())
    return float(agree.mean() - null)


def _gap(d: pd.DataFrame, resid: str) -> float:
    return float(d.loc[d["third"] == "good", resid].mean() - d.loc[d["third"] == "poor", resid].mean())


def _row(aid: str, desc: str, d: pd.DataFrame, stat) -> dict:
    est, ci = X._boot(d.reset_index(drop=True), stat)
    return {"id": aid, "analysis": desc, "n": int(len(d)), "est": round(est, 4), "ci90": [round(ci[0], 4), round(ci[1], 4)],
            "excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}


def d_a(df: pd.DataFrame) -> list[dict]:
    log = []
    for meas in ("f4", "f4_sys", "style"):
        for g in ("all", *GROUPS):
            d = df[df[f"rank_{meas}"].notna() & ((df["group"] == g) if g != "all" else True)]
            d = d.drop_duplicates(["transfer_season", "prev_id"])
            log.append({**_row(f"DA.{meas}.{g}", "mean destination fit rank minus 0.5 (negative: coaches sign toward fit)",
                               d, lambda z, c=f"rank_{meas}": float(z[c].mean() - 0.5)),
                        "best_third_share": round(float((d[f"rank_{meas}"] <= 1 / 3).mean()), 4)})
    return log


def d_b(df: pd.DataFrame) -> tuple[list[dict], dict]:
    log, tables = [], {}
    for meas in ("f4", "f4_sys", "style"):
        for m in ("A", "B"):
            base = df[df[f"rank_{meas}"].notna() & df[f"{m}_cat"].notna()].copy()
            base["third"] = _third(base[f"rank_{meas}"])
            for g in ("all", *GROUPS):
                for coach in ("all", "kept", "new"):
                    keep = pd.Series(True, index=base.index)
                    if g != "all":
                        keep &= base["group"] == g
                    if coach != "all":
                        keep &= base["same_coach"] == (coach == "kept")
                    d = base[keep]
                    if d["third"].isin(["good", "poor"]).sum() < 40:
                        continue
                    aid = f"DB.{meas}.{m}.{g}.{coach}"
                    log.append(_row(f"{aid}.agree", "agreement rate minus its rate if fit and outcome were unrelated", d,
                                    lambda z, c=f"{m}_cat": _agree(z, c)))
                    log.append(_row(f"{aid}.gap", "mean outcome-minus-expectation, good-fit minus poor-fit moves", d,
                                    lambda z, r=f"{m}_resid": _gap(z, r)))
                    if coach == "all":
                        tables[aid] = pd.crosstab(d["third"], d[f"{m}_cat"]).reindex(
                            index=["good", "middle", "poor"], columns=["beat", "fell_short", "didnt_play"]).fillna(0).astype(int).to_dict("index")
    return log, tables


def label_years() -> pd.DataFrame:
    """Every transfer's seasons at the new school with role, F3 success, and value over replacement (f3_report.build)."""
    _, y = f3_report.build()
    return y


def d_c(df: pd.DataFrame, y: pd.DataFrame) -> dict:
    y = y[y["transfer_season"] <= 2024]                                     # two seasons observed (2025 the last)
    key = ["player_id", "group", "transfer_season"]
    y1 = y[y["year"] == 1].groupby(key)[["success", "won_role"]].any()
    y1["vor_pos"] = (y[y["year"] == 1].groupby(key)["vor"].max() > 0).reindex(y1.index).fillna(False)
    two = y[y["year"] <= 2].groupby(key).agg(success=("success", "any"), won_role=("won_role", "any"), vor=("vor", "max"))
    two["vor_pos"] = two["vor"] > 0
    rates = {}
    for g in ("all", *GROUPS):
        a = y1 if g == "all" else y1[y1.index.get_level_values("group") == g]
        b = two if g == "all" else two[two.index.get_level_values("group") == g]
        rates[g] = {"n": int(len(b)), **{f"year1_{c}": round(float(a[c].mean()), 4) for c in ("success", "won_role", "vor_pos")},
                    **{f"within2_{c}": round(float(b[c].mean()), 4) for c in ("success", "won_role", "vor_pos")}}
    # Open share at the destination (F0 availability as of May 1) vs. last season's quality, windows 2021-2024.
    L = Loader()
    av = pd.concat([M.availability(L, t, _decision(t)).assign(season=t) for t in range(2021, 2025)], ignore_index=True)
    av = av.set_index(["season", "team", "group"])["open_share"]
    d = df[df["transfer_season"].between(2021, 2024)].copy()
    d["open"] = av.reindex(list(zip(d["transfer_season"], d["team_to"], d["group"], strict=True))).to_numpy()
    s2 = two["success"].reindex(list(zip(d["player_id"], d["group"], d["transfer_season"], strict=True)))
    d["success2"] = s2.to_numpy()
    d = d[d["open"].notna() & d["success2"].notna()]
    auc = {}
    for target in ("success2", "won_role"):
        auc[target] = {}
        for g in ("all", *GROUPS):
            q = d if g == "all" else d[d["group"] == g]
            yy = q[target].astype(bool)
            if yy.nunique() < 2:
                continue
            auc[target][g] = {"n": int(len(q)), **{f: round(float(roc_auc_score(yy, q[f])), 4) for f in ("open", "p_epa", "p_share", "p_vor")}}
    return {"label_rates": rates, "auc_note": "AUC of each single input for the outcome among FBS-destination transfers 2021-2024 "
            "(open share as of May 1; won_role is year 1; 0.5 = no signal)", "auc": auc}


# ---- D-D: teammate context ---------------------------------------------------------------------------------------------
def context_table() -> pd.DataFrame:
    """Per player-season (2017-2025, his main team): opportunities and mean EPA per opportunity, raw / opponent-adjusted /
    teammate-relative / both, for QB dropbacks (raw and opponent only), RB carries, WR and TE targets."""
    L = Loader()
    rows = []
    for s in range(2017, 2026):
        ev, pos = D.season_events(L, s, pd.Timestamp(f"{s + 1}-02-01"))
        ev = ev[ev["epa"].notna()].copy()
        ev["group"] = ev["player_id"].map(pos).map(G.POSITION_GROUP)
        # Opponent: the defense's mean EPA allowed on this opportunity type, leaving the play's own value out.
        k = ["defense", "opp"]
        tot, n = ev.groupby(k)["epa"].transform("sum"), ev.groupby(k)["epa"].transform("size")
        ev["epa_o"] = ev["epa"] - (tot - ev["epa"]) / (n - 1).clip(lower=1)
        passer = ev[ev["opp"] == "dropback"].drop_duplicates("play_id").set_index("play_id")["player_id"]
        tg = ev[(ev["opp"] == "target") & ev["group"].isin(["WR", "TE", "RB"])].copy()
        tg["passer"] = tg["play_id"].map(passer)
        ca = ev[(ev["opp"] == "carry") & (ev["group"] == "RB")].copy()
        for frame_, by in ((tg, ["passer"]), (ca, ["offense"])):
            for c in ("epa", "epa_o"):
                g_ = frame_.groupby(by)
                pl = frame_.groupby([*by, "player_id"])[c]
                # Mean of the same passer's (team's) other opportunities: leave the player's own out.
                tot_by, n_by = g_[c].transform("sum"), g_[c].transform("size")
                tot_pl, n_pl = pl.transform("sum"), pl.transform("size")
                frame_[f"{c}_mate"] = (tot_by - tot_pl) / (n_by - n_pl).replace(0, np.nan)
            frame_["epa_t"] = frame_["epa"] - frame_["epa_mate"]
            frame_["epa_ot"] = frame_["epa_o"] - frame_["epa_o_mate"]
        db = ev[(ev["opp"] == "dropback") & (ev["group"] == "QB")]
        parts = [tg[tg["group"].isin(["WR", "TE"])], ca, db]
        for p in parts:
            team = p.groupby("player_id")["offense"].agg(lambda z: z.mode().iloc[0])
            agg = p.groupby("player_id").agg(n=("epa", "size"), raw=("epa", "mean"), opp=("epa_o", "mean"),
                                             **({"mate": ("epa_t", "mean"), "both": ("epa_ot", "mean")} if "epa_t" in p else {}))
            rows.append(agg.assign(season=s, team=team.reindex(agg.index), group=p.groupby("player_id")["group"].first()).reset_index())
        print(f"  context {s}", flush=True)
    return pd.concat(rows, ignore_index=True)


def d_d(ctx: pd.DataFrame, df: pd.DataFrame) -> list[dict]:
    log = []
    ctx = ctx[ctx["n"] >= ctx["group"].map(LB.BACKUP_MIN)]
    cur = ctx.set_index(["season", "player_id"])
    nxt = ctx.assign(season=ctx["season"] - 1).set_index(["season", "player_id"])
    stay = cur.join(nxt, suffixes=("", "_next"), how="inner")
    stay = stay[stay["team"] == stay["team_next"]].reset_index()
    tr = df[["transfer_season", "prev_id", "player_id", "group"]].drop_duplicates(["transfer_season", "prev_id"])
    a = cur.reindex(list(zip(tr["transfer_season"] - 1, tr["prev_id"].astype(str), strict=True))).reset_index(drop=True)
    b = cur.reindex(list(zip(tr["transfer_season"], tr["player_id"].astype(str), strict=True))).reset_index(drop=True)
    moved = a.join(b, suffixes=("", "_next")).assign(prev_id=tr["prev_id"].astype(str).to_numpy()).dropna(subset=["raw", "raw_next"])
    stay = stay.assign(prev_id=stay["player_id"].astype(str))
    for g in GROUPS:
        variants = ("raw", "opp") if g == "QB" else ("raw", "opp", "mate", "both")
        s, m = stay[stay["group"] == g], moved[moved["group"] == g]
        for v in variants:
            q = s.dropna(subset=[v, f"{v}_next"])
            log.append(_row(f"DD.{g}.stayers.{v}", f"year-to-year r of {v} EPA/opportunity, players who stayed", q,
                            lambda z, v=v: float(np.corrcoef(z[v], z[f"{v}_next"])[0, 1])))
            q = m.dropna(subset=[v, "opp_next"])
            if len(q) > 30:
                log.append(_row(f"DD.{g}.transfers.{v}", f"r of {v} at the old school with opponent-adjusted EPA/opportunity at the new one", q,
                                lambda z, v=v: float(np.corrcoef(z[v], z["opp_next"])[0, 1])))
        if g != "QB":
            for v in ("mate", "both"):
                q = m.dropna(subset=[v, "opp", "opp_next"])
                if len(q) > 30:
                    log.append(_row(f"DD.{g}.transfers.{v}_minus_opp", f"r({v}, new) minus r(opp, new) on the same transfers", q,
                                    lambda z, v=v: float(np.corrcoef(z[v], z["opp_next"])[0, 1] - np.corrcoef(z["opp"], z["opp_next"])[0, 1])))
    return log


def main() -> None:
    cached = "--cached" in sys.argv
    df = pd.read_parquet(CACHE / "fit_explore.parquet")
    df["prev_id"] = df["prev_id"].astype(str)
    f4p, ctxp = CACHE / "fit_audit_f4.parquet", CACHE / "fit_audit_context.parquet"
    f4 = pd.read_parquet(f4p) if cached and f4p.exists() else f4_scores(df)
    f4.to_parquet(f4p, index=False)
    sty = style_scores(df)
    df["rank_f4"], df["rank_f4_sys"] = destination_rank(df, f4, "f4"), destination_rank(df, f4, "f4_sys")
    df["rank_style"] = destination_rank(df, sty, "style")
    df = outcomes(df)
    ctx = pd.read_parquet(ctxp) if cached and ctxp.exists() else context_table()
    ctx.to_parquet(ctxp, index=False)
    log = d_a(df)
    lb, tables = d_b(df)
    log += lb
    log += d_d(ctx, df)
    yp = CACHE / "fit_audit_years.parquet"
    years = pd.read_parquet(yp) if cached and yp.exists() else label_years()
    years.to_parquet(yp, index=False)
    dc = d_c(df, years)
    n = len(log)
    res = {"status": "EXPLORATORY (FIT-AUDIT, PLAN-fit Phase 1b): every transfer here has been seen; the 2026 vault is not read",
           "transfers": int(len(df)),
           "coverage": {m: int(df[f"rank_{m}"].notna().sum()) for m in ("f4", "f4_sys", "style")},
           "analyses_run": n, "expected_by_chance_at_90pct": round(0.10 * n, 1), "excluding_zero": sum(a["excludes_zero"] for a in log),
           "D_C": dc, "D_B_tables": tables, "analyses": log}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("analyses", "D_B_tables")}, indent=1, default=str))
    for a in log:
        flag = "*" if a["excludes_zero"] else " "
        print(f"{flag} {a['id']:<34} n={a['n']:<5} est={a['est']:+.4f} ci90=[{a['ci90'][0]:+.4f}, {a['ci90'][1]:+.4f}]")


if __name__ == "__main__":
    main()
