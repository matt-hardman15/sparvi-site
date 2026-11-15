"""FIT-OC (PLAN-fit Phase 1b, exploratory): do offensive coordinators carry the system, and do the Phase 1 fit leads
sharpen when the destination kept its coordinator? Coordinators from Wikipedia team-season infoboxes
(ingest/cfb/coordinators.py; CC BY-SA 4.0, not point-in-time). Writes cfb/fit/output/oc.json (aggregates only).

  validation    Wikipedia's head coach vs. CFBD's (last-name match), coverage per season
  persistence   F1 system change from t-1 to t (D1 pass-heaviness, D2 target spread) by continuity: same head coach and
                same coordinator / same head coach, new coordinator / new head coach
  carryover     coordinators who moved schools: is the new school's system closer to his old school's or to its own?
  leads         Phase 1's style-distance leads (explore.py: corr(style distance, residual of the quality-only model))
                split by whether the destination kept its coordinator (all, and within kept head coach)

    python cfb/fit/oc_report.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cfb.fit import explore as X
from sparvi.config import RAW, ROOT

OUT = ROOT / "cfb" / "fit" / "output"
SEED, N_BOOT = 7, 1000


def _names(v) -> set[str]:
    return {n.strip().lower() for n in str(v).split(";") if n.strip()} if isinstance(v, str) else set()


def continuity() -> pd.DataFrame:
    """Per FBS school-season t with coordinators known in t-1 and t: kept_oc (any coordinator shared), kept_hc (CFBD)."""
    oc = pd.read_parquet(RAW / "cfb" / "coordinators.parquet")
    oc = oc[oc["oc"].notna()].set_index(["season", "school"])["oc_all"].map(_names)
    head = X.head_coaches().set_index(["season", "school"])["coach_id"]
    rows = []
    for (s, sc), names in oc.items():
        prev = oc.get((s - 1, sc))
        if prev is None:
            continue
        h, hp = head.get((s, sc)), head.get((s - 1, sc))
        rows.append({"season": s, "school": sc, "kept_oc": bool(names & prev),
                     "kept_hc": None if h is None or hp is None else bool(h == hp)})
    return pd.DataFrame(rows)


def validation() -> dict:
    oc = pd.read_parquet(RAW / "cfb" / "coordinators.parquet")
    co = pd.read_parquet(RAW / "cfb" / "coaches.parquet").sort_values("games").groupby(["season", "school"]).tail(1)
    last = co.set_index(["season", "school"])["coach"].str.split().str[-1].str.lower()
    hc = oc.dropna(subset=["head_coach_wiki"])
    match = None
    if last is not None:
        k = list(zip(hc["season"], hc["school"], strict=True))
        lk = last.reindex(k).to_numpy()
        ok = [isinstance(b, str) and b in a.lower() for a, b in zip(hc["head_coach_wiki"], lk, strict=True) if isinstance(b, str)]
        match = round(float(np.mean(ok)), 4)
    return {"team_seasons": int(len(oc)), "with_page": int(oc["page"].notna().sum()), "with_oc": int(oc["oc"].notna().sum()),
            "by_season_with_oc": oc.groupby("season")["oc"].apply(lambda s: int(s.notna().sum())).to_dict(),
            "head_coach_matches_cfbd": match}


def _boot_teams(d: pd.DataFrame, stat) -> tuple[float, list[float]]:
    rng = np.random.default_rng(SEED)
    keys = d["school"].to_numpy()
    uniq = np.unique(keys)
    pos = {k: np.flatnonzero(keys == k) for k in uniq}
    draws = [stat(d.iloc[np.concatenate([pos[k] for k in rng.choice(uniq, len(uniq))])]) for _ in range(N_BOOT)]
    return float(stat(d)), [float(np.nanpercentile(draws, 5)), float(np.nanpercentile(draws, 95))]


def persistence(cont: pd.DataFrame) -> dict:
    st = pd.read_csv(OUT / "team_styles.csv").set_index(["season", "team"])
    d = cont.copy()
    for dim in ("D1", "D2"):
        d[f"{dim}_prev"] = st[dim].reindex(list(zip(d["season"] - 1, d["school"], strict=True))).to_numpy()
        d[f"{dim}_now"] = st[dim].reindex(list(zip(d["season"], d["school"], strict=True))).to_numpy()
    d = d.dropna(subset=["D1_prev", "D1_now", "kept_hc"])
    d["dist"] = np.hypot(d["D1_now"] - d["D1_prev"], d["D2_now"] - d["D2_prev"])
    groups = {"same_hc_same_oc": d["kept_hc"] & d["kept_oc"], "same_hc_new_oc": d["kept_hc"] & ~d["kept_oc"],
              "new_hc": ~d["kept_hc"].astype(bool)}
    out = {}
    for name, m in groups.items():
        q = d[m.astype(bool)]
        out[name] = {"n": int(len(q)),
                     **{f"r_{dim}": _r(_boot_teams(q, lambda z, dim=dim: float(np.corrcoef(z[f"{dim}_prev"], z[f"{dim}_now"])[0, 1])))
                        for dim in ("D1", "D2")},
                     "mean_distance": _r(_boot_teams(q, lambda z: float(z["dist"].mean())))}
    q = d[d["kept_hc"].astype(bool)].assign(new_oc=lambda z: (~z["kept_oc"]).astype(float))
    out["same_hc_distance_gap_new_minus_same_oc"] = _r(_boot_teams(
        q, lambda z: float(z.loc[z["new_oc"] == 1, "dist"].mean() - z.loc[z["new_oc"] == 0, "dist"].mean())))
    return out


def carryover() -> dict:
    """Coordinators at school X in t-1 and a different school Y in t (as coordinator)."""
    oc = pd.read_parquet(RAW / "cfb" / "coordinators.parquet").dropna(subset=["oc"])
    st = pd.read_csv(OUT / "team_styles.csv").set_index(["season", "team"])
    where = {}
    for r in oc.itertuples():
        for n in _names(r.oc_all):
            where.setdefault((r.season, n), set()).add(r.school)
    rows = []
    for (s, n), schools in where.items():
        for y in schools:
            for x in where.get((s - 1, n), set()) - {y}:
                k_new, k_his, k_prog = (s, y), (s - 1, x), (s - 1, y)
                if all(k in st.index for k in (k_new, k_his, k_prog)):
                    rows.append({"school": y, **{f"{dim}_{w}": st.loc[k, dim] for dim in ("D1", "D2")
                                                 for w, k in (("new", k_new), ("his", k_his), ("prog", k_prog))}})
    d = pd.DataFrame(rows)
    out = {"coordinators_who_moved": int(len(d))}
    for dim in ("D1", "D2"):
        out[dim] = {"r_with_his_old_team": _r(_boot_teams(d, lambda z, dim=dim: float(np.corrcoef(z[f"{dim}_new"], z[f"{dim}_his"])[0, 1]))),
                    "r_with_program_last_year": _r(_boot_teams(d, lambda z, dim=dim: float(np.corrcoef(z[f"{dim}_new"], z[f"{dim}_prog"])[0, 1])))}
    return out


def leads(cont: pd.DataFrame) -> list[dict]:
    df = pd.read_parquet(RAW / "cfb" / "fit_explore.parquet")
    k = cont.set_index(["season", "school"])["kept_oc"]
    df["kept_oc"] = k.reindex(list(zip(df["transfer_season"], df["team_to"], strict=True))).to_numpy()
    log = []
    for g in ("QB", "RB", "WR", "TE"):
        for out in ("A_vor", "A_rate", "B_vor"):
            d = df[(df["group"] == g) & df[out].notna()].copy()
            d["resid"] = d[out] - X._loco(d, X.BASE, out)
            d = d[d["M1_distance"].notna() & d["kept_oc"].notna()]
            for name, m in (("kept_oc", d["kept_oc"] == True), ("new_oc", d["kept_oc"] == False),  # noqa: E712
                            ("kept_hc_kept_oc", d["same_coach"] & (d["kept_oc"] == True)),       # noqa: E712
                            ("kept_hc_new_oc", d["same_coach"] & (d["kept_oc"] == False))):      # noqa: E712
                q = d[m.astype(bool)]
                if len(q) > 30:
                    est, ci = X._boot(q, X._corr("M1_distance", "resid"))
                    log.append({"id": f"OC.{g}.{out}.{name}", "n": int(len(q)), "est": round(est, 4),
                                "ci90": [round(ci[0], 4), round(ci[1], 4)], "excludes_zero": bool(ci[0] > 0 or ci[1] < 0)})
    return log


def _r(est_ci) -> dict:
    est, ci = est_ci
    return {"est": round(est, 4), "ci90": [round(ci[0], 4), round(ci[1], 4)]}


def main() -> None:
    cont = continuity()
    lg = leads(cont)
    res = {"status": "EXPLORATORY (FIT-OC): coordinator names from Wikipedia (CC BY-SA 4.0), current revisions, not point-in-time",
           "validation": validation(), "school_seasons_with_continuity": int(len(cont)),
           "kept_oc_share": round(float(cont["kept_oc"].mean()), 4), "persistence": persistence(cont), "carryover": carryover(),
           "leads_note": "corr(style distance, residual): negative = closer systems do better than the quality-only model expects",
           "analyses_run": len(lg), "excluding_zero": sum(a["excludes_zero"] for a in lg), "leads": lg}
    (OUT / "oc.json").write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in res.items() if k != "leads"}, indent=1, default=str))
    for a in lg:
        print(f"{'*' if a['excludes_zero'] else ' '} {a['id']:<30} n={a['n']:<5} est={a['est']:+.4f} ci90=[{a['ci90'][0]:+.4f}, {a['ci90'][1]:+.4f}]")


if __name__ == "__main__":
    main()
