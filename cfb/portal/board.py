"""The public portal board (SC-3a): every QB, RB, WR, and TE portal entrant for a window, with last season's production,
size, experience, a market tier, and similar players; and, per FBS program, its experimental fit top 25 by position
(as the model ranks them, and among entrants who reached last season's backup minimum: Matt, Nov 15, the floor is a
filter, the model's own list is the default and the one frozen in the ledger), role availability, offensive style,
and whether it kept its head coach.

Everything reads through the point-in-time loader as of the board's date (a live loader: the season just played may be
in progress). The fit model (cfb_system_fit) trains on transfer cohorts through 2024 only, so no 2026 transfer outcome
is used before the Phase 2 vault opens (PLAN-fit). 247Sports star ratings are used only inside the market tier and the
ledger's baseline comparison, never shown (Matt, Nov 15).

    build(t, as_of) -> {"players": DataFrame, "programs": list[dict], "fitted": ..., "board": DataFrame}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.asof import Loader
from cfb.fit import data as D
from cfb.fit import labels as LB
from cfb.fit import model as M
from cfb.fit import player_types as T
from cfb.profile import engine as G
from cfb.proj import pit

LAST_COHORT = 2024            # the vault boundary: two-year outcomes complete without 2026 (PLAN-fit, Phase 2)
DIMS_LAST = 2025              # dimensions fit on complete seasons only
TOP_FIT = 25
N_BEST = 5                    # best landing spots shown per entrant
# Best landing spots (Matt, Nov 17): ranked by relative fit, among programs with a real opening that take transfers.
# Ranking by projected yards (or by raw fit) gave nearly everyone the same schools: both carry program-wide terms
# (volume, level, "easy to succeed" systems) that are the same for every entrant.
INTAKE_WINDOWS = 3            # windows of portal intake counted for "takes transfers"
MIN_INTAKE = 3                # QB/RB/WR/TE transfers signed over those windows
PROJ_LAST_TRAIN = 2025        # the projection trains on outcome seasons through 2025: 2026 is the Phase 2 vault
PROJ_CATEGORY = {"QB": "QB", "RB": "RB", "WR": "WR_rec", "TE": "TE_rec"}
HORIZON_YEARS = 2             # the fit target: success within two seasons at the program (PLAN-fit)
N_SIMILAR, SIMILAR_SEASONS, N_STRENGTHS = 3, 3, 3
FEATS = {f["id"]: f for f in G.load_features()}


# ---- market tier (PLAN-scout S3: public proxies, never NIL dollars) ---------------------------------------------------
TIER_RULES = {
    1: "Power-conference starter with above-average production, or a former 4-5 star recruit producing above replacement",
    2: "Group of Five starter producing at the Power-conference starters' level, or a Power-conference rotation player",
    3: "Group of Five rotation player, or a former 4-5 star recruit with little production",
    4: "FCS player, or a thin sample",
}


def market_tier(r) -> int:
    starter, rotation = bool(r.won_role), r.volume >= LB.BACKUP_MIN[r.group]
    high_recruit = r.hs_stars >= 4 if pd.notna(r.hs_stars) else False
    if (r.level == "Power" and starter and r.above_average) or (high_recruit and r.vor > 0):
        return 1
    if (r.level == "G5" and starter and pd.notna(r.composite) and pd.notna(r.power_bar) and r.composite >= r.power_bar) or \
            (r.level == "Power" and rotation):
        return 2
    if (r.level == "G5" and rotation) or high_recruit:
        return 3
    return 4


# ---- pieces ----------------------------------------------------------------------------------------------------------
def strengths(prof: pd.DataFrame, n: int = N_STRENGTHS) -> dict[str, list[str]]:
    """player_id -> his top `n` performance components by percentile (style stats excluded), among those he qualified for."""
    q = prof[prof["feature"].map(lambda f: FEATS[f]["better"] != "style") & prof["pct"].notna()]
    q = q.sort_values(["player_id", "pct"], ascending=[True, False]).groupby("player_id").head(n)
    return q.groupby("player_id")["feature"].agg(lambda s: [FEATS[f]["label"] for f in s]).to_dict()


def similar(loader, t: int, as_of, ids: pd.Series, groups: pd.Series, rosters: dict) -> dict[str, list[dict]]:
    """For each entrant with a qualified season t-1: the N_SIMILAR nearest player-seasons at his position group over
    the last SIMILAR_SEASONS seasons (type features + size + efficiency composite, standardized), himself excluded."""
    seasons = range(t - SIMILAR_SEASONS, t)
    data = {s: M.season(loader, s, as_of) for s in seasons}
    prof = pd.concat([data[s]["prof"] for s in seasons], ignore_index=True)
    names = pd.concat([data[s]["roster"].assign(season=s, pid=data[s]["roster"]["id"].astype(str)) for s in seasons])
    names = names.drop_duplicates(["season", "pid"]).set_index(["season", "pid"])
    out = {}
    for g in ("QB", "RB", "WR", "TE"):
        m = T.matrix(prof, rosters, g)
        if m.empty:
            continue
        eff = pd.concat([data[s]["eff"].rename(s) for s in seasons], axis=1).stack().swaplevel()
        eff.index.names = ["season", "player_id"]
        m = m.join(eff.rename("eff")).fillna({"eff": 0.0})
        Z = T.standardize(m, m).to_numpy()
        idx = m.index
        mine = [p for p, gg in zip(ids, groups, strict=True) if gg == g and (t - 1, p) in idx]
        for p in mine:
            i = idx.get_loc((t - 1, p))
            d = np.linalg.norm(Z - Z[i], axis=1)
            d[[k for k, (s, q) in enumerate(idx) if q == p]] = np.inf
            near = []
            for k in np.argsort(d, kind="stable")[:N_SIMILAR]:
                s, q = idx[k]
                r = names.loc[(s, q)] if (s, q) in names.index else None
                near.append({"name": None if r is None else f"{r['firstName']} {r['lastName']}".strip(),
                             "team": None if r is None else r["team"], "season": int(s)})
            out[p] = near
    return out


def style_words(sys1: float, sys2: float, q1: tuple[float, float], q2: tuple[float, float]) -> dict[str, str]:
    """Plain-language style from system dimensions D1 (pass-heaviness) and D2 (spread to WRs vs. TE/RB), by thirds of
    the season's FBS programs."""
    rp = "run-heavy" if sys1 < q1[0] else "pass-heavy" if sys1 > q1[1] else "balanced"
    sp = "TE/RB-centered passing game" if sys2 < q2[0] else "wide-receiver-centered passing game" if sys2 > q2[1] else "balanced passing game"
    return {"run_pass": rp, "spread": sp}


def similarity_band(d: float, cuts: tuple[float, float]) -> str:
    return "similar" if d <= cuts[0] else "different" if d > cuts[1] else "somewhat similar"


# ---- the board -------------------------------------------------------------------------------------------------------
def relative_fit(b: pd.DataFrame) -> pd.Series:
    """His fit logit minus the program's mean logit for his position over all entrants: the part of the fit that is
    about him, not about how easy the program is for anyone."""
    q = b["p"].clip(1e-6, 1 - 1e-6)
    lg = np.log(q / (1 - q))
    return lg - lg.groupby([b["team"], b["group"]]).transform("mean")


def transfer_intake(L, t: int, as_of) -> pd.Series:
    """QB/RB/WR/TE portal transfers each program signed in the INTAKE_WINDOWS windows before season t."""
    n = []
    for s in range(t - INTAKE_WINDOWS, t):
        p = M.portal(L, s, as_of)
        n.append(p.loc[p["position"].isin(["QB", "RB", "WR", "TE"]) & p["destination"].notna(), "destination"])
    return pd.concat(n).value_counts() if n else pd.Series(dtype=int)


def best_spots(b: pd.DataFrame, avail: pd.Series, intake: pd.Series, proj_at: dict, n: int = N_BEST) -> dict[str, list]:
    """Per entrant: up to n programs by relative fit, among those with at least the median open share at his position
    and MIN_INTAKE transfers signed; each [program, his projected yards there or None]."""
    x = b[["player", "team", "group"]].copy()
    x["rel"] = relative_fit(b).to_numpy()
    x["open"] = [avail.get((tm, g), 0.0) for tm, g in zip(x["team"], x["group"], strict=True)]
    med = x.drop_duplicates(["team", "group"]).groupby("group")["open"].median()
    keep = (x["open"] >= x["group"].map(med)) & (x["team"].map(intake).fillna(0) >= MIN_INTAKE)
    top = x[keep].sort_values(["player", "rel", "team"], ascending=[True, False, True], kind="stable").groupby("player").head(n)
    return {p: [[tm, proj_at.get((tm, p))] for tm in d["team"]] for p, d in top.groupby("player")}


class UpcomingLoader(Loader):
    """For a window whose season CFBD hasn't published teams for yet (the coming window, before its season): reads
    teams_<t> as teams_<t-1>, so the board can show programs before the window opens. `membership_from` records it."""
    def __init__(self, t: int, **kw):
        super().__init__(**kw)
        self.window, self.membership_from = t, None

    def load(self, table, as_of, columns=None):
        if table == f"teams_{self.window}" and table not in self.tables:
            self.membership_from = self.window - 1
            table = f"teams_{self.window - 1}"
        return super().load(table, as_of, columns)


PLAYER_COLS = ["player_id", "group", "origin", "stars", "rating", "transferDate", "firstName", "lastName", "destination", "name",
               "team_from", "level", "share", "epa_adj", "epa_shrunk", "vor", "won_role", "above_average", "composite", "volume",
               "epa_pct", "height", "weight", "exp", "hs_stars", "power_bar", "tier", "strengths", "similar", "yards", "floor"]


def entrant_frame(L, t: int, as_of, e: pd.DataFrame) -> pd.DataFrame:
    """Every entrant with last season's production, size, experience, market tier, strengths, and similar players. A
    window shown before it opens has no entrants yet: an empty frame with the same columns."""
    if e.empty:
        return pd.DataFrame({c: pd.Series(dtype=bool if c == "floor" else object) for c in PLAYER_COLS})
    prev = M.season(L, t - 1, as_of)
    rosters = {s: M.season(L, s, as_of)["roster"] for s in range(M.FIRST_SEASON, t)}
    rec = M.recruits(L, t - 1, as_of)
    out = prev["outcomes"].sort_values("volume", ascending=False).drop_duplicates("player_id").set_index("player_id")
    ros = prev["roster"].assign(pid=prev["roster"]["id"].astype(str)).drop_duplicates("pid").set_index("pid")
    exp = D.experience(t - 1, rosters, rec)
    hs = (rec[rec["athleteId"].notna()].assign(pid=lambda d: d["athleteId"].astype(str)).sort_values(["year", "stars"], ascending=[True, False])
          .drop_duplicates("pid").set_index("pid")["stars"])
    epa_pct = prev["prof"][prev["prof"]["feature"].isin(set(LB.EPA.values()))].drop_duplicates("player_id").set_index("player_id")["pct"]
    power_bar = out[out["tier"] == "Power"].groupby("group")["bar"].first()
    conf_prev = prev["conf"]
    pl = e.copy()
    pid = pl["player_id"]
    pl["name"] = (pl["firstName"].fillna("") + " " + pl["lastName"].fillna("")).str.strip()
    pl["team_from"] = pid.map(pd.Series(prev["played"])).fillna(pid.map(ros["team"])).fillna(pl["origin"])
    pl["level"] = [LB.tier(x, t - 1, conf_prev) for x in pl["team_from"]]
    for c in ("share", "epa_adj", "epa_shrunk", "vor", "won_role", "above_average", "composite", "volume"):
        pl[c] = pid.map(out[c]).to_numpy()
    pl["volume"] = pl["volume"].fillna(0.0)
    pl["vor"] = pl["vor"].fillna(0.0)
    pl["won_role"] = pl["won_role"].fillna(False).astype(bool)
    pl["above_average"] = pl["above_average"].fillna(False).astype(bool)
    pl["epa_pct"] = pid.map(epa_pct).to_numpy()
    pl["height"] = pid.map(ros["height"]).where(lambda v: v.between(62, 84))
    pl["weight"] = pid.map(ros["weight"]).where(lambda v: v.between(140, 400))
    pl["exp"] = pid.map(exp).fillna(1).astype(int)
    pl["hs_stars"] = pid.map(hs)
    pl["power_bar"] = pl["group"].map(power_bar)
    pl["tier"] = [market_tier(r) for r in pl.itertuples()]
    st = strengths(prev["prof"])
    pl["strengths"] = [st.get(p, []) for p in pid]
    sim = similar(L, t, as_of, pid, pl["group"], rosters)
    pl["similar"] = [sim.get(p, []) for p in pid]
    pl["yards"] = pid.map(prev["yards"]).fillna(0.0)
    pl["floor"] = pl["volume"] >= pl["group"].map(LB.BACKUP_MIN)        # reached last season's backup minimum
    return pl


def build(t: int, as_of, loader: Loader | None = None) -> dict:
    as_of = pd.Timestamp(as_of)
    L = M.live(loader or UpcomingLoader(t))
    fitted = M.fit_fold(L, t, as_of, last_cohort=LAST_COHORT, dims_last=DIMS_LAST)
    b = M.board(fitted, L, t, as_of)
    e = M.entrants(L, t, as_of)
    pl = entrant_frame(L, t, as_of, e)
    # Programs.
    teams_t = L.load(f"teams_{t}", as_of, columns=["school", "conference"])
    conf_t = dict(zip(teams_t["school"], teams_t["conference"], strict=True))
    tf = M.team_frame(L, t, as_of, fitted["dims"], conf_t)
    fbs = tf.loc[tf.index.isin(conf_t)]
    q1, q2 = tuple(np.quantile(fbs["sys1"], [1 / 3, 2 / 3])), tuple(np.quantile(fbs["sys2"], [1 / 3, 2 / 3]))
    origin_sys = tf.reindex(pl["team_from"])[["sys1", "sys2"]].to_numpy().reshape(-1, 2)
    dist_all = np.hypot(fbs["sys1"].to_numpy()[:, None] - origin_sys[None, :, 0], fbs["sys2"].to_numpy()[:, None] - origin_sys[None, :, 1])
    cuts = tuple(np.nanquantile(dist_all, [1 / 3, 2 / 3])) if dist_all.size else (np.nan, np.nan)
    av = M.availability(L, t, as_of).set_index(["team", "group"])
    avail, from_portal = av["open_share"], av["from_portal"]
    coaches = L.load("coaches", as_of)
    head = coaches.sort_values("games").groupby(["season", "school"])["coach_id"].last()
    sysmap = dict(zip(pl["player_id"], map(tuple, origin_sys), strict=True))
    floor_ids = set(pl.loc[pl["floor"], "player_id"])
    # Projected season-t yards if he signed at each program (cfb_player_projection, counterfactual against that
    # program's projected room as of the board's date).
    # Only entrants who reached last season's backup minimum (Matt, Nov 16): below it the projection rests on too little,
    # and low-usage veterans moving down a level were projected well above what they got (cfb/proj/lowuse.py).
    proven = pl[pl["floor"].astype(bool)]
    if proven.empty:
        wf = pd.DataFrame(columns=["team", "key", "pos", "proj", "lo80", "hi80", "share", "category"])
    else:
        wf = pit.whatif(L, t, as_of, proven[["player_id", "group", "team_from"]], sorted(conf_t), last_train=PROJ_LAST_TRAIN)
        wf = wf[wf["category"] == wf["pos"].map(PROJ_CATEGORY)].sort_values(["team", "pos", "proj"], ascending=[True, True, False],
                                                                             kind="stable")
    proj_lists = {(tm, g): [[k, round(float(p)), round(float(lo)), round(float(hi)), round(float(sh), 3)]
                            for k, p, lo, hi, sh in zip(d["key"], d["proj"], d["lo80"], d["hi80"], d["share"], strict=True)][:TOP_FIT]
                  for (tm, g), d in wf.groupby(["team", "pos"])}
    # His projection at every program (shown beside the fit list and his best landing spots), and his best landing
    # spots: by relative fit among programs with an opening that take transfers (best_spots).
    proj_at = {(tm, k): round(float(p)) for tm, k, p in zip(wf["team"], wf["key"], wf["proj"], strict=True)}
    best = best_spots(b, avail, transfer_intake(L, t, as_of), proj_at)
    pl["best"] = [best.get(p, []) for p in pl["player_id"]]
    programs = []
    for team in sorted(conf_t):
        row = tf.loc[team] if team in tf.index else None
        new, old = head.get((t, team)), head.get((t - 1, team))
        prog = {"team": team, "conf": conf_t[team],
                "style": None if row is None else style_words(row["sys1"], row["sys2"], q1, q2),
                "kept_coach": None if new is None or old is None else bool(new == old),
                "open": {g: (None if (team, g) not in avail.index else round(float(avail[(team, g)]), 3)) for g in ("QB", "RB", "WR", "TE")},
                "open_portal": {g: (None if (team, g) not in from_portal.index else round(float(from_portal[(team, g)]), 3))
                                for g in ("QB", "RB", "WR", "TE")},
                "fit": {}, "fit_floor": {}, "proj": {g: proj_lists.get((team, g), []) for g in ("QB", "RB", "WR", "TE")}}
        bt = b[b["team"] == team]
        for g in ("QB", "RB", "WR", "TE"):
            ranked = bt[bt["group"] == g].sort_values("p", ascending=False, kind="stable")
            for name, cand in (("fit", ranked), ("fit_floor", ranked[ranked["player"].isin(floor_ids)])):
                lst = []
                for p in cand["player"].head(TOP_FIT):
                    o = sysmap.get(p, (np.nan, np.nan))
                    d = np.hypot(row["sys1"] - o[0], row["sys2"] - o[1]) if row is not None else np.nan
                    lst.append([p, None if np.isnan(d) else similarity_band(d, cuts), proj_at.get((team, p))])
                prog[name][g] = lst
        programs.append(prog)
    return {"players": pl, "programs": programs, "fitted": fitted, "board": b, "as_of": as_of, "window": t,
            "membership_from": getattr(L, "membership_from", None)}
