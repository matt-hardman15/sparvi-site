"""The system-fit model (SYSTEM-FIT F4, card models/cfb/system_fit): for a program and a portal entrant, the chance he
succeeds there (wins the role and beats the starters' average within two years), from his previous season, his type,
the program's system, and how they combine. Every read goes through the point-in-time loader.

Season data (season()) is computed once per loader and season, as of when that season is complete (March 1 after it,
past the bowls and the playoff), and only handed to a caller whose cutoff is on or after that date.

    rows(...)      feature rows for (player, program) pairs as of a cohort (the season he would play at the program)
    fit_fold(...)  one logistic regression per fit group on the training cohorts, penalty by leave-one-cohort-out
    board(...)     every portal entrant x every FBS program, scored, with the baselines, and each one's board credit
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cfb.fit import data as D
from cfb.fit import labels as LB
from cfb.fit import player_types as T
from cfb.fit import systems as S
from cfb.fit.portal import link
from cfb.profile import engine as G

FIRST_SEASON = 2017                      # profiles, types, and systems are fit from here (cohort 2018's previous season)
FIRST_COHORT = 2018
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)
N_TYPE, N_SYS = 2, 2                     # the stable dimensions in the F1/F2 reports
LEVEL = {"FCS": 0, "G5": 1, "Power": 2}
FIT_GROUP = {"QB": "QB", "RB": "RB", "WR": "REC", "TE": "REC"}
TOP = 10
BASE = ["eff", "no_prior", "share", "exp", "hs", "no_hs", "lvl_from", "lvl_to", "dlvl", "off_q",
        *[f"ty{i + 1}" for i in range(N_TYPE)], *[f"sys{j + 1}" for j in range(N_SYS)]]
INTER = [f"ty{i + 1}_x_sys{j + 1}" for i in range(N_TYPE) for j in range(N_SYS)] + ["dlvl_x_eff"]


def done(s: int) -> pd.Timestamp:
    return pd.Timestamp(f"{s + 1}-03-01")


def _mode(x: pd.Series):
    return x.mode().iloc[0]


def offense_quality(ev: pd.DataFrame) -> dict[str, float]:
    """Opponent-adjusted offensive EPA per play: the profile engine's team-level ridge with offense and defense swapped,
    on dropbacks and carries (one row per play)."""
    r = ev[ev["opp"].isin(["dropback", "carry"])].drop_duplicates("play_id")
    r = r[r["epa"].notna()]
    swapped = pd.DataFrame({"offense": r["defense"].to_numpy(), "defense": r["offense"].to_numpy()})
    return G.opponent_effects(swapped, r["epa"].to_numpy(float))


def production(ev: pd.DataFrame, pos: pd.Series) -> pd.Series:
    """player_id -> yards at his position: QB on dropbacks (passing, net of sacks), RB rushing, WR/TE receiving."""
    grp = ev["player_id"].map(pos).map(G.POSITION_GROUP)
    opp = grp.map({"QB": "dropback", "RB": "carry", "WR": "target", "TE": "target"})
    r = ev[ev["opp"] == opp]
    return r.groupby("player_id")["yards"].sum()


def efficiency(prof: pd.DataFrame) -> pd.Series:
    """player_id -> mean of within-position z-scores of his shrunk efficiency features (labels.EFFICIENCY)."""
    out = []
    for g, feats in LB.EFFICIENCY.items():
        q = prof[(prof["group"] == g) & prof["feature"].isin(feats)]
        if q.empty:
            continue
        w = q.pivot_table(index="player_id", columns="feature", values="shrunk")
        out.append(((w - w.mean()) / w.std()).mean(axis=1))
    s = pd.concat(out)
    return s[~s.index.duplicated()]


def season(loader, s: int, cutoff) -> dict:
    """Everything the model needs from one completed season, cached on the loader. A live loader (live(loader): the
    portal board, never a backtest) may also read a season still in progress, as of the cutoff; those inputs are what
    was public that day."""
    partial = done(s) > pd.Timestamp(cutoff)
    if partial and not getattr(loader, "_fit_live", False):
        raise ValueError(f"season {s} isn't complete until {done(s).date()}; cutoff {pd.Timestamp(cutoff).date()}")
    cache = loader.__dict__.setdefault("_fit_seasons", {})
    key = (s, pd.Timestamp(cutoff).normalize()) if partial else s
    if key not in cache:
        as_of = pd.Timestamp(cutoff) if partial else done(s)
        ev, pos = D.season_events(loader, s, as_of)
        prof = G.season_profile(ev, pos, s)
        teams = loader.load(f"teams_{s}", as_of, columns=["school", "conference"])
        conf = dict(zip(teams["school"], teams["conference"], strict=True))
        roster = loader.load(f"players_roster_{s}", as_of)
        cache[key] = {
            "prof": prof, "roster": roster, "conf": conf, "pos": pos,
            "style": S.style(loader.load(f"plays_{s}", as_of), ev, pos),
            "played": ev.groupby("player_id")["offense"].agg(_mode).to_dict(),
            "outcomes": LB.role_and_output(prof, s, conf, LB.team_plays(ev)),
            "offense": offense_quality(ev), "yards": production(ev, pos), "eff": efficiency(prof),
            "workload": D.workload(ev, pos), "played_pairs": set(zip(ev["offense"], ev["player_id"], strict=True))}
    return cache[key]


def recruits(loader, last_class: int, cutoff) -> pd.DataFrame:
    from ingest.cfb.players import seasons
    years = [y for y in seasons()["recruits"] if y <= last_class]
    r = pd.concat([loader.load(f"players_recruits_{y}", min(pd.Timestamp(cutoff), done(y))) for y in years], ignore_index=True)
    return r


def portal(loader, t: int, as_of) -> pd.DataFrame:
    from ingest.cfb.players import seasons
    if t not in seasons()["portal"]:
        return pd.DataFrame(columns=["season", "firstName", "lastName", "position", "origin", "destination", "transferDate",
                                     "rating", "stars"])
    return loader.load(f"players_portal_{t}", as_of)


# ---- dimensions, refit per fold ---------------------------------------------------------------------------------------
def dims(loader, last: int, cutoff) -> dict:
    ss = range(FIRST_SEASON, last + 1)
    data = {s: season(loader, s, cutoff) for s in ss}
    prof = pd.concat([data[s]["prof"] for s in ss], ignore_index=True)
    rosters = {s: data[s]["roster"] for s in ss}
    styles = pd.concat([data[s]["style"].assign(season=s).reset_index().rename(columns={"offense": "team"}) for s in ss])
    out = {"sys": S.dimensions(styles.set_index(["season", "team"]).sort_index())}
    for g in T.FIT_GROUPS:
        out[g] = T.dimensions(T.group_matrix(prof, rosters, g))
    return out


def _style_rows(prof: pd.DataFrame, roster: pd.DataFrame, group: str, ref: pd.DataFrame) -> pd.DataFrame:
    """Every player of a position group (not only qualified ones) as a type-matrix row; gaps filled with the fit's
    medians, size cleaned as in player_types.matrix."""
    p = prof[prof["group"] == group]
    wide = p.pivot_table(index="player_id", columns="feature", values="shrunk").reindex(columns=T.STYLE[group])
    size = roster.assign(player_id=roster["id"].astype(str)).drop_duplicates("player_id").set_index("player_id")[T.SIZE]
    out = wide.join(size, how="left")
    out["height"] = out["height"].where(out["height"].between(62, 84))
    out["weight"] = out["weight"].where(out["weight"].between(140, 400))
    if "is_te" in ref.columns:
        out["is_te"] = float(group == "TE")
    out = out.reindex(columns=ref.columns)
    return out.fillna(ref.median())


def type_coords(prev: dict, dm: dict) -> pd.DataFrame:
    rows = []
    for g in ("QB", "RB", "WR", "TE"):
        d = dm[FIT_GROUP[g]]
        m = _style_rows(prev["prof"], prev["roster"], g, d["ref"])
        if len(m):
            rows.append(T.project(d, m).iloc[:, :N_TYPE].set_axis([f"ty{i + 1}" for i in range(N_TYPE)], axis=1))
    out = pd.concat(rows)
    return out[~out.index.duplicated()]


# ---- feature rows ---------------------------------------------------------------------------------------------------
def player_frame(loader, c: int, cutoff, dm: dict) -> pd.DataFrame:
    """Per player id in season c-1: the player features for a move into season c."""
    prev = season(loader, c - 1, cutoff)
    rosters = {s: season(loader, s, cutoff)["roster"] for s in range(FIRST_SEASON, c)}
    rec = recruits(loader, c - 1, cutoff)
    ids = prev["roster"]["id"].astype(str).drop_duplicates()
    f = pd.DataFrame(index=pd.Index(ids, name="player_id"))
    f["eff"] = prev["eff"].reindex(f.index)
    f["no_prior"] = f["eff"].isna().astype(float)
    f["eff"] = f["eff"].fillna(0.0)
    share = prev["outcomes"].drop_duplicates("player_id").set_index("player_id")["share"]
    f["share"] = share.reindex(f.index).fillna(0.0)
    f["exp"] = D.experience(c - 1, rosters, rec).reindex(f.index).fillna(1.0) + 1
    hs = (rec[rec["athleteId"].notna()].assign(pid=lambda d: d["athleteId"].astype(str))
          .sort_values(["year", "rating"], ascending=[True, False]).drop_duplicates("pid").set_index("pid")["rating"])
    f["hs"] = hs.reindex(f.index)
    f["no_hs"] = f["hs"].isna().astype(float)
    f["hs"] = f["hs"].fillna(hs.min() if len(hs) else 0.0)
    team = pd.Series(prev["played"]).reindex(f.index)
    rteam = prev["roster"].assign(pid=prev["roster"]["id"].astype(str)).drop_duplicates("pid").set_index("pid")["team"]
    team = team.fillna(rteam.reindex(f.index))
    f["team_from"] = team
    f["lvl_from"] = [LEVEL[LB.tier(x, c - 1, prev["conf"])] if isinstance(x, str) else 0 for x in team]
    tc = type_coords(prev, dm)
    for col in tc.columns:
        f[col] = tc[col].reindex(f.index).fillna(0.0)
    f["yards"] = prev["yards"].reindex(f.index).fillna(0.0)
    f["eff_rank"] = prev["eff"].reindex(f.index)          # NaN = no previous production (ranks last as a baseline)
    return f


def team_frame(loader, c: int, cutoff, dm: dict, conf_c: dict[str, str]) -> pd.DataFrame:
    """Per program: its system and offense in season c-1, and its level in season c."""
    prev = season(loader, c - 1, cutoff)
    sysc = S.project(dm["sys"], prev["style"]).iloc[:, :N_SYS].set_axis([f"sys{j + 1}" for j in range(N_SYS)], axis=1)
    teams = sorted(set(sysc.index) | set(conf_c) | set(prev["offense"]))
    f = pd.DataFrame(index=pd.Index(teams, name="team"))
    for col in sysc.columns:
        f[col] = sysc[col].reindex(f.index).fillna(0.0)
    f["off_q"] = pd.Series(prev["offense"]).reindex(f.index).fillna(0.0)
    f["lvl_to"] = [LEVEL[LB.tier(x, c, conf_c)] for x in f.index]
    return f


def pairs(players: pd.DataFrame, teams: pd.DataFrame, pid: pd.Series, team: pd.Series) -> pd.DataFrame:
    x = players.reindex(pid.to_numpy()).reset_index(drop=True).join(teams.reindex(team.to_numpy()).reset_index(drop=True))
    x["off_q"] = x["off_q"].fillna(0.0)
    x["lvl_to"] = x["lvl_to"].fillna(LEVEL["FCS"])          # a program with no FBS record is FCS
    x["yards"] = x["yards"].fillna(0.0)
    for c in [f"sys{j + 1}" for j in range(N_SYS)]:
        x[c] = x[c].fillna(0.0)
    x["dlvl"] = x["lvl_to"] - x["lvl_from"]
    for i in range(N_TYPE):
        for j in range(N_SYS):
            x[f"ty{i + 1}_x_sys{j + 1}"] = x[f"ty{i + 1}"] * x[f"sys{j + 1}"]
    x["dlvl_x_eff"] = x["dlvl"] * x["eff"]
    return x


def columns(fit_group: str) -> list[str]:
    return [*BASE, *INTER, *(["is_te"] if fit_group == "REC" else [])]


# ---- training ---------------------------------------------------------------------------------------------------------
def cohort_labels(loader, c: int, last: int, cutoff) -> pd.DataFrame:
    """Transfers into season c with success within two years (years beyond `last` unobserved)."""
    data = {s: season(loader, s, cutoff) for s in range(c - 1, min(c + 1, last) + 1)}
    rosters = {s: d["roster"] for s, d in data.items()}
    played = {s: d["played"] for s, d in data.items()}
    tr = LB.transfers(rosters, c, portal(loader, c, done(c) if done(c) <= pd.Timestamp(cutoff) else cutoff), played)
    lab = LB.label(tr, data[c]["outcomes"])
    y = LB.follow(lab, rosters, {s: d["outcomes"] for s, d in data.items()}, min(c + 1, last), played)
    y = y[y["year"] <= 2]
    key = ["player_id", "group", "transfer_season"]
    first = y[y["year"] == 1].set_index(key)
    first["success2"] = y.groupby(key)["success"].any()
    first["efficient2"] = (y["above_average"] & (y["share"] >= 0.10)).groupby([y[k] for k in key]).any()
    first["won_role2"] = y.groupby(key)["won_role"].any()
    return first.reset_index()


def cohort_outcomes(loader, t: int, last: int) -> pd.DataFrame:
    """Every transfer of cohort t with a previous-season id and its labels through season `last` (t+1 for the two-year
    label), read once season `last` is complete; fbs = the destination is an FBS program in season t."""
    lab = cohort_labels(loader, t, last, done(last))
    lab = lab[lab["prev_id"].notna()].copy()
    conf_t = season(loader, t, done(last))["conf"]
    lab["fbs"] = lab["team_to"].isin(set(conf_t))
    lab["dest_tier"] = [LB.tier(x, t, conf_t) for x in lab["team_to"]]
    lab["entity_id"] = str(t) + "|" + lab["prev_id"].astype(str) + "|" + lab["team_to"]
    return lab


def availability(loader, t: int, as_of) -> pd.DataFrame:
    """Role availability per program and position group as of a decision date in season t (F0, data.role_availability):
    last season's workload share leaving through the portal (entries filed by as_of) or expected to leave otherwise,
    with leave rates by experience from the two seasons before."""
    def links(s):
        p = portal(loader, s, min(pd.Timestamp(as_of), done(s))).assign(season=s)
        if p.empty:                                   # no portal records before 2021
            return p.assign(player_id=pd.Series(dtype=object))
        return link(p, {y: season(loader, y, as_of)["roster"] for y in (s - 2, s - 1)})
    def exp(s):
        return D.experience(s, {y: season(loader, y, as_of)["roster"] for y in range(FIRST_SEASON, s + 1)}, recruits(loader, s, as_of))
    hist = []
    for h in (t - 2, t - 1):
        lk = links(h)
        hist.append((season(loader, h - 1, as_of)["workload"], set(lk["player_id"].dropna()), exp(h - 1), season(loader, h, as_of)["played_pairs"]))
    p_leave = D.leave_rates(hist)
    w = season(loader, t - 1, as_of)["workload"].copy()
    lk = links(t)
    filed = lk[pd.to_datetime(lk["transferDate"], utc=True, errors="coerce").dt.tz_localize(None) <= pd.Timestamp(as_of)]
    w["portal"] = w["player_id"].isin(set(filed["player_id"].dropna()))
    e = exp(t - 1)
    w["p_leave"] = np.where(w["portal"], 1.0, w["player_id"].map(e).fillna(1).clip(1, 6).astype(int).map(p_leave).fillna(0.2))
    w["leaving"] = w["share"] * w["p_leave"]
    return w.groupby(["team", "group"])["leaving"].sum().clip(0, 1).rename("open_share").reset_index()


def live(loader):
    """Mark a loader as the live board's: seasons in progress may be read as of the cutoff (season())."""
    loader._fit_live = True
    return loader


def training_rows(loader, fold_t: int, cutoff, dm: dict, last_cohort: int | None = None) -> pd.DataFrame:
    rows = []
    last = fold_t - 2 if last_cohort is None else min(fold_t - 2, last_cohort)
    for c in range(FIRST_COHORT, last + 1):                       # cohorts 2018 .. t-2 (or last_cohort)
        lab = cohort_labels(loader, c, fold_t - 1, cutoff)
        lab = lab[lab["prev_id"].notna()]
        conf_c = season(loader, c, cutoff)["conf"]
        x = pairs(player_frame(loader, c, cutoff, dm), team_frame(loader, c, cutoff, dm, conf_c), lab["prev_id"].astype(str), lab["team_to"])
        x["is_te"] = (lab["group"].to_numpy() == "TE").astype(float)
        x["fit_group"] = lab["group"].map(FIT_GROUP).to_numpy()
        x["y"] = lab["success2"].astype(float).to_numpy()
        x["cohort"] = c
        rows.append(x)
    return pd.concat(rows, ignore_index=True)


def _standardize(X: pd.DataFrame, ref: pd.DataFrame) -> np.ndarray:
    sd = ref.std().replace(0, 1.0).fillna(1.0)
    return ((X - ref.mean()) / sd).fillna(0.0).to_numpy(float)


def _fit_logistic(C: float, X, y) -> LogisticRegression:
    """A fit that doesn't converge raises (DATA-QA: warnings were being filtered, so non-convergence would pass silently)."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return LogisticRegression(C=C, max_iter=5000).fit(X, y)


def fit_group(d: pd.DataFrame, cols: list[str]) -> dict:
    """L2 logistic regression; C by leave-one-cohort-out log loss (ties: the smaller C)."""
    Xdf, y, coh = d[cols], d["y"].to_numpy(float), d["cohort"].to_numpy()
    loss = {}
    for C in C_GRID:
        tot = 0.0
        for k in np.unique(coh):
            tr, te = coh != k, coh == k
            if len(np.unique(y[tr])) < 2:
                tot = np.inf
                break
            m = _fit_logistic(C, _standardize(Xdf[tr], Xdf[tr]), y[tr])
            p = np.clip(m.predict_proba(_standardize(Xdf[te], Xdf[tr]))[:, 1], 1e-9, 1 - 1e-9)
            tot += float(-(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)).sum())
        loss[C] = tot
    best = min(C_GRID, key=lambda C: (loss[C], C))
    m = _fit_logistic(best, _standardize(Xdf, Xdf), y)
    return {"model": m, "ref": Xdf, "cols": cols, "C": best, "cv_logloss": loss, "n": int(len(d)), "rate": float(y.mean()),
            "coef": dict(zip(cols, m.coef_[0].tolist(), strict=True))}


def fit_fold(loader, t: int, cutoff, permute_seed: int | None = None, last_cohort: int | None = None,
             dims_last: int | None = None) -> dict:
    """last_cohort caps the training cohorts (the live board: 2024, so no 2026 outcome is used before the Phase 2
    vault opens); dims_last caps the seasons the dimensions are fit on (complete seasons only)."""
    dm = dims(loader, t - 1 if dims_last is None else min(t - 1, dims_last), cutoff)
    d = training_rows(loader, t, cutoff, dm, last_cohort)
    if permute_seed is not None:
        rng = np.random.default_rng(permute_seed)
        for g in d["fit_group"].unique():
            idx = np.flatnonzero(d["fit_group"].to_numpy() == g)
            d.loc[d.index[idx], "y"] = d["y"].to_numpy()[rng.permutation(idx)]
    models = {g: fit_group(d[d["fit_group"] == g], columns(g)) for g in T.FIT_GROUPS}
    return {"dims": dm, "models": models, "t": t}


def score(fitted: dict, x: pd.DataFrame) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for g, m in fitted["models"].items():
        sel = (x["fit_group"] == g).to_numpy()
        if sel.any():
            out[sel] = m["model"].predict_proba(_standardize(x.loc[sel, m["cols"]], m["ref"]))[:, 1]
    return out


# ---- boards -------------------------------------------------------------------------------------------------------------
def entrants(loader, t: int, as_of) -> pd.DataFrame:
    """Portal entrants for season t filed by as_of, linked to a season t-1 (or t-2) roster id; one row per player, with
    his position group from the t-1 roster (else the portal's)."""
    p = portal(loader, t, as_of)
    filed = pd.to_datetime(p["transferDate"], utc=True, errors="coerce").dt.tz_localize(None)
    p = p[filed <= pd.Timestamp(as_of)].assign(season=t)
    rosters = {s: season(loader, s, as_of)["roster"] for s in (t - 2, t - 1)}
    p = link(p, rosters)
    p = p[p["player_id"].notna()].copy()
    rpos = rosters[t - 1].assign(pid=rosters[t - 1]["id"].astype(str)).drop_duplicates("pid").set_index("pid")["position"]
    p["group"] = p["player_id"].map(rpos).map(G.POSITION_GROUP).fillna(p["position"].map(G.POSITION_GROUP))
    p = p[p["group"].isin(["QB", "RB", "WR", "TE"])]
    p = p.sort_values(["player_id", "transferDate"]).drop_duplicates("player_id", keep="first")
    return p[["player_id", "group", "origin", "stars", "rating", "transferDate", "firstName", "lastName", "destination"]].reset_index(drop=True)


def credit(score_: pd.Series, by: list[pd.Series], top: int = TOP) -> np.ndarray:
    """Chance of ranking in the top `top` of its board when tied scores are ordered at random:
    (top - players strictly above) / players tied, clipped to [0, 1]."""
    df = pd.DataFrame({"s": score_.to_numpy(float)})
    keys = [f"k{i}" for i in range(len(by))]
    for k, b in zip(keys, by, strict=True):
        df[k] = b.to_numpy()
    above = df.groupby(keys)["s"].rank(method="min", ascending=False) - 1
    tied = df.groupby([*keys, "s"])["s"].transform("size")
    return np.clip((top - above.to_numpy()) / tied.to_numpy(), 0.0, 1.0)


def board(fitted: dict, loader, t: int, as_of, top: int = TOP, programs: str = "fbs") -> pd.DataFrame:
    """programs: "fbs" (the primary test) or "fcs" (FCS programs with a system in season t-1; secondary)."""
    dm = fitted["dims"]
    e = entrants(loader, t, as_of)
    teams_t = loader.load(f"teams_{t}", as_of, columns=["school", "conference"])
    conf_t = dict(zip(teams_t["school"], teams_t["conference"], strict=True))
    fbs = sorted(conf_t) if programs == "fbs" else sorted(set(season(loader, t - 1, as_of)["style"].index) - set(conf_t))
    players = player_frame(loader, t, as_of, dm)
    tf = team_frame(loader, t, as_of, dm, conf_t)
    pid = pd.Series(np.repeat(e["player_id"].to_numpy(), len(fbs)))
    team = pd.Series(np.tile(fbs, len(e)))
    x = pairs(players, tf, pid, team)
    grp = np.repeat(e["group"].to_numpy(), len(fbs))
    x["is_te"] = (grp == "TE").astype(float)
    x["fit_group"] = pd.Series(grp).map(FIT_GROUP).to_numpy()
    out = pd.DataFrame({"season": t, "player": pid, "team": team, "group": grp, "p": score(fitted, x)})
    rating = e.set_index("player_id")
    out["stars"] = pid.map(rating["stars"]).to_numpy()
    out["rating"] = pid.map(rating["rating"]).to_numpy()
    out["yards"] = x["yards"].to_numpy()
    out["eff"] = x["eff_rank"].to_numpy()
    out["board_size"] = out.groupby(["team", "group"])["player"].transform("size")
    by = [out["team"], out["group"]]
    out["fit_credit"] = credit(out["p"], by, top)
    out["transfer_rating_credit"] = credit(out["stars"].fillna(0) + out["rating"].fillna(0) / 10, by, top)
    out["raw_production_credit"] = credit(out["yards"], by, top)
    out["profile_only_credit"] = credit(out["eff"].fillna(-1e9), by, top)
    out["random_credit"] = np.minimum(1.0, top / out["board_size"])
    return out
