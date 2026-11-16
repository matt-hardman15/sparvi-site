"""CFB-PROJ: the projection as of a decision date, through the point-in-time loader only (the backtest and, later, the
portal board). Season-t yards for every QB, RB, WR, TE on each FBS team's projected roster as of May 1 of t:
team volume (P1) x share (P2) x efficiency (P3), with 80% intervals (PLAN-fit, CFB-PROJ; development in volume.py,
shares.py, efficiency.py, assemble.py on 2016-2020).

Training uses seasons before t only (outcome seasons TRAIN_FIRST .. t-1); hyperparameters are fixed from development:
volume ridge alpha 10 (no coordinator input: not in the loader, and it added nothing), shares L2 1, efficiency
shrinkage K = 300 dropbacks (QB) / 200 carries (RB), interval bins 5. Training rosters use the September rosters of
past seasons (public by then); the season-t roster is projected from what's public on May 1:
  returners   season t-1 roster at the position, minus players with a portal entry filed by the decision date
  transfers   portal entries filed by the decision date whose destination is an FBS team (destination: today's
              value, a known hindsight source, Matt Nov 15), linked to their t-1 (or t-2) roster id
  freshmen    the high-school signing class committed to the team (public by February 15)

    project(loader, t, as_of, permute_seed=None) -> DataFrame (one row per player x category)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from cfb.fit import data as D
from cfb.fit import labels as LB
from cfb.fit import model as M
from cfb.fit.portal import link
from cfb.profile import engine as G
from cfb.profile import events as EV
from cfb.proj import efficiency as E
from cfb.proj import shares as S
from football.common.filters import cfb_garbage_time

TRAIN_FIRST = 2016                  # first outcome season used for training (inputs from 2014)
FIRST_INPUT = 2014
K = {"QB": 300, "RB": 200}
COVID = 2020
N_BINS = 5
CATEGORY = {"QB": "QB", "RB": "RB"}                     # REC pool -> the player's position (WR, TE; RB receiving extra)
VOL_FEATS = ["pass_pg_1", "run_pg_1", "pass_rate_1", "games_1", "pass_pg_2", "run_pg_2", "pass_rate_2", "off_epa_1",
             "def_epa_1", "new_hc", "has_hc_old", "hc_old_shift"]
GAMES_FEATS = ["off_epa_1", "def_epa_1", "new_hc"]
LEVEL = E.LEVEL


# ---- one completed season, cached on the loader ------------------------------------------------------------------------
def season(L, s: int) -> dict:
    cache = L.__dict__.setdefault("_proj_seasons", {})
    if s in cache:
        return cache[s]
    as_of = M.done(s)
    ev, pos = D.season_events(L, s, as_of)
    teams = L.load(f"teams_{s}", as_of, columns=["school", "conference"])
    conf = dict(zip(teams["school"], teams["conference"], strict=True))
    grp = ev["player_id"].map(pos).map(G.POSITION_GROUP)
    comp = (ev["opp"] == "target") & (ev["complete"] == 1)
    pr = []
    for g, sel in (("QB", (ev["opp"] == "dropback") & (grp == "QB")), ("RB", (ev["opp"] == "carry") & (grp == "RB")),
                   ("REC", comp & grp.isin(["WR", "TE", "RB"]))):
        r = ev[sel]
        a = r.groupby("player_id").agg(n=("yards", "size"), yards=("yards", "sum"), team=("offense", lambda x: x.mode().iloc[0])).reset_index()
        a["level"] = [LEVEL[LB.tier(t, s, conf)] for t in a["team"]]
        pr.append(a.assign(season=s, group=g))
    p = pd.concat(pr, ignore_index=True)
    p["rate"] = p["yards"] / p["n"]
    plays = ev[ev["kind"].isin(["pass", "run"])].drop_duplicates("play_id")
    db = ev[ev["opp"] == "dropback"].drop_duplicates("play_id")
    rb = ev[(ev["opp"] == "carry") & (grp == "RB")].drop_duplicates("play_id")
    rec = ev[comp].drop_duplicates("play_id")
    # Team play counts straight from the play-by-play, same garbage-time rule, whether or not anyone on the play was
    # matched to a roster (DATA-QA: Mississippi State's unmatched lead back left it 8 run plays a game in 2020-2022).
    raw = L.load(f"plays_{s}", as_of, columns=["gameId", "offense", "playType", "period", "offenseScore", "defenseScore"])
    raw = raw[raw["playType"].isin(EV.PASS_TYPES | EV.RUN_TYPES)]
    raw = raw[~cfb_garbage_time(raw["period"].to_numpy(), (raw["offenseScore"] - raw["defenseScore"]).abs().to_numpy())]
    is_pass = raw["playType"].isin(EV.PASS_TYPES)
    team = pd.DataFrame({"games": raw.groupby("offense")["gameId"].nunique(), "pass": raw[is_pass].groupby("offense").size(),
                         "run": raw[~is_pass].groupby("offense").size(),
                         "off_epa": plays.groupby("offense")["epa"].mean(), "def_epa": plays.groupby("defense")["epa"].mean(),
                         "dropbacks": db.groupby("offense").size(), "pass_yards": db.groupby("offense")["yards"].sum(),
                         "runs": raw[~is_pass].groupby("offense").size(), "rb_carries": rb.groupby("offense").size(),
                         "rb_yards": rb.groupby("offense")["yards"].sum(), "rec_yards": rec.groupby("offense")["yards"].sum()}).fillna(0)
    team = team[team.index.isin(conf)].rename_axis("team").reset_index().assign(season=s)
    team["pass_pg"], team["run_pg"] = team["pass"] / team["games"], team["run"] / team["games"]
    team["pass_rate"] = team["pass"] / (team["pass"] + team["run"]).clip(lower=1)
    team["rb_frac"] = team["rb_carries"] / team["runs"].clip(lower=1)
    # Per pass play (counted from every credited pass play), not per matched dropback: a team whose passers went
    # unmatched (UL Monroe 2022: 1 matched dropback, 192 pass plays) otherwise produced 1,980 yards per dropback (1.0.1).
    team["rec_ypd"] = team["rec_yards"] / team["pass"].clip(lower=1)
    validate_season(s, team, p, S.work(ev, pos))
    cache[s] = {"p": p, "team": team, "conf": conf, "work": S.work(ev, pos), "eff": S.efficiency(ev, pos),
                "roster": L.load(f"players_roster_{s}", as_of),
                "played": ev.groupby("player_id")["offense"].agg(lambda x: x.mode().iloc[0]).to_dict()}
    return cache[s]


# ---- range assertions (DATA-QA, Matt Nov 16): a value outside these fails the build instead of reaching a model ---------
# pass plays per game down to 3: the service academies' option offenses (Army 2017: 4.8)
TEAM_BOUNDS = {"pass_pg": (3, 70), "run_pg": (10, 70), "rec_ypd": (2, 15), "rb_frac": (0, 1), "games": (1, 17)}
RATE_BOUNDS = {"QB": (-5, 20), "RB": (-5, 20), "REC": (0, 40)}          # per play, players with RATE_MIN_N plays
RATE_MIN_N = {"QB": 50, "RB": 30, "REC": 20}
PROJ_MAX = {"QB": 7000, "RB": 3500, "REC": 3000}                        # season yards


def validate_season(s: int, team: pd.DataFrame, p: pd.DataFrame, work: pd.DataFrame) -> None:
    bad = []
    full = team[team["games"] >= 5]
    for col, (lo, hi) in TEAM_BOUNDS.items():
        x = full[(full[col] < lo) | (full[col] > hi)]
        bad += [f"{s} {r.team}: {col} {getattr(r, col):.2f} outside [{lo}, {hi}]" for r in x.itertuples()]
    for g, (lo, hi) in RATE_BOUNDS.items():
        x = p[(p["group"] == g) & (p["n"] >= RATE_MIN_N[g]) & ((p["rate"] < lo) | (p["rate"] > hi))]
        bad += [f"{s} {r.team} {r.player_id}: {g} rate {r.rate:.2f} outside [{lo}, {hi}]" for r in x.itertuples()]
    tot = work.groupby(["team", "group"])["share"].sum()
    bad += [f"{s} {t} {g}: shares sum to {v:.3f}" for (t, g), v in tot[tot > 1 + 1e-6].items()]
    if bad:
        raise ValueError(f"season {s} failed range checks:\n" + "\n".join(bad[:20]))


def validate_projection(c: pd.DataFrame) -> None:
    bad = []
    tot = c.groupby(["team", "group"])["share"].sum()
    bad += [f"{t} {g}: projected shares sum to {v:.3f}" for (t, g), v in tot[tot > 1 + 1e-6].items()]
    for g, hi in PROJ_MAX.items():
        x = c[(c["group"] == g) & ((c["proj"] > hi) | (c["proj"] < -100))]
        bad += [f"{r.team} {r.key}: {g} projection {r.proj:.0f}" for r in x.itertuples()]
    if bad:
        raise ValueError("projection failed range checks:\n" + "\n".join(bad[:20]))


def _recruits(L, last: int, as_of) -> pd.DataFrame:
    return M.recruits(L, last, as_of)


def _p_leave(L, t: int, rosters: dict, recruits: pd.DataFrame) -> dict[int, float]:
    """P(not on the same team next season | experience), from the two transitions before t-1 -> t."""
    hist = []
    for h in [h for h in (t - 2, t - 1) if h - 1 in rosters and h in rosters]:
        r0 = rosters[h - 1].assign(pid=lambda d: d["id"].astype(str))
        r1 = set(zip(rosters[h]["team"], rosters[h]["id"].astype(str), strict=True))
        e0 = D.experience(h - 1, {s: rosters[s] for s in rosters if s <= h - 1}, recruits)
        r0 = r0[r0["position"].map(G.POSITION_GROUP).notna()]
        hist.append(pd.DataFrame({"exp": r0["pid"].map(e0).fillna(1).clip(1, 6).astype(int),
                                  "gone": [(tm, p) not in r1 for tm, p in zip(r0["team"], r0["pid"], strict=True)]}))
    return pd.concat(hist).groupby("exp")["gone"].mean().to_dict()


PORTAL_FIRST = 2021                 # CFBD's portal records start with the 2021 window


def candidates(L, t: int, as_of, live: bool) -> pd.DataFrame:
    """Projected-roster rows for season t. live=True: the season being projected, as of the decision date. live=False:
    a past season for training, with its actual shares. In the portal era (PORTAL_FIRST on) a training season's roster
    is built the same way as the projected one, as of May 1 of that season: portal entries filed by then leave, those
    whose destination is an FBS team arrive (1.2.0, Matt Nov 16: QBs and TEs moving down a level in the portal era were
    projected ~1.3-1.5x the share they got, which hindsight September rosters didn't teach). Before the portal: movers
    from the September rosters."""
    may1 = live or t >= PORTAL_FIRST
    dec = pd.Timestamp(as_of) if live else pd.Timestamp(f"{t}-05-01")
    seasons_ = range(FIRST_INPUT, t + (0 if live else 1))
    data = {s: season(L, s) for s in seasons_}
    rosters = {s: d["roster"] for s, d in data.items()}
    recruits = _recruits(L, t, as_of)
    conf_prev = data[t - 1]["conf"]
    if live:
        teams_t = L.load(f"teams_{t}", as_of, columns=["school", "conference"])
        conf_t = dict(zip(teams_t["school"], teams_t["conference"], strict=True))
    else:
        conf_t = data[t]["conf"]
    exp = D.experience(t - 1, {s: r for s, r in rosters.items() if s <= t - 1}, recruits)
    p_leave = _p_leave(L, t, {s: r for s, r in rosters.items() if s <= t - 1}, recruits)
    wk = data[t - 1]["work"]
    prev_w = wk.set_index(["team", "player_id", "group"])["share"]
    any_w = wk.sort_values("share", ascending=False).drop_duplicates(["player_id", "group"]).set_index(["player_id", "group"])["share"]
    eff = data[t - 1]["eff"]
    lvl = LEVEL
    rows = []
    if may1:
        pt = M.portal(L, t, dec).assign(season=t)
        filed = pd.to_datetime(pt["transferDate"], utc=True, errors="coerce").dt.tz_localize(None) <= dec
        pt = pt[filed]
        pt = link(pt, {s: rosters[s] for s in (t - 2, t - 1) if s in rosters}) if len(pt) else pt.assign(player_id=pd.Series(dtype=object))
        pt = pt[pt["player_id"].notna()]
        leaving = set(pt["player_id"].astype(str))
        incoming = pt[pt["destination"].isin(conf_t) & (pt["destination"] != pt["origin"])]
        rpos = rosters[t - 1].assign(pid=lambda d: d["id"].astype(str)).drop_duplicates("pid").set_index("pid")["position"]
        moves = pd.DataFrame({"prev_id": incoming["player_id"].astype(str), "team_to": incoming["destination"], "team_from": incoming["origin"],
                              "group": incoming["player_id"].astype(str).map(rpos).map(G.POSITION_GROUP).fillna(
                                  incoming["position"].map(G.POSITION_GROUP))})
        moves = moves[moves["group"].notna()].drop_duplicates("prev_id")
        moves["out_key"] = moves["prev_id"]
        if not live:                                   # training: follow a transfer whose id changed (his season-t id)
            played = {s: data[s]["played"] for s in (t - 1, t)}
            tr = LB.transfers(rosters, t, M.portal(L, t, M.done(t)).assign(season=t), played)
            new_id = dict(zip(tr["prev_id"].dropna().astype(str), tr.loc[tr["prev_id"].notna(), "player_id"].astype(str), strict=True))
            moves["out_key"] = moves["prev_id"].map(lambda k: new_id.get(k, k))
    else:
        played = {s: data[s]["played"] for s in (t - 1, t)}
        tr = LB.transfers(rosters, t, None, played)
        tr = tr[tr["team_to"].isin(conf_t)]
        leaving = set(tr["prev_id"].dropna().astype(str))
        moves = pd.DataFrame({"prev_id": tr["prev_id"].astype("string"), "team_to": tr["team_to"], "team_from": tr["team_from"],
                              "group": tr["group"], "out_key": tr["player_id"].astype(str)})
    ros_prev = rosters[t - 1].assign(pid=lambda d: d["id"].astype(str), group=lambda d: d["position"].map(G.POSITION_GROUP))
    ros_prev = ros_prev[ros_prev["group"].notna() & ros_prev["team"].isin(conf_t) & ~ros_prev["pid"].isin(leaving)].drop_duplicates(["team", "pid"])
    for r in ros_prev.itertuples():
        e = int(np.clip(exp.get(r.pid, 1), 1, 6))
        rows.append({"team": r.team, "player_id": r.pid, "key": r.pid, "pos": r.group, "status": "returner", "exp": e,
                     "log_stay": float(np.log(max(1 - p_leave.get(e, 0.3), 0.02))), "dlvl": 0.0})
    for r in moves.itertuples():
        pid = None if pd.isna(r.prev_id) else str(r.prev_id)
        rows.append({"team": r.team_to, "player_id": pid, "key": pid or r.out_key, "pos": r.group, "status": "transfer",
                     "exp": int(np.clip(exp.get(pid, 2) if pid else 2, 1, 6)), "log_stay": 0.0,
                     "dlvl": float(lvl[LB.tier(r.team_to, t, conf_t)] - lvl[LB.tier(r.team_from, t - 1, conf_prev)]),
                     "out_key": r.out_key})
    fr = recruits[(recruits["year"] == t) & recruits["committedTo"].isin(conf_t) & recruits["athleteId"].notna()]
    for r in fr.itertuples():
        g = S.RECRUIT_POS.get(r.position)
        if g:
            pid = str(int(r.athleteId))
            rows.append({"team": r.committedTo, "player_id": pid, "key": pid, "pos": g, "status": "freshman", "exp": 0,
                         "log_stay": 0.0, "dlvl": 0.0})
    c = pd.DataFrame(rows)
    if "out_key" not in c:
        c["out_key"] = np.nan
    c["out_key"] = c["out_key"].fillna(c["key"])
    c = c.drop_duplicates(["team", "pos", "key"])
    c = c.assign(group=c["pos"].map(S.POOLS_OF)).explode("group").reset_index(drop=True)
    c["last_share"] = [0.0 if st == "freshman" or p is None else (prev_w.get((tm, p, g), 0.0) if st == "returner" else any_w.get((p, g), 0.0))
                       for tm, p, g, st in zip(c["team"], c["player_id"], c["group"], c["status"], strict=True)]
    c["any_last"] = [any_w.get((p, g), 0.0) if p else 0.0 for p, g in zip(c["player_id"], c["group"], strict=True)]
    c["eff"] = [eff.get((p, g), np.nan) if p else np.nan for p, g in zip(c["player_id"], c["group"], strict=True)]
    hs = (recruits[recruits["athleteId"].notna()].assign(pid=lambda d: d["athleteId"].astype(str))
          .sort_values(["year", "rating"], ascending=[True, False]).drop_duplicates("pid").set_index("pid")["rating"])
    c["hs"] = c["player_id"].map(hs)
    c["season"] = t
    c["level_t"] = [LEVEL[LB.tier(tm, t, conf_t)] for tm in c["team"]]
    if not live:                                       # training rows: actual shares from season t, and the outside option
        actual = data[t]["work"].set_index(["team", "player_id", "group"])["share"]
        c["actual"] = [actual.get((tm, o, g), 0.0) for tm, o, g in zip(c["team"], c["out_key"], c["group"], strict=True)]
        tot = c.groupby(["team", "group"])["actual"].sum()
        c["unprojected"] = [max(0.0, 1 - tot[(tm, g)]) for tm, g in zip(c["team"], c["group"], strict=True)]
    return c


# ---- training and projection -------------------------------------------------------------------------------------------
def _volume_rows(L, seasons_) -> pd.DataFrame:
    co = L.load("coaches", L.__dict__.get("_proj_as_of"))
    head = co.sort_values("games").groupby(["season", "school"])["coach_id"].last()
    where = {(s, c): sc for (s, sc), c in head.items()}
    rows = []
    for t in seasons_:
        prev = {lag: season(L, t - lag)["team"].set_index("team") for lag in (1, 2) if t - lag >= FIRST_INPUT}
        cur = season(L, t)["team"].set_index("team") if t in L.__dict__.get("_proj_outcome_ok", ()) else None
        teams = sorted(prev[1].index) if cur is None else sorted(set(cur.index) & set(prev[1].index))
        for team in teams:
            r = {"season": t, "team": team}
            for lag in (1, 2):
                src = prev.get(lag)
                row = src.loc[team] if src is not None and team in src.index else None
                for c in ("pass_pg", "run_pg", "pass_rate", "games"):
                    r[f"{c}_{lag}"] = np.nan if row is None else row[c]
            r["off_epa_1"], r["def_epa_1"] = prev[1].loc[team, "off_epa"], prev[1].loc[team, "def_epa"]
            h, hp = head.get((t, team)), head.get((t - 1, team))
            r["new_hc"] = float(h is not None and hp is not None and h != hp)
            old = where.get((t - 1, h)) if r["new_hc"] else None
            r["hc_old_pass_rate"] = prev[1].loc[old, "pass_rate"] if old is not None and old in prev[1].index else np.nan
            if cur is not None:
                for c in ("pass_pg", "run_pg", "games"):
                    r[f"y_{c}"] = cur.loc[team, c]
            rows.append(r)
    d = pd.DataFrame(rows)
    for c in ("pass_pg", "run_pg", "pass_rate", "games"):
        d[f"{c}_2"] = d[f"{c}_2"].fillna(d[f"{c}_1"])
    d["has_hc_old"] = d["hc_old_pass_rate"].notna().astype(float)
    d["hc_old_shift"] = (d["hc_old_pass_rate"] - d["pass_rate_1"]).fillna(0.0)
    return d


def _ridge(X_tr, y_tr, X_te, alpha=10.0):
    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd[sd == 0] = 1
    return Ridge(alpha=alpha).fit((X_tr - mu) / sd, y_tr).predict((X_te - mu) / sd)


def _perm(v: np.ndarray, rng) -> np.ndarray:
    return v[rng.permutation(len(v))] if rng is not None else v


def project(L, t: int, as_of, permute_seed: int | None = None, last_train: int | None = None) -> pd.DataFrame:
    """Projected season-t yards per (player, category) as of `as_of`, with 80% intervals and the two baselines.
    permute_seed shuffles every training target (the harness's label-permutation test). last_train caps the outcome
    seasons used for training (the live board: 2025 until the Phase 2 vault opens)."""
    as_of = pd.Timestamp(as_of)
    L.__dict__["_proj_as_of"] = as_of
    rng = np.random.default_rng(permute_seed) if permute_seed is not None else None
    train = list(range(TRAIN_FIRST, t if last_train is None else min(t, last_train + 1)))
    L.__dict__["_proj_outcome_ok"] = set(train)
    # P1: team volume.
    vtr, vte = _volume_rows(L, train), _volume_rows(L, [t])
    vol = vte[["team"]].copy()
    for tg, feats in (("pass_pg", VOL_FEATS), ("run_pg", VOL_FEATS), ("games", GAMES_FEATS)):
        tr = vtr if tg != "games" else vtr[vtr["season"] != COVID]
        y = _perm(tr[f"y_{tg}"].to_numpy(float), rng)
        vol[tg] = _ridge(tr[feats].to_numpy(float), y, vte[feats].to_numpy(float))
    vol = vol.set_index("team")
    # P2: shares.
    ctr = pd.concat([candidates(L, s, as_of, live=False) for s in train], ignore_index=True)
    cte = candidates(L, t, as_of, live=True)
    if rng is not None:
        for g in S.GROUPS:
            m = (ctr["group"] == g).to_numpy()
            ctr.loc[m, "actual"] = _perm(ctr.loc[m, "actual"].to_numpy(), rng)
    Xtr, Xte = S.design(ctr), S.design(cte)
    cte["share"] = np.nan
    state = {"t": t, "betas": {}, "lam": {}}
    for g in S.GROUPS:
        tr, te = (ctr["group"] == g).to_numpy(), (cte["group"] == g).to_numpy()
        mu, sd = Xtr[tr].mean(), Xtr[tr].std().replace(0, 1)
        keys_tr = (ctr["season"].astype(str) + "|" + ctr["team"]).to_numpy()[tr]
        y0 = ctr[tr].groupby(keys_tr)["unprojected"].first().to_dict()
        b = S.fit(((Xtr[tr] - mu) / sd).to_numpy(), keys_tr, ctr.loc[tr, "actual"].to_numpy(), y0)
        cte.loc[te, "share"] = S.predict(((Xte[te] - mu) / sd).to_numpy(), cte.loc[te, "team"].to_numpy(), b)
        cte.loc[te, "utility"] = ((Xte[te] - mu) / sd).to_numpy() @ b[:-1]
        state["betas"][g] = (mu, sd, b)
    # P3: efficiency (players), with the supporting-cast weight fit on the training seasons.
    p = pd.concat([season(L, s)["p"] for s in range(FIRST_INPUT, t)], ignore_index=True)
    pq = p[p["group"].isin(["QB", "RB"])]
    cte["rate"] = np.nan
    for g in ("QB", "RB"):
        d_tr = _rate_rows(pq, g, train)
        base_tr = E.project_players(d_tr, pq[pq["season"] < t], K[g], [s for s in range(FIRST_INPUT, t)])
        dev_tr = E.team_dev(pq, d_tr["season"], d_tr["team_t"], d_tr["group"])
        resid = _perm((d_tr["actual"].to_numpy() - base_tr), rng)
        lam = float((dev_tr * resid).sum() / max((dev_tr * dev_tr).sum(), 1e-9))
        state["lam"][g] = lam
        m = (cte["group"] == g).to_numpy()
        d_te = _history(cte[m], pq, t)
        base = E.project_players(d_te, pq, K[g], list(range(FIRST_INPUT, t)))
        cte.loc[m, "rate"] = base + lam * E.team_dev(pq, d_te["season"], d_te["team_t"], d_te["group"])
    # Team context: RB share of runs, receiving yards per dropback, each from the team's last two seasons (development:
    # adding the projected QBs to receiving yards per dropback changed receiving r by under 0.001, so it isn't used).
    teams = pd.concat([season(L, s)["team"] for s in range(FIRST_INPUT, t)], ignore_index=True).set_index(["season", "team"])
    ctx_tr, ctx_te = _context_rows(teams, train, True), _context_rows(teams, [t], False)
    for col, feats in (("rb_frac", ["rb_frac_1", "rb_frac_2"]), ("rec_ypd", ["rec_ypd_1", "rec_ypd_2"])):
        A = np.c_[np.ones(len(ctx_tr)), ctx_tr[feats].to_numpy(float)]
        beta = np.linalg.lstsq(A, _perm(ctx_tr[col].to_numpy(float), rng), rcond=None)[0]
        ctx_te[f"{col}_hat"] = np.c_[np.ones(len(ctx_te)), ctx_te[feats].to_numpy(float)] @ beta
    ctx_te = ctx_te.set_index("team")
    # Assembly.
    games = cte["team"].map(vol["games"]).to_numpy()
    dropbacks = cte["team"].map(vol["pass_pg"]).to_numpy() * games
    runs = cte["team"].map(vol["run_pg"]).to_numpy() * games
    rb_frac = cte["team"].map(ctx_te["rb_frac_hat"]).to_numpy()
    rec_ypd = cte["team"].map(ctx_te["rec_ypd_hat"]).to_numpy()
    cte["proj"] = np.select([cte["group"] == "QB", cte["group"] == "RB"],
                            [cte["share"] * dropbacks * cte["rate"], cte["share"] * runs * rb_frac * cte["rate"]],
                            cte["share"] * dropbacks * rec_ypd)
    cte["category"] = np.where(cte["group"] == "REC", cte["pos"] + "_rec", cte["pos"])
    # Baselines (pre-registered): his own last-season yards; his last share x the destination's last-season yards.
    yards = p.set_index(["season", "player_id", "group"])["yards"]
    cte["carry"] = [0.0 if not pid else yards.get((t - 1, pid, g), 0.0) for pid, g in zip(cte["player_id"], cte["group"], strict=True)]
    col = {"QB": "pass_yards", "RB": "rb_yards", "REC": "rec_yards"}
    team_prev = [teams[col[g]].get((t - 1, tm), np.nan) for tm, g in zip(cte["team"], cte["group"], strict=True)]
    cte["share_only"] = np.where(cte["status"] == "freshman", 0.0, cte["last_share"] * np.nan_to_num(np.array(team_prev, float)))
    state.update(vol=vol, ctx=ctx_te, pq=pq, cte=cte.copy(), open=_room_open(cte), hs_fill=float(cte["hs"].min()))
    L.__dict__.setdefault("_proj_state", {})[(t, as_of, permute_seed, last_train)] = state
    cte = cte[cte["proj"].notna()].reset_index(drop=True)
    validate_projection(cte)
    return _intervals(cte, _interval_bins())

# ---- counterfactual: an entrant at any program (the portal board) -------------------------------------------------------
def _room_open(c: pd.DataFrame) -> pd.Series:
    """(team, pool) -> the opening shares.design gives a newcomer: 1 - the returners' expected share."""
    ret = c["status"] == "returner"
    kept = (c["last_share"] * np.exp(c["log_stay"])).where(ret, 0.0).groupby([c["team"], c["group"]]).sum()
    return (1 - kept).clip(0, 1)


def counterfactual_share(u: np.ndarray, room_lse: np.ndarray) -> np.ndarray:
    """His share if added to a room whose log-sum-exp of utilities (its candidates and the outside option) is room_lse."""
    m = np.maximum(u, room_lse)
    return np.exp(u - m) / (np.exp(u - m) + np.exp(room_lse - m))


def validate_counterfactual(c: pd.DataFrame) -> None:
    bad = [f"{r.team} {r.key}: {r.group} counterfactual {r.proj:.0f}" for r in c.itertuples()
           if r.proj > PROJ_MAX[r.group] or r.proj < -100 or not 0 <= r.share <= 1]
    if bad:
        raise ValueError("counterfactual projection failed range checks:\n" + "\n".join(bad[:20]))


def whatif(L, t: int, as_of, entrants: pd.DataFrame, programs: list[str], last_train: int | None = None) -> pd.DataFrame:
    """Projected season-t yards for each entrant (columns player_id = his season t-1 id, group = QB/RB/WR/TE,
    team_from) if he joined each program, scored against that program's projected room as of `as_of` (its returners,
    committed transfers, and signing class; himself removed if he's already projected there). Same model and
    intervals as project(); one row per entrant x program x category."""
    as_of = pd.Timestamp(as_of)
    key = (t, as_of, None, last_train)
    if key not in L.__dict__.get("_proj_state", {}):
        project(L, t, as_of, last_train=last_train)
    st = L.__dict__["_proj_state"][key]
    data = {s: season(L, s) for s in range(FIRST_INPUT, t)}
    rosters = {s: d["roster"] for s, d in data.items()}
    recruits = _recruits(L, t, as_of)
    exp = D.experience(t - 1, rosters, recruits)
    wk = data[t - 1]["work"]
    any_w = wk.sort_values("share", ascending=False).drop_duplicates(["player_id", "group"]).set_index(["player_id", "group"])["share"]
    eff = data[t - 1]["eff"]
    hs = (recruits[recruits["athleteId"].notna()].assign(pid=lambda d: d["athleteId"].astype(str))
          .sort_values(["year", "rating"], ascending=[True, False]).drop_duplicates("pid").set_index("pid")["rating"])
    teams_t = L.load(f"teams_{t}", as_of, columns=["school", "conference"])
    conf_t = dict(zip(teams_t["school"], teams_t["conference"], strict=True))
    conf_prev = data[t - 1]["conf"]
    e = entrants.assign(player_id=entrants["player_id"].astype(str)).drop_duplicates("player_id")
    e = e.assign(pool=e["group"].map(S.POOLS_OF)).explode("pool").reset_index(drop=True)
    progs = [p for p in programs if p in conf_t and p in st["vol"].index]
    rows = e.loc[e.index.repeat(len(progs))].reset_index(drop=True)
    rows["team"] = np.tile(progs, len(e))
    c = pd.DataFrame({"team": rows["team"], "player_id": rows["player_id"], "key": rows["player_id"], "pos": rows["group"],
                      "group": rows["pool"], "status": "transfer", "log_stay": 0.0, "season": t})
    c["exp"] = [int(np.clip(exp.get(p, 2), 1, 6)) for p in c["player_id"]]
    lvl_from = {f: LEVEL[LB.tier(f, t - 1, conf_prev)] for f in rows["team_from"].dropna().unique()}
    lvl_to = {p: LEVEL[LB.tier(p, t, conf_t)] for p in progs}
    c["level_t"] = rows["team"].map(lvl_to).to_numpy()
    c["dlvl"] = c["level_t"] - rows["team_from"].map(lvl_from).fillna(0).to_numpy()
    c["last_share"] = [any_w.get((p, g), 0.0) for p, g in zip(c["player_id"], c["group"], strict=True)]
    c["any_last"] = c["last_share"]
    c["eff"] = [eff.get((p, g), np.nan) for p, g in zip(c["player_id"], c["group"], strict=True)]
    c["hs"] = c["player_id"].map(hs)
    c["open_given"] = [st["open"].get((tm, g), 1.0) for tm, g in zip(c["team"], c["group"], strict=True)]
    c["hs_fill"] = st["hs_fill"]
    room = st["cte"]
    c["share"] = np.nan
    for g in S.GROUPS:
        m = (c["group"] == g).to_numpy()
        if not m.any():
            continue
        mu, sd, b = st["betas"][g]
        u = ((S.design(c[m]) - mu) / sd).to_numpy() @ b[:-1]
        r = room[room["group"] == g]
        # Each room's log-sum-exp once (its candidates and the outside option); an entrant already projected in that
        # room is taken out of it exactly (computed without him), so he isn't counted twice.
        out_u = float(b[-1])
        total = r.groupby("team")["utility"].apply(lambda v, o=out_u: float(np.logaddexp.reduce(np.append(v.to_numpy(), o))))
        own = {(tm, k): float(np.logaddexp.reduce(np.append(q.loc[q["key"] != k, "utility"].to_numpy(), b[-1])))
               for tm, q in r.groupby("team") for k in set(q["key"]) & set(c.loc[m, "key"])}
        teams_m, keys_m = c.loc[m, "team"].to_numpy(), c.loc[m, "key"].to_numpy()
        lse = np.array([own.get((tm, k), total.get(tm, b[-1])) for tm, k in zip(teams_m, keys_m, strict=True)])
        c.loc[m, "share"] = counterfactual_share(u, lse)
    pq = st["pq"]
    c["rate"] = np.nan
    for g in ("QB", "RB"):
        m = (c["group"] == g).to_numpy()
        if m.any():
            d = _history_fast(c[m], pq, t)
            c.loc[m, "rate"] = (E.project_players(d, pq, K[g], list(range(FIRST_INPUT, t)))
                                + st["lam"][g] * E.team_dev(pq, d["season"], d["team_t"], d["group"]))
    vol, ctx = st["vol"], st["ctx"]
    games = c["team"].map(vol["games"]).to_numpy()
    dropbacks = c["team"].map(vol["pass_pg"]).to_numpy() * games
    runs = c["team"].map(vol["run_pg"]).to_numpy() * games
    c["proj"] = np.select([c["group"] == "QB", c["group"] == "RB"],
                          [c["share"] * dropbacks * c["rate"], c["share"] * runs * c["team"].map(ctx["rb_frac_hat"]).to_numpy() * c["rate"]],
                          c["share"] * dropbacks * c["team"].map(ctx["rec_ypd_hat"]).to_numpy())
    c["category"] = np.where(c["group"] == "REC", c["pos"] + "_rec", c["pos"])
    c = c[c["proj"].notna()].reset_index(drop=True)
    validate_counterfactual(c)
    return _intervals(c, _interval_bins())



def _history_fast(c: pd.DataFrame, pq: pd.DataFrame, t: int) -> pd.DataFrame:
    """_history for many rows of the same players (the counterfactual: each entrant at every program): each player's
    two previous seasons looked up once, then joined; the destination fields come from each row."""
    k = pq.set_index(["season", "player_id", "group"])
    uniq = c[["player_id", "group"]].drop_duplicates()
    h = []
    for r in uniq.itertuples():
        h1 = k.loc[(t - 1, r.player_id, r.group)] if r.player_id and (t - 1, r.player_id, r.group) in k.index else None
        h2 = k.loc[(t - 2, r.player_id, r.group)] if r.player_id and (t - 2, r.player_id, r.group) in k.index else None
        h.append({"player_id": r.player_id, "group": r.group, "n1": 0 if h1 is None else h1["n"], "y1": 0.0 if h1 is None else h1["yards"],
                  "lvl1": np.nan if h1 is None else h1["level"], "n2": 0 if h2 is None else h2["n"], "y2": 0.0 if h2 is None else h2["yards"]})
    d = c[["player_id", "group", "team", "level_t"]].reset_index().merge(pd.DataFrame(h), on=["player_id", "group"], how="left")
    d = d.set_index("index").loc[c.index]
    return pd.DataFrame({"season": t, "group": d["group"], "team_t": d["team"], "level_t": d["level_t"], "n1": d["n1"], "y1": d["y1"],
                         "lvl1": d["lvl1"], "n2": d["n2"], "y2": d["y2"]}, index=c.index)


def _rate_rows(pq: pd.DataFrame, g: str, seasons_) -> pd.DataFrame:
    """Training rows for the supporting-cast weight: players at the backup minimum in a training season, with history."""
    k = pq.set_index(["season", "player_id", "group"])
    rows = []
    for t in seasons_:
        for r in pq[(pq["season"] == t) & (pq["group"] == g) & (pq["n"] >= E.MIN_EVAL[g])].itertuples():
            h1 = k.loc[(t - 1, r.player_id, g)] if (t - 1, r.player_id, g) in k.index else None
            h2 = k.loc[(t - 2, r.player_id, g)] if (t - 2, r.player_id, g) in k.index else None
            rows.append({"season": t, "group": g, "team_t": r.team, "level_t": r.level, "actual": r.rate,
                         "n1": 0 if h1 is None else h1["n"], "y1": 0.0 if h1 is None else h1["yards"], "lvl1": np.nan if h1 is None else h1["level"],
                         "n2": 0 if h2 is None else h2["n"], "y2": 0.0 if h2 is None else h2["yards"]})
    return pd.DataFrame(rows)


def _history(c: pd.DataFrame, pq: pd.DataFrame, t: int) -> pd.DataFrame:
    k = pq.set_index(["season", "player_id", "group"])
    rows = []
    for r in c.itertuples():
        h1 = k.loc[(t - 1, r.player_id, r.group)] if r.player_id and (t - 1, r.player_id, r.group) in k.index else None
        h2 = k.loc[(t - 2, r.player_id, r.group)] if r.player_id and (t - 2, r.player_id, r.group) in k.index else None
        rows.append({"season": t, "group": r.group, "team_t": r.team, "level_t": r.level_t,
                     "n1": 0 if h1 is None else h1["n"], "y1": 0.0 if h1 is None else h1["yards"], "lvl1": np.nan if h1 is None else h1["level"],
                     "n2": 0 if h2 is None else h2["n"], "y2": 0.0 if h2 is None else h2["yards"]})
    return pd.DataFrame(rows, index=c.index)


def _context_rows(teams: pd.DataFrame, seasons_, with_y: bool) -> pd.DataFrame:
    rows = []
    for s in seasons_:
        names = [tm for (ss, tm) in teams.index if ss == (s if with_y else s - 1)]
        for team in names:
            if (s - 1, team) not in teams.index:
                continue
            r = {"season": s, "team": team}
            for col in ("rb_frac", "rec_ypd"):
                r[f"{col}_1"] = teams.loc[(s - 1, team), col]
                r[f"{col}_2"] = teams.loc[(s - 2, team), col] if (s - 2, team) in teams.index else r[f"{col}_1"]
                if with_y:
                    r[col] = teams.loc[(s, team), col]
            rows.append(r)
    return pd.DataFrame(rows)


def _interval_bins() -> dict:
    """80% intervals from the development residuals (leave-one-season-out, 2016-2020) by category and projected-yards
    bin, fixed before the test (cfb/proj/output/interval_bins.json, written by assemble.py)."""
    import json

    from sparvi.config import ROOT
    return json.loads((ROOT / "cfb" / "proj" / "output" / "interval_bins.json").read_text(encoding="utf-8"))


def _intervals(c: pd.DataFrame, bins: dict) -> pd.DataFrame:
    c = c.copy()
    c["lo80"], c["hi80"] = np.nan, np.nan
    for cat, b in bins.items():
        m = (c["category"] == cat).to_numpy()
        if not m.any():
            continue
        edges = np.array(b["edges"])
        i = np.clip(np.searchsorted(edges, c.loc[m, "proj"], side="right") - 1, 0, len(b["q10"]) - 1)
        c.loc[m, "lo80"] = (c.loc[m, "proj"].to_numpy() + np.array(b["q10"])[i]).clip(min=0)
        c.loc[m, "hi80"] = c.loc[m, "proj"].to_numpy() + np.array(b["q90"])[i]
    return c
