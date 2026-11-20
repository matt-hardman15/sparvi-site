"""The fourth-down decision engine (F4, F4-DECIDE): for a 4th-down situation, the offense's win probability if it
goes for it, kicks a field goal, or punts, from the four registered models (models.py) and a few empirical tables,
all fit on the same training seasons.

Each choice is valued as an expectation over what can happen next, not at a single plugged-in outcome (win
probability is not linear in field position, and the punt model predicts the median):

    go     P(convert) x E[WP | converted, yards gained] + (1 - P) x E[WP | failed, yards gained]
           yards gained beyond the line to gain (or short of it) from the training seasons' 4th-down plays, by
           yards-to-go bucket; a gain to the goal line is a touchdown
    kick   P(good) x WP(after the kickoff, up 3) + (1 - P) x WP(opponent's ball at the spot of the kick, or its 20);
           offered only up to the distances teams actually try (FG_SUPPORT)
    punt   (1 - P(block)) x E[WP | net yards] + P(block) x WP(opponent's ball BLOCK_YARDS behind the line)
           net yards = the punt model's prediction plus the training punts' residuals by field position (quantiles)

After a change of possession the opponent's win probability comes from the same model with the state flipped
(margin, timeouts, home, strength, 2nd-half kickoff), and the offense gets one minus it. A play that runs out the
half moves to the 2nd-half kickoff; one that runs out the game is decided by the score (a tie counts one half).

Uncertainty: `bootstrap()` refits everything on games resampled with replacement and returns the spread of each
situation's go-minus-best-kick margin, so a recommendation can say when it is a toss-up.

Simplifications, stated on the page: a touchdown is worth TD_POINTS (the try isn't modeled); kickoffs put the ball at
the last training season's average start; no turnovers returned for scores, safeties, penalties or fakes beyond what
the conversion rate already contains; clock runoff is the training median for the play type; overtime states are
valued by the model as they are.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl.fourth import fg as F
from nfl.fourth import models as M
from nfl.fourth import wp as W
from nfl.fourth.data import load_plays, scrimmage

TD_POINTS, FG_POINTS = 7, 3
BLOCK_YARDS = 10             # a blocked punt: the opponent's ball this far behind the line of scrimmage
TOUCHBACK = 20               # punts into the end zone, and missed kicks from inside it
PUNT_MIN_YARDLINE = 30       # no punt option closer than this to the goal line (teams almost never punt there)
FG_SUPPORT = 0.99            # no field-goal option beyond this quantile of the distances teams tried (training kicks):
                             # past it the make model is extrapolating, and its curve bends the wrong way
QUANTILES = np.linspace(0.025, 0.975, 20)      # outcome nodes: 20 equal-weight quantiles of each distribution
YTG_BUCKETS = (1, 2, 3, 5, 10, 99)             # upper edges: 1, 2, 3, 4-5, 6-10, 11+
FP_BUCKETS = (40, 50, 60, 70, 80, 100)         # punts by yards from the goal line: <=40, 41-50, ...
STATE = ["season", "score_differential", "game_seconds_remaining", "half_seconds_remaining", "half2", "ot", "down",
         "ydstogo", "yardline_100", "posteam_timeouts_remaining", "defteam_timeouts_remaining", "home", "recv_2h",
         "strength", "indoors", "wind"]


def _bucket(x, edges) -> np.ndarray:
    return np.searchsorted(np.asarray(edges), np.asarray(x, float), side="left").clip(max=len(edges) - 1)


def _nodes(values: pd.Series, by: np.ndarray, n_buckets: int) -> np.ndarray:
    """(bucket, quantile) table of `values`; a bucket with too few rows takes the pooled quantiles."""
    pooled = np.quantile(values, QUANTILES)
    out = np.tile(pooled, (n_buckets, 1))
    for b in range(n_buckets):
        v = values[by == b]
        if len(v) >= 50:
            out[b] = np.quantile(v, QUANTILES)
    return out


@dataclass
class Engine:
    wp: object                   # win-probability model (wp.py: trees on a logistic baseline with field position)
    conv: pd.Series              # conversion rate by yards to go (the registered table)
    fg: object                   # field-goal model (fg.py: distance and its square, last three seasons)
    fg_max: float                # longest kick offered, as yards from the goal line (FG_SUPPORT)
    punt: object                 # net punt model (median)
    extra: np.ndarray            # converted: yards gained beyond the line to gain, (ytg bucket, quantile)
    short: np.ndarray            # failed: yards gained (below the line to gain), (ytg bucket, quantile)
    punt_resid: np.ndarray       # net minus the model's prediction, (field-position bucket, quantile)
    p_block: float
    fg_snap: float               # kick distance minus yards from the goal line
    runoff: dict[str, float]     # seconds off the clock by choice
    kick_start: float            # receiving team's yards from the goal line after a kickoff
    seasons: tuple[int, ...]

    # ---- valuing a state ------------------------------------------------------------------------------------
    def value(self, st: pd.DataFrame) -> np.ndarray:
        """Win probability of the team with the ball in each state (1st down unless the row says otherwise)."""
        st = st.copy()
        late = st["half_seconds_remaining"] <= 0
        over = late & ((st["half2"] == 1) | (st["ot"] == 1))
        half = late & ~over
        # Halftime: the 2nd-half kickoff. The team that receives it has the ball; flip the rows where that's the
        # other team.
        flip = (half & (st["recv_2h"] == 0)).to_numpy()
        st.loc[half, ["game_seconds_remaining", "half_seconds_remaining", "half2", "down", "ydstogo", "yardline_100",
                      "posteam_timeouts_remaining", "defteam_timeouts_remaining"]] = [1800, 1800, 1, 1, 10, self.kick_start, 3, 3]
        st.loc[flip, "score_differential"] *= -1
        st.loc[flip, "strength"] *= -1
        st.loc[flip, ["home", "recv_2h"]] = 1 - st.loc[flip, ["home", "recv_2h"]]
        st[["game_seconds_remaining", "half_seconds_remaining"]] = st[["game_seconds_remaining", "half_seconds_remaining"]].clip(lower=0)
        p = M.predict_wp(self.wp, st)
        p = np.where(flip, 1 - p, p)
        sd = st["score_differential"].to_numpy(float)
        return np.where(over.to_numpy(), np.where(sd > 0, 1.0, np.where(sd < 0, 0.0, 0.5)), p)

    def _after(self, s: pd.DataFrame, yardline, points, keep, seconds) -> np.ndarray:
        """The original offense's win probability once the ball is `yardline` yards from the goal line of whoever
        has it: the offense itself (keep) or the opponent, after the offense scored `points`."""
        st = s[STATE].copy()
        keep = np.asarray(keep, bool)
        yl = np.clip(np.asarray(yardline, float), 1, 99)
        st["score_differential"] = np.where(keep, 1, -1) * (s["score_differential"].to_numpy(float) + points)
        st["down"], st["yardline_100"], st["ydstogo"] = 1.0, yl, np.minimum(10.0, yl)
        mine, theirs = s["posteam_timeouts_remaining"].to_numpy(float), s["defteam_timeouts_remaining"].to_numpy(float)
        st["posteam_timeouts_remaining"], st["defteam_timeouts_remaining"] = np.where(keep, mine, theirs), np.where(keep, theirs, mine)
        for c in ("home", "recv_2h"):
            st[c] = np.where(keep, s[c], 1 - s[c])
        st["strength"] = np.where(keep, 1, -1) * s["strength"].to_numpy(float)
        st["game_seconds_remaining"] = s["game_seconds_remaining"].to_numpy(float) - seconds
        st["half_seconds_remaining"] = s["half_seconds_remaining"].to_numpy(float) - seconds
        v = self.value(st)
        return np.where(keep, v, 1 - v)

    # ---- the three choices ----------------------------------------------------------------------------------
    def go(self, s: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(win probability if it goes for it, conversion probability)."""
        p = M.predict_conv_baseline(self.conv, s)
        y, ytg = s["yardline_100"].to_numpy(float), s["ydstogo"].to_numpy(float)
        b = _bucket(ytg, YTG_BUCKETS)
        won, lost = np.zeros(len(s)), np.zeros(len(s))
        for q in range(len(QUANTILES)):
            spot = y - (ytg + self.extra[b, q])                      # converted: new line of scrimmage
            td = spot <= 0
            won += np.where(td, self._after(s, self.kick_start, TD_POINTS, np.zeros(len(s), bool), self.runoff["go"]),
                            self._after(s, spot, 0, np.ones(len(s), bool), self.runoff["go"]))
            gain = np.minimum(self.short[b, q], ytg - 1)             # failed: short of the line to gain
            lost += self._after(s, 100 - (y - gain), 0, np.zeros(len(s), bool), self.runoff["go"])
        n = len(QUANTILES)
        return p * won / n + (1 - p) * lost / n, p

    def kick(self, s: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(win probability if it kicks, make probability); NaN beyond the longest kick offered."""
        y = s["yardline_100"].to_numpy(float)
        p = F.predict(self.fg, y + self.fg_snap)
        no = np.zeros(len(s), bool)
        good = self._after(s, self.kick_start, FG_POINTS, no, self.runoff["fg"])
        spot = np.maximum(y + self.fg_snap - 10, TOUCHBACK)          # the spot of the kick, or the 20
        miss = self._after(s, 100 - spot, 0, no, self.runoff["fg"])
        return np.where(y <= self.fg_max, p * good + (1 - p) * miss, np.nan), p

    def punt_wp(self, s: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(win probability if it punts, expected net yards); NaN inside PUNT_MIN_YARDLINE."""
        y = s["yardline_100"].to_numpy(float)
        pred = M.predict_punt(self.punt, s)
        b = _bucket(y, FP_BUCKETS)
        no = np.zeros(len(s), bool)
        total, net_sum = np.zeros(len(s)), np.zeros(len(s))
        for q in range(len(QUANTILES)):
            land = y - (pred + self.punt_resid[b, q])                # yards from the goal line where it ends up
            net = np.where(land <= 0, y - TOUCHBACK, y - np.maximum(land, 1))
            net_sum += net
            total += self._after(s, 100 - (y - net), 0, no, self.runoff["punt"])
        blocked = self._after(s, 100 - (y + BLOCK_YARDS), 0, no, self.runoff["punt"])
        n = len(QUANTILES)
        wp = (1 - self.p_block) * total / n + self.p_block * blocked
        return np.where(y >= PUNT_MIN_YARDLINE, wp, np.nan), net_sum / n

    def decide(self, s: pd.DataFrame) -> pd.DataFrame:
        """One row per situation: each choice's win probability, the best one, and go minus the best kick."""
        s = s.reset_index(drop=True)
        go, p_conv = self.go(s)
        fg, p_fg = self.kick(s)
        punt, net = self.punt_wp(s)
        out = pd.DataFrame({"wp_go": go, "wp_fg": fg, "wp_punt": punt, "p_convert": p_conv, "p_fg": p_fg, "punt_net": net})
        kick = np.fmax(out["wp_fg"], out["wp_punt"])                 # fmax: ignore the option that isn't offered
        out["wp_kick"] = kick
        out["best_kick"] = np.where(out["wp_fg"].fillna(-1) >= out["wp_punt"].fillna(-1), "fg", "punt")
        out["margin"] = out["wp_go"] - kick                          # > 0: going for it is better
        out["best"] = np.where(out["margin"] > 0, "go", out["best_kick"])
        return out

    def chosen(self, s: pd.DataFrame, r: pd.DataFrame) -> np.ndarray:
        """Win probability of what the team actually did (`s.choice`), including kicks and punts from spots where
        the engine doesn't offer that option."""
        s = s.reset_index(drop=True)
        y = s["yardline_100"].to_numpy(float)
        far, near = s.assign(yardline_100=np.minimum(y, self.fg_max)), s.assign(yardline_100=np.maximum(y, PUNT_MIN_YARDLINE))
        fg = np.where(np.isnan(r["wp_fg"]), self.kick(far)[0], r["wp_fg"])          # valued at the nearest offered spot
        punt = np.where(np.isnan(r["wp_punt"]), self.punt_wp(near)[0], r["wp_punt"])
        return np.select([s["choice"] == "go", s["choice"] == "fg"], [r["wp_go"], fg], punt)


def plays(loader, seasons, as_of) -> pd.DataFrame:
    """data.load_plays plus yards gained per play (not a model input; only the engine's outcome tables read it)."""
    seasons = list(seasons)
    y = pd.concat([loader.load(f"pbp_{s}", as_of, columns=["game_id", "play_id", "yards_gained"]) for s in seasons],
                  ignore_index=True)
    return load_plays(loader, seasons, as_of).merge(y, on=["game_id", "play_id"], how="left", validate="one_to_one")


def fit(d: pd.DataFrame) -> Engine:
    """Everything the engine needs, from the training plays `d` (plays())."""
    att = d[d["converted"].notna() & d["yards_gained"].notna()]
    ok, bad = att[att["converted"] == 1], att[att["converted"] == 0]
    fgs, punts = d[d["fg_made"].notna()], d[d["punt_net"].notna()]
    pm = M.fit_punt(punts)
    resid = punts["punt_net"].to_numpy(float) - M.predict_punt(pm, punts)
    all_punts = d[d["play_type"] == "punt"]
    g = d.sort_values(["game_id", "play_id"])
    nxt = g.groupby("game_id")[["game_seconds_remaining", "yardline_100", "down"]].shift(-1)
    dt = (g["game_seconds_remaining"] - nxt["game_seconds_remaining"]).clip(lower=0)
    last = g["season"] == g["season"].max()
    start = nxt.loc[(g["kickoff_attempt"] == 1) & (nxt["down"] == 1) & last, "yardline_100"]
    recent = fgs[fgs["season"] > fgs["season"].max() - F.WINDOW]
    snap = float((fgs["kick_distance"] - fgs["yardline_100"]).median())
    return Engine(
        wp=W.fit(scrimmage(d)), conv=M.fit_conv_baseline(d[d["converted"].notna()]), fg=F.fit(fgs), punt=pm,
        fg_max=float(np.floor(recent["kick_distance"].quantile(FG_SUPPORT)) - snap),
        extra=_nodes((ok["yards_gained"] - ok["ydstogo"]).clip(lower=0), _bucket(ok["ydstogo"], YTG_BUCKETS), len(YTG_BUCKETS)),
        short=_nodes(bad["yards_gained"], _bucket(bad["ydstogo"], YTG_BUCKETS), len(YTG_BUCKETS)),
        punt_resid=_nodes(pd.Series(resid), _bucket(punts["yardline_100"], FP_BUCKETS), len(FP_BUCKETS)),
        p_block=float((all_punts["punt_blocked"] == 1).mean()),
        fg_snap=snap,
        runoff={"go": float(dt[g["converted"].notna()].median()), "fg": float(dt[g["fg_made"].notna()].median()),
                "punt": float(dt[g["punt_net"].notna()].median())},
        kick_start=float(start.mean()), seasons=tuple(sorted(int(x) for x in d["season"].unique())))


def situations(d: pd.DataFrame) -> pd.DataFrame:
    """Every real 4th-down decision in `d`: a go, a field-goal try, or a punt (blocked ones included), with the
    state the win-probability model needs. `choice` is what the team did."""
    choice = np.select([d["converted"].notna(), d["play_type"] == "field_goal", d["play_type"] == "punt"], ["go", "fg", "punt"], "")
    keep = (d["down"] == 4) & (choice != "") & d["win"].notna() & d["game_seconds_remaining"].notna() & d["strength"].notna() \
        & d["yardline_100"].notna() & (d["penalty"] != 1)
    return d[keep].assign(choice=choice[keep.to_numpy()]).reset_index(drop=True)


def bootstrap(d: pd.DataFrame, s: pd.DataFrame, n: int, seed: int = M.SEED) -> np.ndarray:
    """(n, situations) margins (go minus the best kick) from engines refit on games resampled with replacement."""
    rng = np.random.default_rng(seed)
    idx = d.groupby("game_id").indices
    games = np.array(sorted(idx))
    out = np.empty((n, len(s)))
    for i in range(n):
        pick = rng.choice(games, len(games), replace=True)
        b = d.iloc[np.concatenate([idx[gm] for gm in pick])].copy()
        # A game drawn twice keeps both copies under new ids, so the per-game steps (clock differences, the
        # kickoff start) treat them as separate games. Team ratings are already on the rows.
        b["game_id"] = np.repeat(np.arange(len(pick)), [len(idx[gm]) for gm in pick]).astype(str) + ":" + b["game_id"].astype(str)
        out[i] = fit(b).decide(s)["margin"].to_numpy()
    return out


def engine_for(loader, season: int, first: int, as_of) -> tuple[Engine, pd.DataFrame]:
    """The engine that decides season `season`: fit on seasons first..season-1 as of `as_of` (walk-forward, like the
    backtest folds). Returns it with its training plays, for the bootstrap."""
    d = plays(loader, range(first, season), as_of)
    return fit(d), d
