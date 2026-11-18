"""CFB-CAROUSEL: what a head-coaching change does to a team in year 1.

For every FBS team-season t (2016-2025) with power ratings in t-2, t-1, and t: the expected end-of-season power rating
from the team's last two seasons, fitted on teams that KEPT their head coach (ordinary least squares, leave-one-season-
out so no season predicts itself). A coaching change's year-1 effect = actual minus expected. Splits:
  level        Power vs. Group of Five (conference in season t)
  coach        first-time head coach vs. one who was an FBS head coach before (coaches data start in 2014, so "first
               time" means no FBS head-coaching job since 2014)
  last_year    the team's season before the change: bottom third of FBS by power rating, or not
  departure    the previous coach left for another FBS head-coaching job (he's a head coach elsewhere the next season),
               or not (fired, retired, or moved to a job outside FBS head coaching)
And: does a moving head coach bring his pass-heaviness with him (correlation of the new team's style with his old
team's, from the committed Phase 1 output, cfb/fit/output/explore.json)? 2026's first-year coaches are shown as a season in progress (their
current rating against the same expectation).

Head coach of a season = the coach with the most games that season. Interim seasons count toward the coach who
coached most games. Aggregates only -> cfb/carousel/output/results.json.

    python cfb/carousel/analysis.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cfb.fit import labels as LB
from sparvi.config import RAW, ROOT, cfb

OUT = ROOT / "cfb" / "carousel" / "output"
FIRST = 2016
N_BOOT, SEED = 2000, 7


def ratings() -> pd.DataFrame:
    """Each team's power rating at the end of each season (its last week), plus the current season's latest."""
    p = pd.read_csv(ROOT / "cfb" / "adjusted_ratings" / "output" / "power_team_week.csv")
    last = p.sort_values("week").groupby(["season", "team"]).tail(1)
    return last[["season", "team", "power", "week"]]


def coaches() -> pd.DataFrame:
    c = pd.read_parquet(RAW / "cfb" / "coaches.parquet")
    return c.sort_values("games").groupby(["season", "school"]).tail(1)[["season", "school", "coach_id", "coach", "hire_date"]]


def frame() -> pd.DataFrame:
    r = ratings().set_index(["season", "team"])["power"]
    h = coaches().set_index(["season", "school"])
    seasons = sorted({s for s, _ in r.index})
    rows = []
    for t in [s for s in seasons if s >= FIRST]:
        conf = dict(pd.read_parquet(RAW / "cfb" / f"teams_{t}.parquet")[["school", "conference"]].values)
        for team in sorted(conf):
            if not all((t - k, team) in r.index for k in (0, 1, 2)):
                continue
            now, prev = (t, team), (t - 1, team)
            if now not in h.index or prev not in h.index:
                continue
            c_now, c_prev = h.loc[now], h.loc[prev]
            # Was this head coach an FBS head coach anywhere before season t (since 2014)?
            before = h[(h.index.get_level_values(0) < t) & (h["coach_id"] == c_now["coach_id"])]
            moved_from = [sc for (s, sc) in before.index if s == t - 1 and sc != team]
            # Did the previous coach leave for another FBS head-coaching job (he's a head coach elsewhere in season t)?
            now_jobs = h[(h.index.get_level_values(0) == t) & (h["coach_id"] == c_prev["coach_id"])]
            left_for = [sc for (s, sc) in now_jobs.index if sc != team]
            rows.append({"season": t, "team": team, "level": LB.tier(team, t, conf), "power": r[now],
                         "power_1": r[prev], "power_2": r[(t - 2, team)], "change": bool(c_now["coach_id"] != c_prev["coach_id"]),
                         "coach": c_now["coach"], "prev_coach": c_prev["coach"], "experienced": bool(len(before)),
                         "moved_from": moved_from[0] if moved_from else None, "left_for": left_for[0] if left_for else None,
                         "in_progress": t == cfb().season})
    d = pd.DataFrame(rows)
    # The season before, among FBS teams that season: bottom third by power rating?
    third = d.groupby("season")["power_1"].transform(lambda v: v.quantile(1 / 3))
    d["bad_last_year"] = d["power_1"] <= third
    return d


def expected(d: pd.DataFrame) -> np.ndarray:
    """Leave-one-season-out OLS on teams that kept their coach (completed seasons only)."""
    pred = np.full(len(d), np.nan)
    done = d[~d["in_progress"]]
    for s in d["season"].unique():
        tr = done[(done["season"] != s) & ~done["change"]]
        A = np.c_[np.ones(len(tr)), tr[["power_1", "power_2"]].to_numpy()]
        beta = np.linalg.lstsq(A, tr["power"].to_numpy(), rcond=None)[0]
        m = (d["season"] == s).to_numpy()
        pred[m] = np.c_[np.ones(m.sum()), d.loc[m, ["power_1", "power_2"]].to_numpy()] @ beta
    return pred


def _boot_mean(x: np.ndarray, teams: np.ndarray) -> dict:
    """Mean and 90% interval, resampling teams (a team's changes move together)."""
    rng = np.random.default_rng(SEED)
    uniq = np.unique(teams)
    pos = {t: np.flatnonzero(teams == t) for t in uniq}
    draws = [x[np.concatenate([pos[t] for t in rng.choice(uniq, len(uniq))])].mean() for _ in range(N_BOOT)]
    return {"n": int(len(x)), "mean": round(float(x.mean()), 2),
            "ci90": [round(float(np.percentile(draws, 5)), 2), round(float(np.percentile(draws, 95)), 2)],
            "share_beat": round(float((x > 0).mean()), 3)}


def main() -> None:
    d = frame()
    d["expected"] = expected(d)
    d["effect"] = d["power"] - d["expected"]
    done = d[~d["in_progress"]]
    ch, kept = done[done["change"]], done[~done["change"]]
    summary = {"all": _boot_mean(ch["effect"].to_numpy(), ch["team"].to_numpy()),
               "kept": _boot_mean(kept["effect"].to_numpy(), kept["team"].to_numpy())}
    for name, mask in (("power", ch["level"] == "Power"), ("g5", ch["level"] == "G5"),
                       ("first_time", ~ch["experienced"]), ("experienced", ch["experienced"]),
                       ("bad_last_year", ch["bad_last_year"]), ("not_bad_last_year", ~ch["bad_last_year"]),
                       ("left_for_job", ch["left_for"].notna()), ("did_not_leave_for_job", ch["left_for"].isna())):
        q = ch[mask.to_numpy()]
        summary[name] = _boot_mean(q["effect"].to_numpy(), q["team"].to_numpy())
    # Style carryover from the committed Phase 1 output (computed on the F1 team styles, which aren't committed).
    carry = json.loads((ROOT / "cfb" / "fit" / "output" / "explore.json").read_text(encoding="utf-8"))["coach_carryover"]
    sd_kept = float(kept["effect"].std())
    rows = [{"season": int(r.season), "team": r.team, "level": r.level, "coach": r.coach, "prev_coach": r.prev_coach,
             "experienced": bool(r.experienced), "moved_from": r.moved_from if isinstance(r.moved_from, str) else None,
             "left_for": r.left_for if isinstance(r.left_for, str) else None, "bad_last_year": bool(r.bad_last_year),
             "expected": round(float(r.expected), 2), "actual": round(float(r.power), 2), "effect": round(float(r.effect), 2),
             "in_progress": bool(r.in_progress)} for r in d[d["change"]].itertuples()]
    res = {"seasons": [int(done["season"].min()), int(done["season"].max())], "current_season": int(cfb().season),
           "n_team_seasons": int(len(done)), "n_changes": int(len(ch)), "kept_effect_sd": round(sd_kept, 2),
           "summary": summary, "style_carryover": carry, "changes": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in res.items() if k != "changes"}, indent=1))
    cur = [r for r in rows if r["in_progress"]]
    print(f"{len(cur)} first-year head coaches in {res['current_season']} (in progress):")
    for r in sorted(cur, key=lambda r: -r["effect"])[:8]:
        print("  ", r["team"], r["coach"], r["effect"])


if __name__ == "__main__":
    main()
