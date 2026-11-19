"""Where in a game the win-probability model can be trusted (F4): its calibration by time remaining, on seasons it
hadn't seen. Season t is predicted by the model fit on FIRST..t-1, as in the backtest. The fourth-down page judges
no decision in the last summary.LATE_SECONDS of regulation or in overtime because of what this shows.

    python -m nfl.fourth.late_check      -> nfl/fourth/output/late_calibration.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import metrics
from backtest.asof import Loader
from nfl.fourth import models as M
from nfl.fourth.data import FIRST_SEASON, load_plays, scrimmage
from nfl.fourth.decisions import FIRST_DECIDED
from nfl.fourth.summary import LATE_SECONDS
from sparvi.config import nfl

OUT = Path(__file__).resolve().parent / "output"
BUCKETS = ["rest of the game", "last 2:00 of the first half", "4th quarter, 2:00 to 1:00", "4th quarter, last 1:00", "overtime"]


def main() -> None:
    loader, parts = Loader(), []
    seasons = list(range(FIRST_DECIDED, nfl().season))            # complete seasons only
    for t in seasons:
        tr = scrimmage(load_plays(loader, range(FIRST_SEASON, t), f"{t}-09-01"))
        te = scrimmage(load_plays(loader, [t], f"{t + 1}-03-01"))
        parts.append(te.assign(p=M.predict_wp(M.fit_wp(tr), te)))
    a = pd.concat(parts, ignore_index=True)
    hs, h2 = a["half_seconds_remaining"], a["half2"] == 1
    a["bucket"] = np.select([a["ot"] == 1, h2 & (hs <= 60), h2 & (hs <= LATE_SECONDS), hs <= LATE_SECONDS],
                            [BUCKETS[4], BUCKETS[3], BUCKETS[2], BUCKETS[1]], BUCKETS[0])
    rows = []
    for b in BUCKETS:
        d = a[a["bucket"] == b]
        rows.append({"bucket": b, "plays": int(len(d)), "ece": round(metrics.ece(d["p"].to_numpy(), d["win"].to_numpy()), 4),
                     "brier": round(metrics.brier(d["p"].to_numpy(), d["win"].to_numpy()), 4)})
    rest = a[a["bucket"] == BUCKETS[0]]
    late = a[a["bucket"] != BUCKETS[0]]
    # A situation that should be near certain: the offense leads by 1-8 with 40 seconds or less.
    sure = a[h2 & (hs <= 40) & a["score_differential"].between(1, 8)]
    res = {"seasons": seasons, "late_seconds": LATE_SECONDS, "buckets": rows,
           "ece_judged": round(metrics.ece(rest["p"].to_numpy(), rest["win"].to_numpy()), 4),
           "ece_late": round(metrics.ece(late["p"].to_numpy(), late["win"].to_numpy()), 4),
           "leading_late": {"plays": int(len(sure)), "predicted": round(float(sure["p"].mean()), 4), "won": round(float(sure["win"].mean()), 4)}}
    (OUT / "late_calibration.json").write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
