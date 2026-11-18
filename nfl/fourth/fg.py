"""NFL field-goal make probability, 2.0.0 (F4-V2): logistic regression on kick distance and its square, fit on the
last WINDOW seasons before the one being predicted.

Why: 1.0.0 (linear in distance, every season since 2016) failed its calibration gate on 2020-2025. It predicted too few
makes from long range (50-54 yards: 65% predicted, 73% made) and fell further behind each season, because long kicks
kept improving and the fit averaged over older seasons. The square lets the curve bend at long range; the short
window keeps the fit on recent kicking. Chosen by Matt (Nov 18, 2026) after that backtest was read, so 2020-2025 are
not a clean test of this version: the 2026 season is.

    python -m nfl.fourth.fg        -> nfl/fourth/output/development_fg.json (development seasons 2016-2019 only)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

WINDOW = 3                     # seasons of kicks the fit uses
CENTER, SCALE = 40.0, 100.0    # the square term is ((distance - 40) ** 2) / 100: about 1 at 30 and 50 yards


def _x(distance) -> np.ndarray:
    d = np.asarray(distance, float)
    return np.column_stack([d, (d - CENTER) ** 2 / SCALE])


def fit(kicks: pd.DataFrame, window: int | None = WINDOW):
    """`kicks`: field-goal attempts (season, kick_distance, fg_made) of every training season; the last `window`
    seasons present are used (None: all of them)."""
    if window:
        kicks = kicks[kicks["season"] > kicks["season"].max() - window]
    # Effectively unpenalized (two terms, thousands of kicks); lbfgs is deterministic.
    return LogisticRegression(C=1e6, max_iter=5000).fit(_x(kicks["kick_distance"]), kicks["fg_made"].to_numpy(float))


def predict(m, distance) -> np.ndarray:
    return m.predict_proba(_x(distance))[:, 1]


def main() -> None:
    """Development check on the seasons the F4 models were developed on (validation 2018 and 2019): this version
    next to 1.0.0's distance-only model. With two and three training seasons the window isn't binding there; the
    era effect it is for shows only in later seasons."""
    from backtest import metrics
    from backtest.asof import Loader
    from nfl.fourth import models as M
    from nfl.fourth.data import load_plays

    d = load_plays(Loader(), (2016, 2017, 2018, 2019), "2020-09-01")
    k = d[d["fg_made"].notna()]
    val = {2018: (2016, 2017), 2019: (2016, 2017, 2018)}
    new, old = [], []
    for v, tr in val.items():
        a, b = k[k["season"].isin(tr)], k[k["season"] == v]
        new.append(metrics.brier(predict(fit(a), b["kick_distance"]), b["fg_made"]))
        old.append(metrics.brier(M.predict_fg_baseline(M.fit_fg_baseline(a), b), b["fg_made"]))
    res = {"seasons": [2016, 2017, 2018, 2019], "validation": {str(v): list(tr) for v, tr in val.items()}, "n": int(len(k)),
           "window": WINDOW, "scores": {"distance_squared_last3": round(float(np.mean(new)), 5),
                                        "distance_only_1_0_0": round(float(np.mean(old)), 5)}}
    out = Path(__file__).resolve().parent / "output" / "development_fg.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(res["scores"]))


if __name__ == "__main__":
    main()
