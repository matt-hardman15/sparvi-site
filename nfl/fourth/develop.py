"""Development for the fourth-down models (F4) on the 2016-2019 seasons only: a small grid per model, scored on two
validation seasons (2018 trained on 2016-2017, 2019 on 2016-2018), next to each baseline. The settings chosen here
are fixed in models.py before the backtest (2020-2025) runs; later seasons are never read.

    uv run python nfl/fourth/develop.py      -> nfl/fourth/output/development.json
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np

from backtest import metrics
from backtest.asof import Loader
from nfl.fourth import models as M
from nfl.fourth.data import load_plays, scrimmage

OUT = Path(__file__).resolve().parent / "output"
DEV = (2016, 2017, 2018, 2019)
VAL = {2018: (2016, 2017), 2019: (2016, 2017, 2018)}


def main() -> None:
    loader = Loader()
    d = load_plays(loader, DEV, "2020-09-01")
    assert d["season"].max() <= 2019 and loader.max_available_at() <= np.datetime64("2020-09-01")
    res = {"seasons": list(DEV), "validation": {str(k): list(v) for k, v in VAL.items()}, "models": {}}

    def run(name, rows, label, fits, metric):
        out = {}
        for tag, (fit, pred) in fits.items():
            scores = []
            for v, tr in VAL.items():
                a, b = rows[rows["season"].isin(tr)], rows[rows["season"] == v]
                scores.append(metrics.score(metric, pred(fit(a), b), b[label]))
            out[tag] = round(float(np.mean(scores)), 5)
            print(f"{name:<11} {tag:<22} {metric} {out[tag]}")
        res["models"][name] = {"metric": metric, "n": int(len(rows)), "scores": out}

    s = scrimmage(d)
    wp = {f"lgb_r{r}_l{lv}": (partial(M.fit_wp, rounds=r, params={"num_leaves": lv}), M.predict_wp)
          for r, lv in ((150, 7), (150, 15), (300, 7), (300, 15), (300, 31), (600, 15))}
    wp["baseline"] = (M.fit_wp_baseline, M.predict_wp_baseline)
    run("win_prob", s, "win", wp, "brier")

    # Conversion variants train on all plays of the training seasons (fit_conv picks its rows) and are scored on
    # 4th-down attempts of the validation season.
    out = {}
    for fs in M.CONV_SETS:
        for C in (0.1, 1.0):
            sc = [metrics.brier(M.predict_conv(M.fit_conv(d[d["season"].isin(tr)], C=C, features=fs),
                                               d[(d["season"] == v) & d["converted"].notna()]),
                                d.loc[(d["season"] == v) & d["converted"].notna(), "converted"]) for v, tr in VAL.items()]
            out[f"logit_{fs}_C{str(C).replace('.', 'p')}"] = round(float(np.mean(sc)), 5)
    c = d[d["converted"].notna()]
    sc = [metrics.brier(M.predict_conv_baseline(M.fit_conv_baseline(c[c["season"].isin(tr)]), c[c["season"] == v]),
                        c.loc[c["season"] == v, "converted"]) for v, tr in VAL.items()]
    out["baseline"] = round(float(np.mean(sc)), 5)
    for k, v in out.items():
        print(f"conversion  {k:<26} brier {v}")
    res["models"]["conversion"] = {"metric": "brier", "n": int(len(c)), "scores": out}

    f = d[d["fg_made"].notna()]
    fg = {f"logit_{fs}_C{str(C).replace('.', 'p')}": (partial(M.fit_fg, C=C, features=fs), M.predict_fg)
          for fs in M.FG_SETS for C in (0.1, 1.0, 10.0)}
    fg["baseline"] = (M.fit_fg_baseline, M.predict_fg_baseline)
    run("fg_make", f, "fg_made", fg, "brier")

    p = d[d["punt_net"].notna()]
    pn = {f"lgb_r{r}": (partial(M.fit_punt, rounds=r), M.predict_punt) for r in (75, 150, 300)}
    pn["baseline"] = (M.fit_punt_baseline, M.predict_punt_baseline)
    run("punt_net", p, "punt_net", pn, "mae")

    OUT.mkdir(exist_ok=True)
    (OUT / "development.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
