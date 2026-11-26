"""What the NFL team rating expects from a game between two teams (NFL-MATCHUP): the inputs the matchup page needs.

The published forecast (adjusted.py) is home field plus a slope times the gap in two teams' ridge ratings (net EPA per
play, penalty chosen on next-week margins). This writes, with the published model's own functions and settings:
  - the line: points for home field and points per unit of rating gap, fit on every next-week game of the complete
    seasons, and the spread of actual margins around it
  - each team's rating through the latest week of the current season
adjusted.py is not changed, so the model's version is not either.

A margin and a win probability only: the rating does not model points scored, so there is no expected score or total.

    python nfl/team_efficiency/matchup.py [--through-week N]      # after adjusted.py (reads its penalty), ~30 seconds

Writes nfl/team_efficiency/output/matchup.json. Deterministic (closed form).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adjusted as A  # noqa: E402

from football.common.adjust import ridge_ratings  # noqa: E402

Z80 = 1.2815515655446004


def expected(a: dict, b: dict, site: str, line: dict) -> dict:
    """Margin (team A less team B), its 80% range and A's chance to win. `a`, `b`: {'net': rating}; `site`: 'a' (A at
    home), 'b' or 'n' (neutral). The same arithmetic as the page's."""
    hc = {"a": 1.0, "b": -1.0, "n": 0.0}[site]
    margin = line["home_points"] * hc + line["points_per_net"] * (a["net"] - b["net"])
    sd = line["margin_sd"]
    return {"margin": margin, "lo80": margin - Z80 * sd, "hi80": margin + Z80 * sd,
            "p_a": 0.5 * (1 + math.erf(margin / (sd * math.sqrt(2))))}


def build(through_week: int | None = None) -> dict:
    lam = json.loads((A.OUT / "adjustment.json").read_text(encoding="utf-8"))["lambda"]
    plays, g = A.variant(A.load_plays(), A.SEARCH_GT, "all"), A.games()
    p = A.predictions(plays, g, lam, A.CFG.past_seasons)
    X = p[["home_site", "d_ridge"]].to_numpy(float)
    y = p["margin"].to_numpy(float)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ coef
    cur = plays[plays["season"] == A.CFG.season]
    week = int(cur["week"].max()) if through_week is None else int(through_week)
    r = ridge_ratings(cur[cur["week"] <= week], lam).set_index("team")
    teams = [{"id": t, "net": round(float(r.loc[t, "off_ridge"] - r.loc[t, "def_ridge"]), 4), "off": round(float(r.loc[t, "off_ridge"]), 4),
              "def": round(float(r.loc[t, "def_ridge"]), 4)} for t in sorted(r.index)]
    return {"season": A.CFG.season, "through_week": week, "lambda": lam,
            "line": {"home_points": round(float(coef[0]), 3), "points_per_net": round(float(coef[1]), 2), "margin_sd": round(float(resid.std()), 2)},
            "fit": {"seasons": [int(s) for s in sorted(p["season"].unique())], "games": int(len(p)), "mae_in_sample": round(float(np.abs(resid).mean()), 3),
                    "mae_out_of_sample": round(A.loso_mae(p, ["d_ridge"]), 3), "mae_home_only": round(A.loso_mae(p, []), 3)},
            "teams": teams}


def site_data(res: dict, teams: list[dict]) -> tuple[dict, dict]:
    """(site/data/nfl/matchup.json, prose numbers) from this module's output and the site's team list."""
    names = {t["id"]: t for t in teams}
    rows = sorted(({"id": t["id"], "name": names[t["id"]]["name"], "net": t["net"], "off": t["off"], "def": t["def"]} for t in res["teams"]),
                  key=lambda t: (-t["net"], t["id"]))
    for i, t in enumerate(rows, 1):
        t["rank"] = i
    line, fit = res["line"], res["fit"]
    spread = rows[0]["net"] - rows[-1]["net"]
    data = {"season": res["season"], "through_week": res["through_week"], "line": line, "teams": sorted(rows, key=lambda t: t["name"])}
    N = {"mu_week": str(res["through_week"]), "mu_home_points": f"{line['home_points']:.1f}", "mu_sd": f"{line['margin_sd']:.1f}",
         "mu_points_per_hundredth": f"{line['points_per_net'] / 100:.1f}", "mu_games": f"{fit['games']:,}",
         "mu_seasons": f"{fit['seasons'][0]}–{fit['seasons'][-1]}", "mu_mae": f"{fit['mae_out_of_sample']:.1f}",
         "mu_mae_home": f"{fit['mae_home_only']:.1f}", "mu_best": rows[0]["name"], "mu_worst": rows[-1]["name"],
         "mu_best_vs_worst": f"{line['points_per_net'] * spread:.1f}"}
    return data, N


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--through-week", type=int, default=None)
    res = build(ap.parse_args().through_week)
    (A.OUT / "matchup.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    t = sorted(res["teams"], key=lambda x: -x["net"])
    print(f"{res['season']} through week {res['through_week']}: {res['line']}, {res['fit']}; best {t[0]['id']} {t[0]['net']:+.3f}, worst {t[-1]['id']} {t[-1]['net']:+.3f}")


if __name__ == "__main__":
    main()
