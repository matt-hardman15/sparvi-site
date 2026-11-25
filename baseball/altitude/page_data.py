"""Site data and prose numbers for the altitude article (site/baseball/triple-a-altitude/), from the by-hand analysis
(baseball/altitude/output/results.json) and the published grades.

A pitcher's park-adjusted Stuff+ is his PUBLISHED Stuff+ plus the change the analysis measured (adjusted less
as-recorded, both graded on the same pitches), so the article always matches the board and the leaderboard.
Called by baseball/export_site_data.py, which writes what this returns.
"""
from __future__ import annotations

import pandas as pd

from sparvi.io import StaleProse

TOP = 25


def _sgn(v: float, d: int = 1) -> str:
    return f"{v:+.{d}f}".replace("-", "−")


def _pct(v: float) -> str:
    return f"{100 * v:.0f}%"


def build(res: dict, published: list[dict], board: list[dict], clubs: dict[str, str]) -> tuple[dict, dict]:
    """(altitude.json, prose numbers). `published`: the leaderboard's Triple-A pitcher rows; `board`: the Rule 5 board's
    pitchers; `clubs`: Triple-A club code -> name."""
    alt = set(res["altitude_parks"])
    pub = {r["id"]: r for r in published if r["level"] == "AAA"}
    on_board = {r["player_id"]: r for r in board}
    rows = []
    for x in res["pitchers"]:
        p = pub.get(x["player_id"])
        if p is None or p.get("stuff_plus") is None:
            continue
        b = on_board.get(x["player_id"])
        rows.append({"id": x["player_id"], "name": p["name"], "team": p["team"], "role": p["role"], "pitches": p["pitches"],
                     "stuff_plus": p["stuff_plus"], "stuff_adj": round(p["stuff_plus"] + x["delta"], 1), "delta": x["delta"],
                     "alt_share": x["altitude_share"], "alt_team": p["team"] in alt, "r5": b is not None, "org": b["org"] if b else None})
    d = pd.DataFrame(rows)
    parks = [{"park": p["park"], "club": clubs.get(p["park"], p["park"]), "altitude": p["altitude"], "factor": round(p["factor"], 3),
              "lo": round(p["factor_ci90"][0], 3), "hi": round(p["factor_ci90"][1], 3), "points": round(p["points"], 1),
              "fit_pitchers": p["fit_pitchers"]} for p in res["parks"]]
    v = res["validation"]
    data = {"season": res["season"], "fit_seasons": res["fit_seasons"], "min_pitches": res["min_pitches"], "altitude_parks": sorted(alt),
            "extension_offset_in": round(12 * res["extension_offset_ft"], 1), "parks": parks, "pitchers": rows,
            "validation": {"pitchers": v["pitchers"], **{k: {m: round(v[k][m], 1) for m in ("gap", "altitude", "others")} for k in ("as_recorded", "adjusted")}}}

    hi_parks = [p for p in parks if p["altitude"]]
    lo_parks = [p for p in parks if not p["altitude"]]
    a, o = d[d["alt_team"]], d[~d["alt_team"]]
    b = d[d["r5"]].copy()
    b["rank"], b["rank_adj"] = b["stuff_plus"].rank(ascending=False, method="min"), b["stuff_adj"].rank(ascending=False, method="min")
    ba, bo = b[b["alt_team"]], b[~b["alt_team"]]
    risers = ba.sort_values(["delta", "id"], ascending=[False, True]).head(5)
    city = lambda c: clubs.get(c, c).rsplit(" ", 1)[0] if c in clubs else c  # noqa: E731  ("Salt Lake Bees" -> "Salt Lake")
    N = {
        "alt_season": str(res["season"]), "alt_choose_season": str(res["season"] - 1), "alt_fit_seasons": " and ".join(map(str, res["fit_seasons"])),
        "alt_min_pitches": str(res["min_pitches"]), "alt_n_parks": str(len(parks)), "alt_n_alt_parks": {5: "five"}.get(len(hi_parks), str(len(hi_parks))),
        "alt_factor_lo": _pct(min(p["factor"] for p in hi_parks)), "alt_factor_hi": _pct(max(p["factor"] for p in hi_parks)),
        "alt_other_factor_lo": _pct(min(p["factor"] for p in lo_parks)), "alt_other_factor_hi": _pct(max(p["factor"] for p in lo_parks)),
        "alt_park_list": ", ".join(f"{city(p['park'])} ({_pct(p['factor'])})" for p in sorted(hi_parks, key=lambda p: p["factor"])),
        "alt_fit_pitchers": f"{res['fit_reference_pitchers']:,}", "alt_extension_in": f"{abs(12 * res['extension_offset_ft']):.1f}",
        "alt_n_pitchers": f"{len(d):,}", "alt_team_n": str(len(a)),
        "alt_team_delta": _sgn(a["delta"].mean()), "alt_other_delta": _sgn(o["delta"].mean()),
        "alt_net_delta": f"{a['delta'].mean() - o['delta'].mean():.1f}",
        "alt_val_n": str(v["pitchers"]),
        "alt_val_alt_before": _sgn(v["as_recorded"]["altitude"]), "alt_val_alt_after": _sgn(v["adjusted"]["altitude"]),
        "alt_val_other_before": _sgn(v["as_recorded"]["others"]), "alt_val_other_after": _sgn(v["adjusted"]["others"]),
        "alt_val_gap_before": _sgn(v["as_recorded"]["gap"]), "alt_val_gap_after": _sgn(v["adjusted"]["gap"]),
        "alt_r5_n": str(len(b)), "alt_r5_alt_n": str(len(ba)),
        "alt_r5_alt_delta": _sgn(ba["delta"].mean()), "alt_r5_other_delta": _sgn(bo["delta"].mean()),
        "alt_r5_alt_100_before": str(int((ba["stuff_plus"] >= 100).sum())), "alt_r5_alt_100_after": str(int((ba["stuff_adj"] >= 100).sum())),
        "alt_r5_top": str(TOP),
        "alt_r5_alt_top_before": str(int((ba["rank"] <= TOP).sum())), "alt_r5_alt_top_after": str(int((ba["rank_adj"] <= TOP).sum())),
        "alt_r5_max_delta": _sgn(ba["delta"].max()),
        "alt_r5_risers": "; ".join(f"[{r.name}](/baseball/player.html?id={r.id}) ({r.org}, {r.stuff_plus:.0f} to {r.stuff_adj:.0f})"
                                   for r in risers.itertuples()),
    }
    # Claims the article makes. If one stops holding, rewrite the sentence.
    if not max(p["hi"] for p in hi_parks) < min(p["lo"] for p in lo_parks):
        raise StaleProse("Altitude article: 'the five altitude parks read less movement than every other park' (intervals apart) no longer holds.")
    if not a["delta"].mean() > 2 * o["delta"].mean():
        raise StaleProse("Altitude article: 'pitchers on the five altitude teams gain more than twice what others do' no longer holds.")
    if not (abs(v["adjusted"]["altitude"]) < abs(v["as_recorded"]["altitude"]) / 3 and v["adjusted"]["altitude"] < 0):
        raise StaleProse("Altitude article: 'the adjustment removes most of the altitude gap and slightly overshoots' no longer holds.")
    if not d["alt_share"].corr(d["delta"]) > 0.5:
        raise StaleProse("Altitude article: 'the more of a pitcher's pitches were thrown at altitude, the larger his change' no longer holds.")
    if not v["additive_altitude"] > v["as_recorded"]["altitude"] / 2:
        raise StaleProse("Altitude article: 'an additive correction removed less than half of the altitude gap' no longer holds.")
    if not v["adjusted"]["others"] > 0.5:
        raise StaleProse("Altitude article: 'about a point remains at every park (velocity is not adjusted)' no longer holds.")
    if not (int((ba["stuff_adj"] >= 100).sum()) > int((ba["stuff_plus"] >= 100).sum()) and len(risers) == 5):
        raise StaleProse("Altitude article: 'more of the board's altitude pitchers grade 100 or better after the adjustment' no longer holds.")
    if not int((ba["rank_adj"] <= TOP).sum()) - int((ba["rank"] <= TOP).sum()) <= 2:
        raise StaleProse(f"Altitude article: 'the top {TOP} of the board barely changes' no longer holds.")
    return data, N
