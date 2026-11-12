"""Feature engineering shared by training and scoring.

Three feature sets, one model family each (all graded on the same target):
  FEATURES          Stuff+: only what the pitcher controls physically (velocity, movement, spin,
                    release, extension, arm angle, and gaps from his primary fastball). No location,
                    count, or outcome, so it grades the pitch itself rather than how it was used.
  LOC_FEATURES      Location+: only where the pitch crossed the plate, in the count it was thrown.
                    Horizontal location is mirrored by batter side (positive = inside); height is
                    scaled to the rulebook zone from the batter's listed height (0 = bottom at 27%,
                    1 = top at 53.5%). No velocity or movement.
  PITCHING_FEATURES Pitching+: both, so the model can value a pitch's shape where it was thrown.
"""
from functools import cache

import numpy as np
import pandas as pd

from sparvi.config import RAW

# The rulebook strike zone as a share of the batter's height: what Statcast's sz_top/sz_bot record
# from 2026 (checked: 53.5% and 27.0% of listed height, to rounding). Before 2026 those fields were set
# from the batter's stance on each pitch, a different and taller zone, so Location+ uses height for
# every season to keep one definition.
ZONE_TOP, ZONE_BOT = 0.535, 0.27


@cache
def batter_heights() -> pd.Series:
    """Listed height in feet by batter id (scripts/pull_data.py writes batter_heights.parquet)."""
    h = pd.read_parquet(RAW / "batter_heights.parquet")
    return (h.set_index("batter")["height_in"] / 12).astype(float)

FASTBALLS = {"FF", "SI", "FC"}
BREAKING = {"SL", "ST", "CU", "KC", "SV", "CS"}
OFFSPEED = {"CH", "FS", "FO", "SC"}

WOBA_SCALE = 1.24  # approx. FanGraphs wOBA scale, 2024-2026

FEATURES = [
    "velo", "ivb", "hb", "spin", "spin_axis", "extension",
    "rel_z", "rel_x", "arm_angle",
    "velo_diff", "ivb_diff", "hb_diff",
]
LOC_FEATURES = ["loc_x", "loc_z", "balls", "strikes", "same_hand"]
PITCHING_FEATURES = FEATURES + LOC_FEATURES
FEATURE_SETS = {"stuff": FEATURES, "loc": LOC_FEATURES, "pitching": PITCHING_FEATURES}


def pitch_group(pt: pd.Series) -> pd.Series:
    return np.select(
        [pt.isin(FASTBALLS), pt.isin(BREAKING), pt.isin(OFFSPEED)],
        ["fastball", "breaking", "offspeed"],
        default="other",
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per pitch with handedness-normalized features.

    Lefties are mirrored so arm-side movement and release side are positive for
    everyone; a LHP sinker and a RHP sinker then live in the same feature space.
    """
    out = df[[
        "game_pk", "game_date", "game_year", "pitcher", "player_name", "p_throws",
        "pitch_type", "at_bat_number", "pitch_number", "home_team", "away_team",
        "inning_topbot", "delta_run_exp",
    ]].copy()
    lefty = df["p_throws"].eq("L")
    sign = np.where(lefty, -1.0, 1.0)

    # The pitcher's team is the home team when the visitors bat (top of the inning).
    out["team"] = np.where(df["inning_topbot"].eq("Top"), df["home_team"], df["away_team"])
    out["group"] = pitch_group(df["pitch_type"])
    out["velo"] = df["release_speed"]
    out["ivb"] = df["pfx_z"] * 12          # feet -> inches, induced vertical break
    out["hb"] = df["pfx_x"] * 12 * -sign   # positive = arm side for both hands
    out["spin"] = df["release_spin_rate"]
    out["spin_axis"] = np.where(lefty, 360 - df["spin_axis"], df["spin_axis"])
    out["extension"] = df["release_extension"]
    out["rel_z"] = df["release_pos_z"]
    out["rel_x"] = df["release_pos_x"] * -sign
    out["arm_angle"] = df["arm_angle"]

    # Location and count. Statcast's plate_x is from the catcher's view (positive = first-base side),
    # so inside is negative for a right-handed batter: mirror so positive = inside for everyone.
    out["loc_x"] = np.where(df["stand"].eq("R"), -df["plate_x"], df["plate_x"])
    height = df["batter"].map(batter_heights())  # NaN if unknown; LightGBM handles missing values
    out["loc_z"] = (df["plate_z"] - ZONE_BOT * height) / ((ZONE_TOP - ZONE_BOT) * height)
    out["balls"] = df["balls"]
    out["strikes"] = df["strikes"]
    out["same_hand"] = (df["p_throws"] == df["stand"]).astype(float)

    # Primary fastball = the pitcher's most-thrown FF/SI/FC that season.
    fb = out[out["group"].eq("fastball")]
    primary = (
        fb.groupby(["pitcher", "game_year", "pitch_type"]).size()
        .rename("n").reset_index()
        .sort_values("n", ascending=False)
        .drop_duplicates(["pitcher", "game_year"])
    )
    fb_shape = (
        fb.merge(primary[["pitcher", "game_year", "pitch_type"]])
        .groupby(["pitcher", "game_year"])[["velo", "ivb", "hb"]].mean()
        .add_prefix("fb_").reset_index()
    )
    out = out.merge(fb_shape, on=["pitcher", "game_year"], how="left")
    out["velo_diff"] = out["velo"] - out["fb_velo"]
    out["ivb_diff"] = out["ivb"] - out["fb_ivb"]
    out["hb_diff"] = out["hb"] - out["fb_hb"]
    # Pitchers with no fastball: diffs are undefined; LightGBM handles NaN natively.
    out = out.drop(columns=["fb_velo", "fb_ivb", "fb_hb"])

    # Pitcher's perspective: positive = good for the pitcher.
    out["rv"] = -out["delta_run_exp"]
    # Denoised target: on balls in play, swap the actual outcome for the expected
    # outcome given exit velo and launch angle. A 105 mph liner caught at the wall
    # and a 70 mph blooper that falls in are mostly defense and luck, not the pitch.
    # (woba - xwoba) / WOBA_SCALE converts the wOBA gap into runs.
    bip = df["estimated_woba_using_speedangle"].notna() & df["woba_value"].notna()
    luck = (df["woba_value"] - df["estimated_woba_using_speedangle"]) / WOBA_SCALE
    out["rv_x"] = out["rv"] + np.where(bip, luck, 0.0)
    keep = out["group"].ne("other") & out["velo"].notna() & out["rv"].notna()
    return out[keep].reset_index(drop=True)
