"""Feature engineering shared by training and scoring.

Every pitch is described only by what the pitcher controls physically: velocity,
movement, spin, release point, extension, arm angle, and how the pitch differs
from that pitcher's primary fastball. No location, count, or outcome information
goes into the features, so the model grades the pitch itself rather than how it
was used.
"""
import numpy as np
import pandas as pd

FASTBALLS = {"FF", "SI", "FC"}
BREAKING = {"SL", "ST", "CU", "KC", "SV", "CS"}
OFFSPEED = {"CH", "FS", "FO", "SC"}

WOBA_SCALE = 1.24  # approx. FanGraphs wOBA scale, 2024-2026

FEATURES = [
    "velo", "ivb", "hb", "spin", "spin_axis", "extension",
    "rel_z", "rel_x", "arm_angle",
    "velo_diff", "ivb_diff", "hb_diff",
]


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
