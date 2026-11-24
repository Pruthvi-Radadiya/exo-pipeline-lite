"""Step 3b: remove constant sensor bias, measured while the worker stands still.

At rest the true trunk gyro and the true cable tensions are zero, so whatever
they read during the still period is bias.
"""

from __future__ import annotations

import pandas as pd

STILL_PERIOD_S = 5.0  # protocol: the worker stands still for the first 5 seconds
BIAS_COLUMNS = ["trunk_gyro_y_dps", "left_cable_tension_n", "right_cable_tension_n"]
STILL_MAX_GYRO_STD_DPS = (
    2.0  # if the gyro moves more than this, the worker was not still
)


def estimate_bias(
    aligned_df: pd.DataFrame,
    bias_columns: list[str] = BIAS_COLUMNS,
    still_period_s: float = STILL_PERIOD_S,
) -> dict[str, float]:
    """Average of each column during the still period."""
    still_df = aligned_df[aligned_df["timestamp_s"] < still_period_s]
    gyro_std_dps = still_df["trunk_gyro_y_dps"].std()
    if gyro_std_dps > STILL_MAX_GYRO_STD_DPS:
        raise ValueError(
            f"worker was not still at the start (gyro std {gyro_std_dps:.1f} deg/s)"
        )
    return {column: float(still_df[column].mean()) for column in bias_columns}


def remove_bias(aligned_df: pd.DataFrame, bias: dict[str, float]) -> pd.DataFrame:
    """Return a copy with the bias subtracted from each column."""
    calibrated_df = aligned_df.copy()
    for column, bias_value in bias.items():
        calibrated_df[column] = calibrated_df[column] - bias_value
    return calibrated_df
