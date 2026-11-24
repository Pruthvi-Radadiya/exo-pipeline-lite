import numpy as np
import pandas as pd
import pytest

from exo.calibrate import BIAS_COLUMNS, estimate_bias, remove_bias
from exo.generate import Config, generate_recording
from exo.sync import align_streams


def test_estimated_bias_is_close_to_the_injected_bias():
    streams, metadata = generate_recording(
        0, Config(seed=5, n_recordings=1, duration_s=90.0)
    )
    aligned_df, _ = align_streams(streams)
    estimated_bias = estimate_bias(aligned_df)
    true_bias = metadata["ground_truth"]["bias"]
    for column in BIAS_COLUMNS:
        assert abs(estimated_bias[column] - true_bias[column]) < 0.15


def test_remove_bias_subtracts_and_does_not_change_the_input():
    aligned_df = pd.DataFrame(
        {"timestamp_s": [0.0, 1.0], "trunk_gyro_y_dps": [3.0, 4.0]}
    )
    calibrated_df = remove_bias(aligned_df, {"trunk_gyro_y_dps": 2.0})
    np.testing.assert_allclose(calibrated_df["trunk_gyro_y_dps"], [1.0, 2.0])
    np.testing.assert_allclose(aligned_df["trunk_gyro_y_dps"], [3.0, 4.0])


def test_moving_worker_at_the_start_is_rejected():
    times_s = np.arange(0.0, 6.0, 0.01)
    shaking_df = pd.DataFrame(
        {
            "timestamp_s": times_s,
            "trunk_gyro_y_dps": 30 * np.sin(2 * np.pi * times_s),
            "left_cable_tension_n": 0.0,
            "right_cable_tension_n": 0.0,
        }
    )
    with pytest.raises(ValueError):
        estimate_bias(shaking_df)
