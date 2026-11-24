import numpy as np

from exo.generate import Config, generate_recording, make_sample_times
from exo.sync import (
    align_streams,
    correct_timestamps,
    mark_gaps_as_nan,
    nearest_sample_distance_s,
)


def make_streams(recording_index=0, duration_s=90.0):
    config = Config(seed=5, n_recordings=1, duration_s=duration_s)
    streams, metadata = generate_recording(recording_index, config)
    return streams, metadata["ground_truth"]


def max_clock_error_s(clock, true_offset_s, true_drift_ppm, duration_s):
    times_s = np.linspace(0, duration_s, 91)
    estimated_s = clock["offset_s"] + clock["drift_ppm"] * 1e-6 * times_s
    true_s = true_offset_s + true_drift_ppm * 1e-6 * times_s
    return float(np.max(np.abs(estimated_s - true_s)))


def test_correct_timestamps_undoes_the_generator_clock():
    device_times_s, true_times_s = make_sample_times(
        1000, 100, clock_offset_s=0.035, clock_drift_ppm=50.0
    )
    corrected_times_s = correct_timestamps(
        device_times_s, offset_s=0.035, drift_ppm=50.0
    )
    np.testing.assert_allclose(corrected_times_s, true_times_s)


def test_nearest_sample_distance():
    sample_times_s = np.array([0.0, 1.0, 2.0, 5.0])
    distances_s = nearest_sample_distance_s(
        sample_times_s, np.array([0.1, 1.5, 3.4, 4.9])
    )
    np.testing.assert_allclose(distances_s, [0.1, 0.5, 1.4, 0.1])


def test_gaps_become_nan_and_the_rest_stays():
    stream_times_s = np.array([0.0, 0.1, 0.2, 1.0, 1.1])  # hole between 0.2 and 1.0
    grid_times_s = np.arange(0.0, 1.1, 0.1)
    values = np.ones(len(grid_times_s))
    cleaned = mark_gaps_as_nan(values, stream_times_s, grid_times_s, max_gap_s=0.25)
    assert np.isnan(cleaned[5])  # t = 0.5 is far from any sample
    assert not np.isnan(cleaned[0]) and not np.isnan(cleaned[10])


def test_sync_recovers_the_injected_clock_errors():
    streams, ground_truth = make_streams()
    _, report = align_streams(streams)
    cable_error_s = max_clock_error_s(
        report["clocks"]["cable_drive"],
        ground_truth["cable_offset_s"],
        ground_truth["cable_drift_ppm"],
        90.0,
    )
    forearm_error_s = max_clock_error_s(
        report["clocks"]["forearm_sensors"],
        ground_truth["forearm_offset_s"],
        ground_truth["forearm_drift_ppm"],
        90.0,
    )
    assert cable_error_s < 0.003
    assert forearm_error_s < 0.003


def test_aligned_table_is_one_clean_100_hz_grid():
    streams, _ = make_streams()
    aligned_df, report = align_streams(streams)
    np.testing.assert_allclose(np.diff(aligned_df["timestamp_s"]), 0.01, atol=1e-9)
    assert len(aligned_df) == report["grid"]["num_rows"]
    still_df = aligned_df[aligned_df["timestamp_s"] < 5.0]
    assert not still_df.isna().any().any()  # no dropouts injected in the still period
    assert (
        report["nan_rows_per_stream"]["cable_drive"] > 0
    )  # dropouts become NaN, not invented data
