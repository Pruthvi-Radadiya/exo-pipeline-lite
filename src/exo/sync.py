"""Step 3a: find the clock offset and drift of every stream, then put all streams
on one common time grid.

The trunk IMU is the reference clock. For every other stream we measure how far
its clock is off (offset) and how fast it runs away (drift) by comparing how fast
the worker moves, as seen by the two devices.

Run:  python -m exo.sync --mcap data/mcap --out data/synced
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exo.calibrate import estimate_bias, remove_bias
from exo.mcap_io import read_mcap

REFERENCE_STREAM_NAME = "trunk_imu"

# Which columns show "how fast does the worker move", and do we need the derivative?
# The gyro is already a speed. Cable travel and forearm angle are positions, so we
# take their derivative to get a speed that looks like the gyro.
# We use only the LEFT forearm: the right one can get stuck (see step 4, QC).
SPEED_SIGNAL_SOURCES = {
    "trunk_imu": (["trunk_gyro_y_dps"], False),
    "cable_drive": (["left_cable_travel_mm", "right_cable_travel_mm"], True),
    "forearm_sensors": (["left_forearm_angle_deg"], True),
}

FINE_GRID_STEP_S = 0.001  # 1 ms grid for measuring the offset
MAX_LAG_S = 0.2  # search offsets between -200 ms and +200 ms
SMOOTHING_WINDOW_S = 0.1  # smooth before taking the derivative
WINDOW_LENGTH_S = 30.0  # measure the offset in 30 second windows
WINDOW_STEP_S = 15.0  # a new window every 15 seconds
FIRST_WINDOW_START_S = 5.0
MIN_VALID_SAMPLES = 5000  # a window needs at least 5 s of valid data
MIN_CORRELATION = 0.5  # ignore windows where the match is poor
MIN_WINDOWS_FOR_FIT = 3
OUTPUT_RATE_HZ = 100  # the common grid
MAX_GAP_PERIODS = 3  # a hole longer than 3 sample periods counts as missing data


def moving_average(values: np.ndarray, window_samples: int) -> np.ndarray:
    kernel = np.ones(window_samples) / window_samples
    return np.convolve(values, kernel, mode="same")


def nearest_sample_distance_s(
    sample_times_s: np.ndarray, query_times_s: np.ndarray
) -> np.ndarray:
    """For every query time, how far away is the closest real sample?"""
    right_index = np.clip(
        np.searchsorted(sample_times_s, query_times_s), 1, len(sample_times_s) - 1
    )
    distance_to_left_s = np.abs(query_times_s - sample_times_s[right_index - 1])
    distance_to_right_s = np.abs(sample_times_s[right_index] - query_times_s)
    return np.minimum(distance_to_left_s, distance_to_right_s)


def make_speed_signal(
    stream_df: pd.DataFrame, stream_name: str, grid_times_s: np.ndarray
) -> np.ndarray:
    """A smooth 'how fast does the worker move' signal on the fine grid.

    Where the stream had a dropout, the result is NaN: we do not trust invented data.
    """
    column_names, needs_derivative = SPEED_SIGNAL_SOURCES[stream_name]
    sample_times_s = stream_df["timestamp_s"].to_numpy()
    values = stream_df[column_names].to_numpy().mean(axis=1)  # average the columns
    values_on_grid = np.interp(grid_times_s, sample_times_s, values)
    window_samples = int(SMOOTHING_WINDOW_S / FINE_GRID_STEP_S)
    smooth_values = moving_average(values_on_grid, window_samples)
    speed = (
        np.gradient(smooth_values, FINE_GRID_STEP_S)
        if needs_derivative
        else smooth_values
    )

    max_gap_s = MAX_GAP_PERIODS * float(np.median(np.diff(sample_times_s)))
    is_missing = (
        nearest_sample_distance_s(sample_times_s, grid_times_s) > max_gap_s
    ).astype(float)
    margin_samples = 2
    is_near_missing = (
        np.convolve(is_missing, np.ones(window_samples + margin_samples), mode="same")
        > 0
    )
    speed[is_near_missing] = np.nan
    return speed


def refine_peak_position(correlations: np.ndarray, best_index: int) -> float:
    """Sub-sample peak position (-0.5 to 0.5) from a parabola through the peak and its neighbors."""
    if best_index == 0 or best_index == len(correlations) - 1:
        return 0.0
    left, center, right = (
        correlations[best_index - 1],
        correlations[best_index],
        correlations[best_index + 1],
    )
    denominator = left - 2 * center + right
    if denominator == 0:
        return 0.0
    return float(0.5 * (left - right) / denominator)


def correlation_at_shift(
    reference_window: np.ndarray, other_signal: np.ndarray, start_index: int, shift: int
) -> float:
    """How well does `other` match `reference` if we read `other` `shift` samples later?"""
    end_index = start_index + len(reference_window)
    if start_index + shift < 0 or end_index + shift > len(other_signal):
        return -1.0
    other_window = other_signal[start_index + shift : end_index + shift]
    is_valid = ~np.isnan(reference_window) & ~np.isnan(other_window)
    if is_valid.sum() < MIN_VALID_SAMPLES:
        return -1.0
    reference_centered = reference_window[is_valid] - reference_window[is_valid].mean()
    other_centered = other_window[is_valid] - other_window[is_valid].mean()
    denominator = np.linalg.norm(reference_centered) * np.linalg.norm(other_centered)
    if denominator == 0:
        return -1.0
    return float(np.dot(reference_centered, other_centered) / denominator)


def estimate_lag_in_window(reference_speed, other_speed, window_start_s, window_end_s):
    """Return (lag_s, correlation, shifts_s, correlations) for one window.

    lag_s > 0 means the other device's clock is ahead: it shows the same event later.
    """
    start_index = int(window_start_s / FINE_GRID_STEP_S)
    end_index = int(window_end_s / FINE_GRID_STEP_S)
    reference_window = reference_speed[start_index:end_index]
    max_shift = int(MAX_LAG_S / FINE_GRID_STEP_S)
    shifts = np.arange(-max_shift, max_shift + 1)
    correlations = np.array(
        [
            correlation_at_shift(reference_window, other_speed, start_index, int(shift))
            for shift in shifts
        ]
    )
    best_index = int(np.argmax(correlations))
    best_shift = shifts[best_index] + refine_peak_position(correlations, best_index)
    return (
        float(best_shift * FINE_GRID_STEP_S),
        float(correlations[best_index]),
        shifts * FINE_GRID_STEP_S,
        correlations,
    )


def estimate_offset_and_drift(
    reference_speed, other_speed, end_of_data_s: float
) -> dict:
    """Measure the lag in several windows, then fit a straight line lag(t) = offset + drift * t."""
    window_centers_s, lags_s, correlations = [], [], []
    window_start_s = FIRST_WINDOW_START_S
    while window_start_s + WINDOW_LENGTH_S <= end_of_data_s:
        window_end_s = window_start_s + WINDOW_LENGTH_S
        lag_s, correlation, _, _ = estimate_lag_in_window(
            reference_speed, other_speed, window_start_s, window_end_s
        )
        window_centers_s.append((window_start_s + window_end_s) / 2)
        lags_s.append(lag_s)
        correlations.append(correlation)
        window_start_s += WINDOW_STEP_S

    is_good = np.array(correlations) >= MIN_CORRELATION
    if is_good.sum() < MIN_WINDOWS_FOR_FIT:
        raise ValueError(
            f"only {int(is_good.sum())} windows matched well, cannot estimate the offset"
        )
    drift_slope, offset_s = np.polyfit(
        np.array(window_centers_s)[is_good], np.array(lags_s)[is_good], 1
    )
    return {
        "offset_s": float(offset_s),
        "drift_ppm": float(drift_slope * 1e6),
        "windows_used": int(is_good.sum()),
        "min_correlation": float(np.min(np.array(correlations)[is_good])),
        "window_centers_s": [float(value) for value in window_centers_s],
        "window_lags_s": [float(value) for value in lags_s],
    }


def correct_timestamps(
    device_times_s: np.ndarray, offset_s: float, drift_ppm: float
) -> np.ndarray:
    """Turn device clock times into reference clock times."""
    return (device_times_s - offset_s) / (1.0 + drift_ppm * 1e-6)


def mark_gaps_as_nan(
    values: np.ndarray,
    stream_times_s: np.ndarray,
    grid_times_s: np.ndarray,
    max_gap_s: float,
) -> np.ndarray:
    """Set grid points to NaN when the nearest real sample is further away than max_gap_s."""
    cleaned = values.copy()
    cleaned[nearest_sample_distance_s(stream_times_s, grid_times_s) > max_gap_s] = (
        np.nan
    )
    return cleaned


def align_streams(streams: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Estimate every stream's clock error and resample all streams to one 100 Hz grid."""
    end_of_data_s = min(
        stream_df["timestamp_s"].iloc[-1] for stream_df in streams.values()
    )
    fine_grid_times_s = np.arange(0.0, end_of_data_s, FINE_GRID_STEP_S)
    reference_speed = make_speed_signal(
        streams[REFERENCE_STREAM_NAME], REFERENCE_STREAM_NAME, fine_grid_times_s
    )

    clock_report = {REFERENCE_STREAM_NAME: {"offset_s": 0.0, "drift_ppm": 0.0}}
    corrected_times_s = {
        REFERENCE_STREAM_NAME: streams[REFERENCE_STREAM_NAME]["timestamp_s"].to_numpy()
    }
    for stream_name, stream_df in streams.items():
        if stream_name == REFERENCE_STREAM_NAME:
            continue
        other_speed = make_speed_signal(stream_df, stream_name, fine_grid_times_s)
        clock = estimate_offset_and_drift(reference_speed, other_speed, end_of_data_s)
        clock_report[stream_name] = clock
        corrected_times_s[stream_name] = correct_timestamps(
            stream_df["timestamp_s"].to_numpy(), clock["offset_s"], clock["drift_ppm"]
        )

    # common grid: only the time range where every stream has data
    grid_start_s = (
        np.ceil(max(times[0] for times in corrected_times_s.values()) * OUTPUT_RATE_HZ)
        / OUTPUT_RATE_HZ
    )
    grid_end_s = (
        np.floor(
            min(times[-1] for times in corrected_times_s.values()) * OUTPUT_RATE_HZ
        )
        / OUTPUT_RATE_HZ
    )
    num_rows = int(round((grid_end_s - grid_start_s) * OUTPUT_RATE_HZ)) + 1
    grid_times_s = grid_start_s + np.arange(num_rows) / OUTPUT_RATE_HZ

    aligned_df = pd.DataFrame({"timestamp_s": grid_times_s})
    nan_rows_per_stream = {}
    for stream_name, stream_df in streams.items():
        stream_times_s = corrected_times_s[stream_name]
        max_gap_s = MAX_GAP_PERIODS * float(np.median(np.diff(stream_times_s)))
        value_columns = [
            column for column in stream_df.columns if column != "timestamp_s"
        ]
        for column_name in value_columns:
            values_on_grid = np.interp(
                grid_times_s, stream_times_s, stream_df[column_name].to_numpy()
            )
            aligned_df[column_name] = mark_gaps_as_nan(
                values_on_grid, stream_times_s, grid_times_s, max_gap_s
            )
        nan_rows_per_stream[stream_name] = int(
            aligned_df[value_columns[0]].isna().sum()
        )

    report = {
        "reference_stream": REFERENCE_STREAM_NAME,
        "clocks": clock_report,
        "grid": {
            "rate_hz": OUTPUT_RATE_HZ,
            "start_s": float(grid_start_s),
            "end_s": float(grid_end_s),
            "num_rows": num_rows,
        },
        "nan_rows_per_stream": nan_rows_per_stream,
    }
    return aligned_df, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mcap", default="data/mcap", help="folder with the .mcap files"
    )
    parser.add_argument(
        "--out", default="data/synced", help="folder for the synced parquet files"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mcap_paths = sorted(Path(args.mcap).glob("*.mcap"))
    for mcap_path in mcap_paths:
        streams, file_metadata = read_mcap(mcap_path)
        aligned_df, report = align_streams(streams)
        bias = estimate_bias(aligned_df)
        calibrated_df = remove_bias(aligned_df, bias)
        report["bias"] = bias
        report["recording_id"] = file_metadata["recording_id"]

        calibrated_df.to_parquet(out_dir / f"{mcap_path.stem}.parquet", index=False)
        (out_dir / f"{mcap_path.stem}_report.json").write_text(
            json.dumps(report, indent=2)
        )
        cable = report["clocks"]["cable_drive"]
        forearm = report["clocks"]["forearm_sensors"]
        print(
            f"{mcap_path.stem}  cable offset={cable['offset_s'] * 1000:5.1f} ms drift={cable['drift_ppm']:5.1f} ppm"
            f"   forearm offset={forearm['offset_s'] * 1000:6.1f} ms drift={forearm['drift_ppm']:5.1f} ppm"
        )
    print(f"done: {len(mcap_paths)} files in {out_dir}")


if __name__ == "__main__":
    main()
