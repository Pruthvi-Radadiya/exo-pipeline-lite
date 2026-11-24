"""Show how the clock offset of the cable drive is found, and what changes after the fix.

Run:  python scripts/plot_sync.py rec_007
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exo.mcap_io import read_mcap
from exo.sync import (
    FINE_GRID_STEP_S,
    align_streams,
    correct_timestamps,
    estimate_lag_in_window,
    make_speed_signal,
)


def plot_sync(recording_id: str) -> None:
    streams, _ = read_mcap(Path(f"data/mcap/{recording_id}.mcap"))
    ground_truth = json.loads(
        Path(f"data/raw/{recording_id}/metadata.json").read_text()
    )["ground_truth"]
    _, report = align_streams(streams)
    clock = report["clocks"]["cable_drive"]

    end_of_data_s = min(
        stream_df["timestamp_s"].iloc[-1] for stream_df in streams.values()
    )
    grid_times_s = np.arange(0.0, end_of_data_s, FINE_GRID_STEP_S)
    reference_speed = make_speed_signal(streams["trunk_imu"], "trunk_imu", grid_times_s)
    cable_df = streams["cable_drive"]
    speed_before = make_speed_signal(cable_df, "cable_drive", grid_times_s)
    corrected_df = cable_df.copy()
    corrected_df["timestamp_s"] = correct_timestamps(
        cable_df["timestamp_s"].to_numpy(), clock["offset_s"], clock["drift_ppm"]
    )
    speed_after = make_speed_signal(corrected_df, "cable_drive", grid_times_s)

    figure, (correlation_axis, drift_axis, zoom_axis) = plt.subplots(
        3, 1, figsize=(10, 11)
    )

    # 1. correlation against lag, for one 30 second window
    lag_s, correlation, shifts_s, correlations = estimate_lag_in_window(
        reference_speed, speed_before, 35.0, 65.0
    )
    correlation_axis.plot(shifts_s * 1000, correlations)
    correlation_axis.axvline(
        lag_s * 1000,
        color="tab:red",
        linestyle="--",
        label=f"best lag {lag_s * 1000:.1f} ms",
    )
    correlation_axis.set_xlabel("shift of the cable drive signal [ms]")
    correlation_axis.set_ylabel("correlation")
    correlation_axis.set_title(
        "1. Window 35 to 65 s: how well do the two signals match at each shift?"
    )
    correlation_axis.legend()

    # 2. lag per window, with the fitted line and the true line
    window_times_s = np.array(clock["window_centers_s"])
    drift_axis.plot(
        window_times_s,
        np.array(clock["window_lags_s"]) * 1000,
        "o",
        label="measured per window",
    )
    time_axis_s = np.linspace(0, 120, 121)
    drift_axis.plot(
        time_axis_s,
        (clock["offset_s"] + clock["drift_ppm"] * 1e-6 * time_axis_s) * 1000,
        label="fitted line",
    )
    true_offset_ms = (
        ground_truth["cable_offset_s"]
        + ground_truth["cable_drift_ppm"] * 1e-6 * time_axis_s
    ) * 1000
    drift_axis.plot(time_axis_s, true_offset_ms, "--", label="ground truth")
    drift_axis.set_xlabel("time [s]")
    drift_axis.set_ylabel("clock offset [ms]")
    drift_axis.set_title("2. The offset grows slowly over the recording (drift)")
    drift_axis.legend()

    # 3. zoom into the top of one lift, before and after the fix
    # remove the gyro bias (its mean while standing still) so the zero crossings can be compared
    still_samples = int(5.0 / FINE_GRID_STEP_S)
    reference_no_bias = reference_speed - np.nanmean(reference_speed[:still_samples])
    crossing_indices = np.where(
        (reference_no_bias[:-1] > 0) & (reference_no_bias[1:] <= 0)
    )[0]  # gyro goes from + to -
    crossing_s = grid_times_s[crossing_indices[grid_times_s[crossing_indices] > 8.0][0]]
    in_zoom = (grid_times_s > crossing_s - 0.15) & (grid_times_s < crossing_s + 0.15)
    for label, speed in [
        ("trunk gyro (reference, bias removed)", reference_no_bias),
        ("cable drive, before fix", speed_before),
        ("cable drive, after fix", speed_after),
    ]:
        zoom_axis.plot(
            (grid_times_s[in_zoom] - crossing_s) * 1000,
            speed[in_zoom] / np.nanstd(speed),
            label=label,
        )
    zoom_axis.axhline(0, color="gray", linewidth=0.5)
    zoom_axis.set_xlabel("time around the top of one lift [ms]")
    zoom_axis.set_ylabel("speed (normalized)")
    zoom_axis.set_title(
        "3. Zoom: before the fix the curves are shifted, after the fix they lie on top of each other"
    )
    zoom_axis.legend()

    figure.tight_layout()
    output_path = Path("docs") / f"{recording_id}_sync.png"
    figure.savefig(output_path, dpi=110)
    print(f"saved {output_path}")


if __name__ == "__main__":
    plot_sync(sys.argv[1] if len(sys.argv) > 1 else "rec_007")
