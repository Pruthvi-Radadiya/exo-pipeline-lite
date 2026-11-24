"""Show what the quality checks found in one recording.

Run:  python scripts/plot_qc.py rec_007
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def shade_dropouts(axis, dropouts: list[dict], stream_name: str) -> None:
    for index, dropout in enumerate(
        [d for d in dropouts if d["stream"] == stream_name]
    ):
        axis.axvspan(
            dropout["start_s"],
            dropout["start_s"] + dropout["length_s"],
            color="gray",
            alpha=0.35,
            label="dropout (data missing)" if index == 0 else None,
        )


def plot_qc(recording_id: str) -> None:
    aligned_df = pd.read_parquet(f"data/synced/{recording_id}.parquet")
    report = json.loads(Path(f"data/qc/{recording_id}_qc.json").read_text())
    checks = report["checks"]
    times_s = aligned_df["timestamp_s"]

    figure, (tension_axis, forearm_axis, gyro_axis) = plt.subplots(
        3, 1, figsize=(11, 8), sharex=True
    )

    # 1. cable tension: spikes and dropouts
    tension_axis.plot(
        times_s, aligned_df["left_cable_tension_n"], label="left cable tension"
    )
    spike_times_s = [
        s["timestamp_s"]
        for s in checks["spikes"]
        if s["column"] == "left_cable_tension_n"
    ]
    spike_rows = [int((times_s - time_s).abs().idxmin()) for time_s in spike_times_s]
    tension_axis.plot(
        times_s[spike_rows],
        aligned_df["left_cable_tension_n"].iloc[spike_rows],
        "rx",
        markersize=9,
        label="spike found",
    )
    shade_dropouts(tension_axis, checks["dropouts"], "cable_drive")
    tension_axis.set_ylabel("tension [N]")
    tension_axis.set_title("1. Cable drive: spikes (red crosses) and dropouts (grey)")

    # 2. forearm angles: a stuck sensor and disagreeing arms
    forearm_axis.plot(
        times_s, aligned_df["left_forearm_angle_deg"], label="left forearm"
    )
    forearm_axis.plot(
        times_s, aligned_df["right_forearm_angle_deg"], label="right forearm"
    )
    for index, flatline in enumerate(checks["flatlines"]):
        forearm_axis.axvspan(
            flatline["start_s"],
            flatline["end_s"],
            color="tab:orange",
            alpha=0.2,
            label="flat line found" if index == 0 else None,
        )
    for index, disagreement in enumerate(checks["disagreements"]):
        forearm_axis.axvspan(
            disagreement["start_s"],
            disagreement["end_s"],
            ymin=0.0,
            ymax=0.06,
            color="tab:red",
            label="left and right disagree" if index == 0 else None,
        )
    forearm_axis.set_ylabel("forearm angle [deg]")
    forearm_axis.set_title(
        "2. Forearm sensors: one arm stops moving while the other keeps lifting"
    )

    # 3. trunk gyro: dropouts
    gyro_axis.plot(
        times_s, aligned_df["trunk_gyro_y_dps"], color="tab:green", label="trunk gyro"
    )
    shade_dropouts(gyro_axis, checks["dropouts"], "trunk_imu")
    gyro_axis.set_ylabel("trunk gyro [deg/s]")
    gyro_axis.set_xlabel("time [s]")
    gyro_axis.set_title("3. Trunk IMU: short dropouts")

    for axis in (tension_axis, forearm_axis, gyro_axis):
        axis.legend(loc="upper right")
        axis.grid(alpha=0.3)
    figure.suptitle(f"{recording_id}: decision = {report['status'].upper()}")
    figure.tight_layout()

    output_path = Path("docs") / f"{recording_id}_qc.png"
    figure.savefig(output_path, dpi=110)
    print(f"saved {output_path}")


if __name__ == "__main__":
    plot_qc(sys.argv[1] if len(sys.argv) > 1 else "rec_007")
