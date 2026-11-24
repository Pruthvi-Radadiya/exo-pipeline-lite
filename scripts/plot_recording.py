"""Quick look at one recording. Saves a PNG so it works on any machine.

Run:  python scripts/plot_recording.py data/raw/rec_000
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_recording(recording_dir: str) -> None:
    recording_dir = Path(recording_dir)
    trunk_imu_df = pd.read_csv(recording_dir / "trunk_imu.csv")
    cable_df = pd.read_csv(recording_dir / "cable_drive.csv")
    forearm_df = pd.read_csv(recording_dir / "forearm_sensors.csv")
    metadata = json.loads((recording_dir / "metadata.json").read_text())

    figure, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    forearm_axis, tension_axis, travel_axis, gyro_axis = axes

    forearm_axis.plot(
        forearm_df.timestamp_s, forearm_df.left_forearm_angle_deg, label="left forearm"
    )
    forearm_axis.plot(
        forearm_df.timestamp_s,
        forearm_df.right_forearm_angle_deg,
        label="right forearm",
    )
    forearm_axis.set_ylabel("forearm angle [deg]")

    tension_axis.plot(
        cable_df.timestamp_s, cable_df.left_cable_tension_n, label="left cable tension"
    )
    tension_axis.plot(
        cable_df.timestamp_s,
        cable_df.right_cable_tension_n,
        label="right cable tension",
    )
    tension_axis.set_ylabel("tension [N]")

    travel_axis.plot(
        cable_df.timestamp_s, cable_df.left_cable_travel_mm, label="left cable travel"
    )
    travel_axis.plot(
        cable_df.timestamp_s, cable_df.right_cable_travel_mm, label="right cable travel"
    )
    travel_axis.set_ylabel("cable pulled in [mm]")

    gyro_axis.plot(
        trunk_imu_df.timestamp_s,
        trunk_imu_df.trunk_gyro_y_dps,
        label="trunk gyro y",
        color="tab:green",
    )
    gyro_axis.set_ylabel("trunk gyro [deg/s]")
    gyro_axis.set_xlabel("device time [s]")

    for axis in axes:
        axis.legend(loc="upper right")
        axis.grid(alpha=0.3)
    num_faults = len(metadata["ground_truth"]["faults"])
    figure.suptitle(f"{metadata['recording_id']}  ({num_faults} injected faults)")
    figure.tight_layout()

    output_path = Path("docs") / f"{metadata['recording_id']}.png"
    output_path.parent.mkdir(exist_ok=True)
    figure.savefig(output_path, dpi=110)
    print(f"saved {output_path}")


if __name__ == "__main__":
    plot_recording(sys.argv[1] if len(sys.argv) > 1 else "data/raw/rec_000")
