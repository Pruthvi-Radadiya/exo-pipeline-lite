"""Step 1: generate fake recordings of a cable-driven arm exoskeleton.

Model (our assumption, based on the public product description): a backpack
unit with motors pulls cables that run to cuffs on both forearms. Three
"devices" record the same lifting movement, each with its own sample rate,
clock offset, clock drift, sensor bias and a few faults. We keep the ground
truth in metadata.json so later steps (sync, calibration, QC) can be tested
against known answers.

Run:  python -m exo.generate --out data/raw --n 20
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- body and sensor model
# How big each signal is at the deepest point of a lift (lift_level = 1).
TRUNK_REST_ANGLE_DEG = 5
TRUNK_LIFT_ANGLE_DEG = 45  # trunk bends forward this much more at the deepest point
FOREARM_REST_ANGLE_DEG = 5
FOREARM_LIFT_ANGLE_DEG = 80  # forearms rise this much during a lift
LEFT_CABLE_TENSION_PEAK_N = 70
RIGHT_CABLE_TENSION_PEAK_N = 65
CABLE_TRAVEL_PEAK_MM = 200  # how much cable the motor pulls in
GRAVITY_MS2 = 9.81

# Random jitter added to every sample.
ANGLE_NOISE_DEG = 0.3
GYRO_NOISE_DPS = 0.5
ACCEL_NOISE_MS2 = 0.05
TENSION_NOISE_N = 0.8
TRAVEL_NOISE_MM = 0.3


@dataclass
class Config:
    seed: int = 42
    n_recordings: int = 20
    duration_s: float = 120.0
    still_s: float = (
        5.0  # worker stands still at the start (used for calibration later)
    )
    trunk_imu_hz: int = 200
    cable_hz: int = 100
    forearm_hz: int = 50
    out_dir: str = "data/raw"


@dataclass
class Lift:
    start_s: float  # when the lift begins
    duration_s: float  # how long it takes
    strength: float  # 1.0 = normal lift, 1.2 = a bit deeper and harder


def make_lifts(random_gen: np.random.Generator, config: Config) -> list[Lift]:
    """Schedule: lift, rest 2 to 5 seconds, lift again, until the recording ends."""
    lifts = []
    next_start_s = config.still_s + 2.0
    while next_start_s < config.duration_s - 5.0:
        duration_s = float(random_gen.uniform(2.5, 3.5))
        strength = float(random_gen.uniform(0.8, 1.2))
        lifts.append(Lift(next_start_s, duration_s, strength))
        rest_s = float(random_gen.uniform(2.0, 5.0))
        next_start_s += duration_s + rest_s
    return lifts


def simulate_lift_motion(
    times_s: np.ndarray, lifts: list[Lift]
) -> tuple[np.ndarray, np.ndarray]:
    """For every time, return (lift_level, lift_speed).

    lift_level: 0 = standing still, about 1 = deepest point of a lift.
    lift_speed: how fast lift_level changes per second (the derivative).

    One lift is a smooth bump: 0 -> 1 -> 0. We use the same hidden movement for
    every sensor, so the trunk gyro (speed) and the cable travel (level) agree.
    """
    lift_level = np.zeros_like(times_s)
    lift_speed = np.zeros_like(times_s)
    for lift in lifts:
        progress = (
            times_s - lift.start_s
        ) / lift.duration_s  # 0 at lift start, 1 at lift end
        is_lifting = (progress > 0) & (progress < 1)
        progress_in_lift = progress[is_lifting]
        lift_level[is_lifting] += lift.strength * np.sin(np.pi * progress_in_lift) ** 2
        lift_speed[is_lifting] += (
            lift.strength
            * (np.pi / lift.duration_s)
            * np.sin(2 * np.pi * progress_in_lift)
        )
    return lift_level, lift_speed


def make_sample_times(
    num_samples: int, sample_rate_hz: int, clock_offset_s: float, clock_drift_ppm: float
):
    """Return (device_times_s, true_times_s) for one device.

    device_times_s: what the device writes into the file (its own clock)
    true_times_s:   when each sample really happened

    true_time = (device_time - clock_offset) / (1 + clock_drift)
    """
    device_times_s = np.arange(num_samples) / sample_rate_hz
    true_times_s = (device_times_s - clock_offset_s) / (1.0 + clock_drift_ppm * 1e-6)
    return device_times_s, true_times_s


def apply_dropouts(
    stream_df: pd.DataFrame, random_gen, config: Config, stream_name: str, faults: list
) -> pd.DataFrame:
    """Cut 1 to 3 random time windows out of the stream (the rows disappear)."""
    for _ in range(int(random_gen.integers(1, 4))):
        dropout_start_s = float(
            random_gen.uniform(config.still_s + 5, config.duration_s - 5)
        )
        dropout_length_s = float(random_gen.uniform(0.3, 1.0))
        in_dropout = (stream_df["timestamp_s"] >= dropout_start_s) & (
            stream_df["timestamp_s"] < dropout_start_s + dropout_length_s
        )
        stream_df = stream_df[~in_dropout]
        faults.append(
            {
                "type": "dropout",
                "stream": stream_name,
                "start_s": round(dropout_start_s, 3),
                "length_s": round(dropout_length_s, 3),
            }
        )
    return stream_df.reset_index(drop=True)


def add_spikes(
    cable_df: pd.DataFrame, column: str, random_gen, config: Config, faults: list
) -> pd.DataFrame:
    """Add 3 to 6 huge single-sample spikes to one column (like a glitchy load cell)."""
    for _ in range(int(random_gen.integers(3, 7))):
        first_allowed_row = int(
            config.still_s * config.cable_hz
        )  # keep the still period clean
        row_index = int(random_gen.integers(first_allowed_row, len(cable_df) - 1))
        spike_size = float(random_gen.choice([-1, 1]) * random_gen.uniform(150, 250))
        cable_df.loc[row_index, column] += spike_size
        faults.append(
            {
                "type": "spike",
                "stream": "cable_drive",
                "column": column,
                "timestamp_s": round(float(cable_df.loc[row_index, "timestamp_s"]), 3),
            }
        )
    return cable_df


def generate_recording(recording_index: int, config: Config):
    random_gen = np.random.default_rng(
        [config.seed, recording_index]
    )  # reproducible per recording
    lifts = make_lifts(random_gen, config)
    faults: list[dict] = []

    # Device quirks of this recording. Later steps must recover these numbers.
    ground_truth = {
        "cable_offset_s": float(random_gen.uniform(0.025, 0.045)),
        "forearm_offset_s": float(random_gen.uniform(-0.030, -0.010)),
        "cable_drift_ppm": float(random_gen.uniform(30, 70)),
        "forearm_drift_ppm": float(random_gen.uniform(10, 50)),
        "bias": {
            "trunk_gyro_y_dps": float(random_gen.normal(0, 0.8)),
            "left_cable_tension_n": float(random_gen.normal(0, 1.5)),
            "right_cable_tension_n": float(random_gen.normal(0, 1.5)),
        },
        "lifts": [asdict(lift) for lift in lifts],
    }
    bias = ground_truth["bias"]

    # Trunk IMU in the backpack: the reference clock (no offset, no drift)
    num_samples = int(config.duration_s * config.trunk_imu_hz)
    device_times_s, true_times_s = make_sample_times(
        num_samples, config.trunk_imu_hz, 0.0, 0.0
    )
    lift_level, lift_speed = simulate_lift_motion(true_times_s, lifts)
    trunk_pitch_rad = np.deg2rad(
        TRUNK_REST_ANGLE_DEG + TRUNK_LIFT_ANGLE_DEG * lift_level
    )
    trunk_imu_df = pd.DataFrame(
        {
            "timestamp_s": device_times_s,
            "trunk_gyro_y_dps": (
                TRUNK_LIFT_ANGLE_DEG * lift_speed
                + bias["trunk_gyro_y_dps"]
                + random_gen.normal(0, GYRO_NOISE_DPS, num_samples)
            ),
            "trunk_acc_x_ms2": GRAVITY_MS2 * np.sin(trunk_pitch_rad)
            + random_gen.normal(0, ACCEL_NOISE_MS2, num_samples),
            "trunk_acc_z_ms2": GRAVITY_MS2 * np.cos(trunk_pitch_rad)
            + random_gen.normal(0, ACCEL_NOISE_MS2, num_samples),
        }
    )

    # Cable drive (motors): clock roughly 35 ms off and drifting
    num_samples = int(config.duration_s * config.cable_hz)
    device_times_s, true_times_s = make_sample_times(
        num_samples,
        config.cable_hz,
        ground_truth["cable_offset_s"],
        ground_truth["cable_drift_ppm"],
    )
    lift_level, _ = simulate_lift_motion(true_times_s, lifts)
    cable_df = pd.DataFrame(
        {
            "timestamp_s": device_times_s,
            "left_cable_tension_n": (
                LEFT_CABLE_TENSION_PEAK_N * lift_level
                + bias["left_cable_tension_n"]
                + random_gen.normal(0, TENSION_NOISE_N, num_samples)
            ),
            "right_cable_tension_n": (
                RIGHT_CABLE_TENSION_PEAK_N * lift_level
                + bias["right_cable_tension_n"]
                + random_gen.normal(0, TENSION_NOISE_N, num_samples)
            ),
            "left_cable_travel_mm": (
                CABLE_TRAVEL_PEAK_MM * lift_level
                + random_gen.normal(0, TRAVEL_NOISE_MM, num_samples)
            ),
            "right_cable_travel_mm": (
                CABLE_TRAVEL_PEAK_MM * lift_level
                + random_gen.normal(0, TRAVEL_NOISE_MM, num_samples)
            ),
        }
    )

    # Forearm sensor modules: slower, with their own offset and drift
    num_samples = int(config.duration_s * config.forearm_hz)
    device_times_s, true_times_s = make_sample_times(
        num_samples,
        config.forearm_hz,
        ground_truth["forearm_offset_s"],
        ground_truth["forearm_drift_ppm"],
    )
    lift_level, _ = simulate_lift_motion(true_times_s, lifts)
    forearm_df = pd.DataFrame(
        {
            "timestamp_s": device_times_s,
            "left_forearm_angle_deg": (
                FOREARM_REST_ANGLE_DEG
                + FOREARM_LIFT_ANGLE_DEG * lift_level
                + random_gen.normal(0, ANGLE_NOISE_DEG, num_samples)
            ),
            "right_forearm_angle_deg": (
                FOREARM_REST_ANGLE_DEG
                + FOREARM_LIFT_ANGLE_DEG * lift_level
                + random_gen.normal(0, ANGLE_NOISE_DEG, num_samples)
            ),
        }
    )

    # Faults, injected on purpose so the QC step has something to find
    cable_df = add_spikes(cable_df, "left_cable_tension_n", random_gen, config, faults)
    cable_df = apply_dropouts(cable_df, random_gen, config, "cable_drive", faults)
    trunk_imu_df = apply_dropouts(trunk_imu_df, random_gen, config, "trunk_imu", faults)
    if (
        recording_index % 10 == 7
    ):  # recordings 7, 17, ... get a stuck right forearm sensor
        stuck_from_s = 60.0
        stuck_rows = forearm_df["timestamp_s"] >= stuck_from_s
        stuck_value = float(
            forearm_df.loc[stuck_rows, "right_forearm_angle_deg"].iloc[0]
        )
        forearm_df.loc[stuck_rows, "right_forearm_angle_deg"] = stuck_value
        faults.append(
            {
                "type": "flatline",
                "stream": "forearm_sensors",
                "column": "right_forearm_angle_deg",
                "from_s": stuck_from_s,
            }
        )
    ground_truth["faults"] = faults

    metadata = {
        "recording_id": f"rec_{recording_index:03d}",
        "worker_id": f"W{(recording_index % 6) + 1:03d}",
        "device_serial": f"LU20-{1000 + recording_index}",
        "site": ["Warehouse Munich", "Plant Ingolstadt", "Hub Hamburg"][
            recording_index % 3
        ],
        "recorded_at": f"2026-10-{(recording_index % 28) + 1:02d}T08:00:00Z",
        "task": "parcel_lifting",
        "config": asdict(config),
        "ground_truth": ground_truth,
    }
    streams = {
        "trunk_imu": trunk_imu_df,
        "cable_drive": cable_df,
        "forearm_sensors": forearm_df,
    }
    return streams, metadata


def write_recording(streams: dict, metadata: dict, out_dir: Path) -> Path:
    recording_dir = Path(out_dir) / metadata["recording_id"]
    recording_dir.mkdir(parents=True, exist_ok=True)
    for stream_name, stream_df in streams.items():
        stream_df.to_csv(
            recording_dir / f"{stream_name}.csv", index=False, float_format="%.6f"
        )
    (recording_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return recording_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=120.0)
    args = parser.parse_args()

    config = Config(
        seed=args.seed, n_recordings=args.n, duration_s=args.duration, out_dir=args.out
    )
    out_dir = Path(config.out_dir)
    for recording_index in range(config.n_recordings):
        streams, metadata = generate_recording(recording_index, config)
        recording_dir = write_recording(streams, metadata, out_dir)
        row_counts = {name: len(df) for name, df in streams.items()}
        num_faults = len(metadata["ground_truth"]["faults"])
        print(f"{recording_dir}  rows={row_counts}  faults={num_faults}")
    print(f"done: {config.n_recordings} recordings in {out_dir}")


if __name__ == "__main__":
    main()
