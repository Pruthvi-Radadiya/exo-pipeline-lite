"""Step 4: quality checks on the synced tables.

We look for four kinds of problems and write a report with a decision per recording:
  1. dropouts      (runs of missing data)
  2. spikes        (single samples that are far away from their neighbors)
  3. flat lines    (a sensor that keeps repeating the same value)
  4. disagreement  (the left and the right sensor tell different stories)

Run:  python -m exo.qc --synced data/synced --out data/qc --clean data/clean
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# One representative column per stream, used to find dropouts
# (when a device sends nothing, all of its columns are empty together).
STREAM_REPRESENTATIVE_COLUMN = {
    "trunk_imu": "trunk_gyro_y_dps",
    "cable_drive": "left_cable_tension_n",
    "forearm_sensors": "left_forearm_angle_deg",
}

# Pairs of sensors that should show about the same movement: (left, right, minimum range).
# "minimum range": the pair only counts as "moving" if the more active sensor
# changes by at least this much inside a window.
CHANNEL_PAIRS = [
    ("left_forearm_angle_deg", "right_forearm_angle_deg", 20.0),  # degrees
    ("left_cable_tension_n", "right_cable_tension_n", 15.0),  # newtons
    ("left_cable_travel_mm", "right_cable_travel_mm", 40.0),  # millimeters
]

NEIGHBOR_HALF_WINDOW_SAMPLES = (
    5  # look 5 samples to the left and 5 to the right (0.05 s at 100 Hz)
)
MIN_VALID_NEIGHBORS_PER_SIDE = (
    3  # need 3 real neighbors on EACH side, otherwise we do not judge the sample
)
SPIKE_SIGMA_THRESHOLD = (
    15.0  # a spike is at least 15 "normal noise sizes" away from its neighbors
)
SPIKE_PADDING_SAMPLES = 1  # also blank one sample on each side of a spike
FLATLINE_TOLERANCE = 1e-9  # "the same value" means equal within this
FLATLINE_MIN_DURATION_S = 2.0
AGREEMENT_WINDOW_S = 10.0
AGREEMENT_MIN_RATIO = (
    0.2  # the quieter side must move at least 20% as much as the other
)
MAX_MISSING_FRACTION = (
    0.05  # more than 5% missing in a column means the recording fails
)


def find_dropouts(aligned_df: pd.DataFrame) -> list[dict]:
    times_s = aligned_df["timestamp_s"].to_numpy()
    step_s = float(np.median(np.diff(times_s)))
    dropouts = []
    for stream_name, column in STREAM_REPRESENTATIVE_COLUMN.items():
        is_missing = aligned_df[column].isna().to_numpy()
        for start_index, end_index in find_runs(is_missing):
            dropouts.append(
                {
                    "stream": stream_name,
                    "start_s": round(float(times_s[start_index]), 3),
                    "length_s": round((end_index - start_index) * step_s, 3),
                }
            )
    return dropouts


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start (included) and end (excluded) index of every run of True values."""
    padded = np.concatenate([[False], mask, [False]]).astype(int)
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def value_columns_of(aligned_df: pd.DataFrame) -> list[str]:
    return [column for column in aligned_df.columns if column != "timestamp_s"]


def neighbor_median(values: np.ndarray) -> np.ndarray:
    """For every sample: the median of its neighbors on both sides, NOT counting the sample itself.

    Leaving the sample out matters. If it were included, the median would often be the
    sample itself, and the "distance to the neighbors" would be exactly zero.
    """
    half_window = NEIGHBOR_HALF_WINDOW_SAMPLES
    padded = np.concatenate(
        [np.full(half_window, np.nan), values, np.full(half_window, np.nan)]
    )
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half_window + 1)
    neighbors = np.delete(windows, half_window, axis=1)  # remove the sample itself
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", category=RuntimeWarning
        )  # "all-NaN slice" is fine here
        medians = np.nanmedian(neighbors, axis=1)
    # Next to a hole we only see neighbors on one side, and on a steep slope that is misleading.
    # So we only judge a sample when it has enough real neighbors on both sides.
    valid_left = np.sum(~np.isnan(neighbors[:, :half_window]), axis=1)
    valid_right = np.sum(~np.isnan(neighbors[:, half_window:]), axis=1)
    medians[
        (valid_left < MIN_VALID_NEIGHBORS_PER_SIDE)
        | (valid_right < MIN_VALID_NEIGHBORS_PER_SIDE)
    ] = np.nan
    return medians


def find_spikes(aligned_df: pd.DataFrame, column: str) -> tuple[list[dict], np.ndarray]:
    """Return (spike events, true/false mask of the spiky samples) for one column."""
    values = aligned_df[column].to_numpy()
    times_s = aligned_df["timestamp_s"].to_numpy()
    residual = values - neighbor_median(
        values
    )  # how far is each sample from its neighbors?
    # How big is the normal noise? Exact zeros are left out: a sensor that repeats its value
    # (a flat line) has zero distance everywhere, and tells us nothing about the noise.
    real_residuals = residual[~np.isnan(residual) & (residual != 0)]
    is_spike = np.zeros(len(values), dtype=bool)
    if len(real_residuals) < 100:
        return [], is_spike
    typical_noise = 1.4826 * np.median(
        np.abs(real_residuals - np.median(real_residuals))
    )
    if not typical_noise > 0:
        return [], is_spike
    is_spike = (
        np.abs(residual) > SPIKE_SIGMA_THRESHOLD * typical_noise
    )  # NaN counts as "no spike"

    spikes = []
    for start_index, end_index in find_runs(is_spike):
        worst_index = start_index + int(
            np.argmax(np.abs(residual[start_index:end_index]))
        )
        spikes.append(
            {
                "column": column,
                "timestamp_s": round(float(times_s[worst_index]), 3),
                "jump": round(float(residual[worst_index]), 1),
            }
        )
    return spikes, is_spike


def find_flatlines(aligned_df: pd.DataFrame) -> list[dict]:
    times_s = aligned_df["timestamp_s"].to_numpy()
    step_s = float(np.median(np.diff(times_s)))
    flatlines = []
    for column in value_columns_of(aligned_df):
        values = aligned_df[column].to_numpy()
        is_same_as_previous = np.zeros(len(values), dtype=bool)
        is_same_as_previous[1:] = (
            np.abs(np.diff(values)) <= FLATLINE_TOLERANCE
        )  # NaN gives False
        for start_index, end_index in find_runs(is_same_as_previous):
            duration_s = (end_index - start_index) * step_s
            if duration_s >= FLATLINE_MIN_DURATION_S:
                flatlines.append(
                    {
                        "column": column,
                        "start_s": round(float(times_s[start_index - 1]), 3),
                        "end_s": round(float(times_s[end_index - 1]), 3),
                        "duration_s": round(duration_s, 3),
                    }
                )
    return flatlines


def robust_range(values: np.ndarray) -> float:
    """Spread of the values, ignoring the most extreme 5% on each side (so spikes don't count)."""
    values = values[~np.isnan(values)]
    if len(values) < 10:
        return float("nan")
    return float(np.percentile(values, 95) - np.percentile(values, 5))


def find_disagreements(aligned_df: pd.DataFrame) -> list[dict]:
    times_s = aligned_df["timestamp_s"].to_numpy()
    window_starts_s = np.arange(
        times_s[0], times_s[-1] - AGREEMENT_WINDOW_S + 1e-9, AGREEMENT_WINDOW_S
    )
    if (
        window_starts_s[-1] + AGREEMENT_WINDOW_S < times_s[-1] - 0.05
    ):  # one more window that ends exactly at the end
        window_starts_s = np.append(window_starts_s, times_s[-1] - AGREEMENT_WINDOW_S)
    disagreements = []
    for left_column, right_column, minimum_range in CHANNEL_PAIRS:
        left_values = aligned_df[left_column].to_numpy()
        right_values = aligned_df[right_column].to_numpy()
        bad_windows = []
        for window_start_s in window_starts_s:
            in_window = (times_s >= window_start_s) & (
                times_s < window_start_s + AGREEMENT_WINDOW_S
            )
            left_range = robust_range(left_values[in_window])
            right_range = robust_range(right_values[in_window])
            if np.isnan(left_range) or np.isnan(right_range):
                continue
            larger, smaller = max(left_range, right_range), min(left_range, right_range)
            if larger >= minimum_range and smaller / larger < AGREEMENT_MIN_RATIO:
                weak_side = "right" if right_range < left_range else "left"
                bad_windows.append((float(window_start_s), weak_side))

        # join neighboring bad windows into one event
        for weak_side in ("left", "right"):
            starts = [start for start, side in bad_windows if side == weak_side]
            for run_start, run_end in find_runs(np.isin(window_starts_s, starts)):
                disagreements.append(
                    {
                        "left": left_column,
                        "right": right_column,
                        "weak_side": weak_side,
                        "start_s": round(float(window_starts_s[run_start]), 3),
                        "end_s": round(
                            float(window_starts_s[run_end - 1] + AGREEMENT_WINDOW_S), 3
                        ),
                    }
                )
    return disagreements


def blank_out_spikes(
    aligned_df: pd.DataFrame, spike_masks: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Return a copy where spiky samples (and one neighbor on each side) are NaN."""
    cleaned_df = aligned_df.copy()
    for column, is_spike in spike_masks.items():
        padded = is_spike.copy()
        for shift in range(1, SPIKE_PADDING_SAMPLES + 1):
            padded[shift:] |= is_spike[:-shift]
            padded[:-shift] |= is_spike[shift:]
        cleaned_df.loc[padded, column] = np.nan
    return cleaned_df


def decide_status(
    checks: dict, missing_fraction_per_column: dict[str, float]
) -> tuple[str, list[str]]:
    """pass: nothing found. warn: small problems, repaired. fail: do not use this recording."""
    status = "pass"
    reasons = []
    if checks["dropouts"] or checks["spikes"]:
        status = "warn"
        reasons.append(
            f"{len(checks['dropouts'])} dropouts and {len(checks['spikes'])} spikes, left as missing data"
        )
    for flatline in checks["flatlines"]:
        status = "fail"
        reasons.append(
            f"{flatline['column']} is stuck from {flatline['start_s']} s for {flatline['duration_s']} s"
        )
    for disagreement in checks["disagreements"]:
        status = "fail"
        side_column = disagreement[disagreement["weak_side"]]
        reasons.append(
            f"{side_column} barely moves from {disagreement['start_s']} s to {disagreement['end_s']} s "
            f"while its partner does"
        )
    for column, fraction in missing_fraction_per_column.items():
        if fraction > MAX_MISSING_FRACTION:
            status = "fail"
            reasons.append(f"{column} is {fraction:.0%} missing")
    return status, reasons


def find_healthy_columns(aligned_df: pd.DataFrame, checks: dict) -> list[str]:
    """Columns without a flat line and not on the weak side of a disagreement."""
    broken = {flatline["column"] for flatline in checks["flatlines"]}
    broken |= {
        disagreement[disagreement["weak_side"]]
        for disagreement in checks["disagreements"]
    }
    return [column for column in value_columns_of(aligned_df) if column not in broken]


def run_quality_checks(aligned_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Run all checks. Returns (report, cleaned table)."""
    spikes, spike_masks = [], {}
    for column in value_columns_of(aligned_df):
        column_spikes, is_spike = find_spikes(aligned_df, column)
        spikes += column_spikes
        spike_masks[column] = is_spike

    checks = {
        "dropouts": find_dropouts(
            aligned_df
        ),  # must run before the spikes are blanked out
        "spikes": spikes,
        "flatlines": find_flatlines(aligned_df),
        "disagreements": find_disagreements(aligned_df),
    }
    cleaned_df = blank_out_spikes(aligned_df, spike_masks)
    missing_fraction_per_column = {
        column: float(cleaned_df[column].isna().mean())
        for column in value_columns_of(cleaned_df)
    }
    status, reasons = decide_status(checks, missing_fraction_per_column)
    report = {
        "status": status,
        "reasons": reasons,
        "checks": checks,
        "healthy_columns": find_healthy_columns(aligned_df, checks),
        "missing_fraction_per_column": {
            column: round(fraction, 4)
            for column, fraction in missing_fraction_per_column.items()
        },
    }
    return report, cleaned_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synced", default="data/synced", help="folder with the synced parquet files"
    )
    parser.add_argument("--out", default="data/qc", help="folder for the QC reports")
    parser.add_argument(
        "--clean",
        default="data/clean",
        help="folder for the cleaned tables of usable recordings",
    )
    args = parser.parse_args()

    out_dir, clean_dir = Path(args.out), Path(args.clean)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for parquet_path in sorted(Path(args.synced).glob("*.parquet")):
        recording_id = parquet_path.stem
        aligned_df = pd.read_parquet(parquet_path)
        report, cleaned_df = run_quality_checks(aligned_df)
        report["recording_id"] = recording_id
        (out_dir / f"{recording_id}_qc.json").write_text(json.dumps(report, indent=2))
        if report["status"] != "fail":
            cleaned_df.to_parquet(clean_dir / f"{recording_id}.parquet", index=False)
        summary[recording_id] = {
            "status": report["status"],
            "reasons": report["reasons"],
        }
        print(f"{recording_id}  {report['status']:<5} {'; '.join(report['reasons'])}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    counts = {
        status: sum(1 for item in summary.values() if item["status"] == status)
        for status in ("pass", "warn", "fail")
    }
    print(f"done: {counts['pass']} pass, {counts['warn']} warn, {counts['fail']} fail")


if __name__ == "__main__":
    main()
