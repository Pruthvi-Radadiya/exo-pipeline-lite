"""Step 5: cut every usable recording into episodes, one episode per lift.

We look at how much cable the motors have pulled in. When it is above a small
threshold, the worker is lifting. Every stretch above the threshold becomes one
episode, with a little padding before and after.

Run:  python -m exo.segment --clean data/clean --out data/episodes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from exo.qc import find_runs, value_columns_of

ACTIVITY_COLUMNS = [
    "left_cable_travel_mm",
    "right_cable_travel_mm",
]  # clean signals: no spikes, no bias
LIFT_THRESHOLD_MM = 20.0  # above this much pulled-in cable, the worker is lifting
MERGE_GAP_S = (
    1.0  # two stretches closer than this are the same lift (a short hole, a tiny dip)
)
MIN_LIFT_DURATION_S = 1.0  # shorter than this is a blip, not a lift
PADDING_S = 0.5  # context before and after every lift
LABEL = "lift"


def make_activity_signal(clean_df: pd.DataFrame) -> np.ndarray:
    """How much cable is pulled in (mm). Average of left and right. NaN where both are missing."""
    return clean_df[ACTIVITY_COLUMNS].mean(axis=1).to_numpy()


def merge_close_runs(
    runs: list[tuple[int, int]], step_s: float, max_gap_s: float
) -> list[tuple[int, int]]:
    """Join runs that are separated by less than max_gap_s."""
    merged: list[tuple[int, int]] = []
    for start_index, end_index in runs:
        if merged and (start_index - merged[-1][1]) * step_s < max_gap_s:
            merged[-1] = (merged[-1][0], end_index)
        else:
            merged.append((start_index, end_index))
    return merged


def drop_short_runs(
    runs: list[tuple[int, int]], step_s: float, min_duration_s: float
) -> list[tuple[int, int]]:
    return [
        (start, end) for start, end in runs if (end - start) * step_s >= min_duration_s
    ]


def segment_recording(clean_df: pd.DataFrame, recording_id: str) -> pd.DataFrame:
    """Return one row per episode (one per lift) of this recording."""
    times_s = clean_df["timestamp_s"].to_numpy()
    step_s = float(np.median(np.diff(times_s)))
    activity_mm = make_activity_signal(clean_df)

    is_lifting = activity_mm > LIFT_THRESHOLD_MM  # NaN counts as "not lifting"
    runs = find_runs(is_lifting)
    runs = merge_close_runs(runs, step_s, MERGE_GAP_S)
    runs = drop_short_runs(runs, step_s, MIN_LIFT_DURATION_S)

    value_columns = value_columns_of(clean_df)
    episodes = []
    for episode_index, (start_index, end_index) in enumerate(runs):
        lift_start_s = float(times_s[start_index])
        lift_end_s = float(times_s[end_index - 1])
        start_s = max(float(times_s[0]), lift_start_s - PADDING_S)
        end_s = min(float(times_s[-1]), lift_end_s + PADDING_S)
        in_episode = (times_s >= start_s) & (times_s <= end_s)
        window_df = clean_df[in_episode]
        episodes.append(
            {
                "episode_id": f"{recording_id}_ep{episode_index:02d}",
                "recording_id": recording_id,
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "lift_start_s": round(lift_start_s, 3),
                "lift_end_s": round(lift_end_s, 3),
                "peak_left_tension_n": round(
                    float(window_df["left_cable_tension_n"].max()), 1
                ),
                "peak_right_tension_n": round(
                    float(window_df["right_cable_tension_n"].max()), 1
                ),
                "peak_travel_mm": round(float(np.nanmax(activity_mm[in_episode])), 1),
                "missing_fraction": round(
                    float(window_df[value_columns].isna().to_numpy().mean()), 4
                ),
                "label": LABEL,
            }
        )
    return pd.DataFrame(episodes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean", default="data/clean", help="folder with the cleaned tables"
    )
    parser.add_argument(
        "--out", default="data/episodes", help="folder for the episode table"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = []
    for parquet_path in sorted(Path(args.clean).glob("*.parquet")):
        episodes_df = segment_recording(
            pd.read_parquet(parquet_path), parquet_path.stem
        )
        tables.append(episodes_df)
        print(f"{parquet_path.stem}  {len(episodes_df)} episodes")

    all_episodes_df = pd.concat(tables, ignore_index=True)
    all_episodes_df.to_parquet(out_dir / "episodes.parquet", index=False)
    all_episodes_df.to_csv(out_dir / "episodes.csv", index=False)
    print(
        f"done: {len(all_episodes_df)} episodes from {len(tables)} recordings in {out_dir}"
    )


if __name__ == "__main__":
    main()
