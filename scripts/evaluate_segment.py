"""Compare the episodes we found with the lifts we injected in step 1.

Run:  python scripts/evaluate_segment.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

MATCH_TOLERANCE_S = (
    0.5  # a found lift counts as correct if its center is this close to the true center
)


def evaluate_recording(episodes_df: pd.DataFrame, true_lifts: list[dict]) -> dict:
    true_centers_s = [lift["start_s"] + lift["duration_s"] / 2 for lift in true_lifts]
    true_durations_s = [lift["duration_s"] for lift in true_lifts]
    found_centers_s = (
        (episodes_df["lift_start_s"] + episodes_df["lift_end_s"]) / 2
    ).to_list()
    found_durations_s = (
        episodes_df["lift_end_s"] - episodes_df["lift_start_s"]
    ).to_list()

    unmatched_true = list(range(len(true_centers_s)))
    center_errors_s, duration_ratios = [], []
    for found_center_s, found_duration_s in zip(found_centers_s, found_durations_s):
        close = [
            i
            for i in unmatched_true
            if abs(true_centers_s[i] - found_center_s) <= MATCH_TOLERANCE_S
        ]
        if close:
            best = min(close, key=lambda i: abs(true_centers_s[i] - found_center_s))
            unmatched_true.remove(best)
            center_errors_s.append(found_center_s - true_centers_s[best])
            duration_ratios.append(found_duration_s / true_durations_s[best])
    found_correctly = len(center_errors_s)
    return {
        "true": len(true_lifts),
        "found": found_correctly,
        "false_alarms": len(found_centers_s) - found_correctly,
        "missed": len(unmatched_true),
        "center_errors_s": center_errors_s,
        "duration_ratios": duration_ratios,
    }


def main() -> None:
    all_episodes_df = pd.read_parquet("data/episodes/episodes.parquet")
    totals = {"true": 0, "found": 0, "false_alarms": 0, "missed": 0}
    all_center_errors_s, all_duration_ratios = [], []

    print("recording  true lifts  episodes found  false alarms  missed")
    for recording_id, episodes_df in all_episodes_df.groupby("recording_id"):
        ground_truth = json.loads(
            Path(f"data/raw/{recording_id}/metadata.json").read_text()
        )["ground_truth"]
        result = evaluate_recording(episodes_df, ground_truth["lifts"])
        print(
            f"{recording_id}   {result['true']:>9}  {result['found']:>14}  {result['false_alarms']:>12}  {result['missed']:>6}"
        )
        for key in totals:
            totals[key] += result[key]
        all_center_errors_s += result["center_errors_s"]
        all_duration_ratios += result["duration_ratios"]

    print()
    print(
        f"lifts: {totals['true']} true, {totals['found']} found, {totals['false_alarms']} false alarms, {totals['missed']} missed"
    )
    print(
        f"center time error: median {np.median(np.abs(all_center_errors_s)) * 1000:.0f} ms, worst {np.max(np.abs(all_center_errors_s)) * 1000:.0f} ms"
    )
    print(
        f"found lift length compared with the true lift length: {np.mean(all_duration_ratios):.0%} on average"
    )


if __name__ == "__main__":
    main()
