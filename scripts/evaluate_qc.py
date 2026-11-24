"""Score the QC reports against the faults we injected in step 1.

Run:  python scripts/evaluate_qc.py
"""

import json
from pathlib import Path

MATCH_TOLERANCE_S = (
    0.15  # a found problem counts as correct if it is this close to the injected one
)


def match(
    found_times_s: list[float], true_times_s: list[float]
) -> tuple[int, int, int]:
    """Return (true positives, false positives, false negatives)."""
    unmatched_true = list(true_times_s)
    true_positives = 0
    for found_time_s in found_times_s:
        close = [
            t for t in unmatched_true if abs(t - found_time_s) <= MATCH_TOLERANCE_S
        ]
        if close:
            unmatched_true.remove(close[0])
            true_positives += 1
    return true_positives, len(found_times_s) - true_positives, len(unmatched_true)


def evaluate_recording(report: dict, ground_truth: dict) -> dict:
    faults = ground_truth["faults"]
    checks = report["checks"]

    true_spikes = [f["timestamp_s"] for f in faults if f["type"] == "spike"]
    found_spikes = [s["timestamp_s"] for s in checks["spikes"]]
    spike_result = match(found_spikes, true_spikes)

    dropout_result = [0, 0, 0]
    for stream_name in ("trunk_imu", "cable_drive", "forearm_sensors"):
        true_starts = [
            f["start_s"]
            for f in faults
            if f["type"] == "dropout" and f["stream"] == stream_name
        ]
        found_starts = [
            d["start_s"] for d in checks["dropouts"] if d["stream"] == stream_name
        ]
        dropout_result = [
            a + b for a, b in zip(dropout_result, match(found_starts, true_starts))
        ]

    has_true_flatline = any(f["type"] == "flatline" for f in faults)
    return {
        "recording_id": report["recording_id"],
        "spikes": spike_result,
        "dropouts": tuple(dropout_result),
        "flatline_found": len(checks["flatlines"]) > 0,
        "disagreement_found": len(checks["disagreements"]) > 0,
        "flatline_true": has_true_flatline,
        "status": report["status"],
    }


def main() -> None:
    results = []
    for report_path in sorted(Path("data/qc").glob("*_qc.json")):
        report = json.loads(report_path.read_text())
        ground_truth = json.loads(
            Path(f"data/raw/{report['recording_id']}/metadata.json").read_text()
        )["ground_truth"]
        results.append(evaluate_recording(report, ground_truth))

    print(
        "recording  spikes (found/false/missed)  dropouts (found/false/missed)  stuck sensor     status"
    )
    for result in results:
        stuck = "found" if result["flatline_found"] else "-"
        expected = " (expected)" if result["flatline_true"] else ""
        print(
            f"{result['recording_id']}   {result['spikes'][0]:>6}/{result['spikes'][1]}/{result['spikes'][2]}"
            f"                  {result['dropouts'][0]:>6}/{result['dropouts'][1]}/{result['dropouts'][2]}"
            f"                  {stuck + expected:<16} {result['status']}"
        )

    totals = {
        "spikes": [sum(r["spikes"][i] for r in results) for i in range(3)],
        "dropouts": [sum(r["dropouts"][i] for r in results) for i in range(3)],
    }
    wrong_stuck = [
        r["recording_id"] for r in results if r["flatline_found"] != r["flatline_true"]
    ]
    wrong_disagree = [
        r["recording_id"]
        for r in results
        if r["disagreement_found"] != r["flatline_true"]
    ]
    print()
    print(
        f"spikes:   found {totals['spikes'][0]}, false alarms {totals['spikes'][1]}, missed {totals['spikes'][2]}"
    )
    print(
        f"dropouts: found {totals['dropouts'][0]}, false alarms {totals['dropouts'][1]}, missed {totals['dropouts'][2]}"
    )
    print(
        f"stuck sensor wrong in: {wrong_stuck or 'no recording'}; sensors-disagree wrong in: {wrong_disagree or 'no recording'}"
    )


if __name__ == "__main__":
    main()
