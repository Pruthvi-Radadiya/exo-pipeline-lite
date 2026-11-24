"""Compare the estimated clocks and biases with the ground truth of every recording.

Run:  python scripts/evaluate_sync.py
"""

import json
from pathlib import Path

import numpy as np

STREAM_TRUTH_KEYS = {
    "cable_drive": ("cable_offset_s", "cable_drift_ppm"),
    "forearm_sensors": ("forearm_offset_s", "forearm_drift_ppm"),
}


def evaluate_recording(report_path: Path, raw_dir: Path) -> dict:
    report = json.loads(report_path.read_text())
    ground_truth = json.loads(
        (raw_dir / report["recording_id"] / "metadata.json").read_text()
    )["ground_truth"]
    times_s = np.linspace(0, 120, 121)

    result = {"recording_id": report["recording_id"]}
    for stream_name, (offset_key, drift_key) in STREAM_TRUTH_KEYS.items():
        estimated = report["clocks"][stream_name]
        estimated_offset_s = (
            estimated["offset_s"] + estimated["drift_ppm"] * 1e-6 * times_s
        )
        true_offset_s = (
            ground_truth[offset_key] + ground_truth[drift_key] * 1e-6 * times_s
        )
        result[f"{stream_name}_max_error_ms"] = float(
            np.max(np.abs(estimated_offset_s - true_offset_s)) * 1000
        )
        result[f"{stream_name}_drift_error_ppm"] = float(
            estimated["drift_ppm"] - ground_truth[drift_key]
        )
    result["bias_max_error"] = max(
        abs(report["bias"][name] - ground_truth["bias"][name])
        for name in report["bias"]
    )
    return result


def main() -> None:
    raw_dir = Path("data/raw")
    results = [
        evaluate_recording(path, raw_dir)
        for path in sorted(Path("data/synced").glob("*_report.json"))
    ]

    print(
        "recording   cable error [ms]  forearm error [ms]  cable drift err [ppm]  bias error"
    )
    for result in results:
        print(
            f"{result['recording_id']}   {result['cable_drive_max_error_ms']:14.2f}"
            f"  {result['forearm_sensors_max_error_ms']:18.2f}"
            f"  {result['cable_drive_drift_error_ppm']:20.1f}  {result['bias_max_error']:10.3f}"
        )
    print(
        f"worst case: cable {max(r['cable_drive_max_error_ms'] for r in results):.2f} ms, "
        f"forearm {max(r['forearm_sensors_max_error_ms'] for r in results):.2f} ms, "
        f"bias {max(r['bias_max_error'] for r in results):.3f}"
    )


if __name__ == "__main__":
    main()
