"""Step 7: export the usable, anonymized episodes as a versioned dataset.

Layout (inspired by LeRobotDataset v3.0, see the tutorial for what is and is not the same):

    exo_dataset_v0001/
      data/chunk-000/file-000.parquet        one row per frame, many episodes per file
      meta/info.json                         schema, units, frame rate
      meta/stats.json                        mean, std, min, max of every signal
      meta/tasks.parquet                     task index to task text
      meta/episodes/chunk-000/file-000.parquet   one row per episode, with where its frames are
      manifest.json                          versions, inputs, and a SHA-256 hash of every file

Every run writes a NEW version folder. Old versions are never changed.

Run:  python -m exo.export --clean data/clean --episodes data/anonymized/episodes.parquet --out data/export
      python -m exo.export --verify data/export/exo_dataset_v0001
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_NAME = "exo_dataset"
SCHEMA_VERSION = "1.0"  # change this when columns or formats change
FPS = 100
ROWS_PER_FILE = 50_000  # start a new data file after this many frames
FILES_PER_CHUNK = 1000
MAX_MISSING_FRACTION = (
    0.05  # episodes with more missing data than this are not exported
)
TIME_TOLERANCE_S = 0.0005  # times in the episode table are rounded to 1 ms
LABEL_TO_TASK = {"lift": "lift a parcel with exoskeleton support"}

INDEX_COLUMNS = ["index", "episode_index", "frame_index", "timestamp", "task_index"]
SENSOR_COLUMN_INFO = {
    "trunk_gyro_y_dps": (
        "deg/s",
        "trunk rotation speed around the y axis (backpack IMU), bias removed",
    ),
    "trunk_acc_x_ms2": ("m/s^2", "trunk acceleration, x axis (backpack IMU)"),
    "trunk_acc_z_ms2": ("m/s^2", "trunk acceleration, z axis (backpack IMU)"),
    "left_cable_tension_n": ("N", "pull force of the left cable, bias removed"),
    "right_cable_tension_n": ("N", "pull force of the right cable, bias removed"),
    "left_cable_travel_mm": ("mm", "how much left cable the motor has pulled in"),
    "right_cable_travel_mm": ("mm", "how much right cable the motor has pulled in"),
    "left_forearm_angle_deg": (
        "deg",
        "how high the left forearm is raised (about 5 at rest)",
    ),
    "right_forearm_angle_deg": (
        "deg",
        "how high the right forearm is raised (about 5 at rest)",
    ),
}
SENSOR_COLUMNS = list(SENSOR_COLUMN_INFO)
EPISODE_COLUMNS = [
    "episode_index",
    "episode_id",
    "recording_id",
    "worker_pseudonym",
    "recorded_month",
    "task_index",
    "length",
    "dataset_from_index",
    "dataset_to_index",
    "data_chunk_index",
    "data_file_index",
    "start_s",
    "end_s",
    "peak_left_tension_n",
    "peak_right_tension_n",
    "peak_travel_mm",
    "missing_fraction",
]


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_pipeline_version(repo_dir: Path) -> dict:
    """The git commit of the code that made the dataset, and whether there are uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {"commit": commit, "dirty": bool(changes)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": True}


def next_version_number(export_root: Path) -> int:
    """1 if there is no version yet, otherwise the highest version plus 1."""
    numbers = []
    if export_root.exists():
        for folder in export_root.iterdir():
            match = re.fullmatch(rf"{DATASET_NAME}_v(\d{{4}})", folder.name)
            if match:
                numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def select_episodes(
    episodes_df: pd.DataFrame, max_missing_fraction: float
) -> tuple[pd.DataFrame, int]:
    """Keep episodes with little missing data. Returns (kept, number excluded)."""
    is_good = episodes_df["missing_fraction"] <= max_missing_fraction
    return episodes_df[is_good].reset_index(drop=True), int((~is_good).sum())


def assign_data_files(lengths: list[int], rows_per_file: int) -> list[int]:
    """File number for every episode. An episode is never split over two files."""
    file_indices, current_file, rows_in_file = [], 0, 0
    for length in lengths:
        if rows_in_file > 0 and rows_in_file + length > rows_per_file:
            current_file += 1
            rows_in_file = 0
        file_indices.append(current_file)
        rows_in_file += length
    return file_indices


def cut_frames(clean_df: pd.DataFrame, start_s: float, end_s: float) -> pd.DataFrame:
    """The rows of one episode, with its own clock (timestamp 0 at the start of the episode)."""
    times_s = clean_df["timestamp_s"].to_numpy()
    in_episode = (times_s >= start_s - TIME_TOLERANCE_S) & (
        times_s <= end_s + TIME_TOLERANCE_S
    )
    window_df = clean_df[in_episode]
    frames_df = window_df[SENSOR_COLUMNS].reset_index(drop=True)
    frames_df.insert(
        0,
        "timestamp",
        np.round(
            window_df["timestamp_s"].to_numpy() - window_df["timestamp_s"].iloc[0], 3
        ),
    )
    return frames_df


def build_frames_and_episodes(
    clean_dir: Path, episodes_df: pd.DataFrame, rows_per_file: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (all frames, one row per episode)."""
    labels = sorted(episodes_df["label"].unique())
    task_index_of = {label: index for index, label in enumerate(labels)}

    clean_tables: dict[str, pd.DataFrame] = {}
    frame_tables, lengths = [], []
    for episode_index, episode in enumerate(episodes_df.itertuples(index=False)):
        if episode.recording_id not in clean_tables:
            clean_tables[episode.recording_id] = pd.read_parquet(
                Path(clean_dir) / f"{episode.recording_id}.parquet"
            )
        frames_df = cut_frames(
            clean_tables[episode.recording_id], episode.start_s, episode.end_s
        )
        frames_df.insert(0, "frame_index", np.arange(len(frames_df)))
        frames_df.insert(0, "episode_index", episode_index)
        frames_df["task_index"] = task_index_of[episode.label]
        frame_tables.append(frames_df)
        lengths.append(len(frames_df))

    all_frames_df = pd.concat(frame_tables, ignore_index=True)
    all_frames_df.insert(0, "index", np.arange(len(all_frames_df)))
    all_frames_df = all_frames_df[INDEX_COLUMNS + SENSOR_COLUMNS]
    for column in SENSOR_COLUMNS:
        all_frames_df[column] = all_frames_df[column].astype("float32")

    file_indices = assign_data_files(lengths, rows_per_file)
    ends = np.cumsum(lengths)
    meta_df = episodes_df.copy()
    meta_df["episode_index"] = np.arange(len(meta_df))
    meta_df["task_index"] = meta_df["label"].map(task_index_of)
    meta_df["length"] = lengths
    meta_df["dataset_to_index"] = ends
    meta_df["dataset_from_index"] = ends - np.array(lengths)
    meta_df["data_chunk_index"] = [
        file_index // FILES_PER_CHUNK for file_index in file_indices
    ]
    meta_df["data_file_index"] = [
        file_index % FILES_PER_CHUNK for file_index in file_indices
    ]
    return all_frames_df, meta_df[EPISODE_COLUMNS].reset_index(drop=True)


def compute_stats(frames_df: pd.DataFrame) -> dict:
    """Mean, std, min, max and count of every signal. Missing values are skipped."""
    stats = {}
    for column in SENSOR_COLUMNS:
        values = frames_df[column]
        stats[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
            "count": int(values.notna().sum()),
        }
    return stats


def build_info(
    version_name: str,
    frames_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    task_count: int,
    rows_per_file: int,
) -> dict:
    features = {
        "index": {
            "dtype": "int64",
            "description": "number of the frame in the whole dataset",
        },
        "episode_index": {"dtype": "int64", "description": "number of the episode"},
        "frame_index": {
            "dtype": "int64",
            "description": "number of the frame inside its episode",
        },
        "timestamp": {
            "dtype": "float64",
            "unit": "s",
            "description": "seconds since the start of the episode",
        },
        "task_index": {"dtype": "int64", "description": "row of meta/tasks.parquet"},
    }
    for column, (unit, description) in SENSOR_COLUMN_INFO.items():
        features[column] = {
            "dtype": "float32",
            "unit": unit,
            "description": description,
        }
    return {
        "dataset_name": DATASET_NAME,
        "dataset_version": version_name,
        "schema_version": SCHEMA_VERSION,
        "layout_note": "inspired by LeRobotDataset v3.0, not tested with the lerobot library",
        "fps": FPS,
        "total_episodes": len(meta_df),
        "total_frames": len(frames_df),
        "total_tasks": task_count,
        "rows_per_file": rows_per_file,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "episodes_path": "meta/episodes/chunk-000/file-000.parquet",
        "features": features,
        "missing_values": "signals can contain NaN where a device sent no data",
    }


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, sort_keys=False))


def list_files(dataset_dir: Path) -> list[str]:
    """Every file of the dataset (relative paths with /), except the manifest itself."""
    return sorted(
        path.relative_to(dataset_dir).as_posix()
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )


def verify_dataset(dataset_dir: Path) -> list[str]:
    """Check the dataset against its manifest. An empty list means: nothing changed."""
    dataset_dir = Path(dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    problems = []
    for relative_path, expected in manifest["files"].items():
        path = dataset_dir / relative_path
        if not path.exists():
            problems.append(f"missing file: {relative_path}")
        elif sha256_of_file(path) != expected["sha256"]:
            problems.append(f"changed file: {relative_path}")
    for relative_path in set(list_files(dataset_dir)) - set(manifest["files"]):
        problems.append(f"file that is not in the manifest: {relative_path}")
    return problems


def export_dataset(
    clean_dir: Path,
    episodes_path: Path,
    export_root: Path,
    repo_dir: Path = Path("."),
    rows_per_file: int = ROWS_PER_FILE,
    max_missing_fraction: float = MAX_MISSING_FRACTION,
) -> Path:
    """Write a new dataset version. Returns the folder."""
    clean_dir, episodes_path, export_root = (
        Path(clean_dir),
        Path(episodes_path),
        Path(export_root),
    )
    episodes_in_df = pd.read_parquet(episodes_path)
    episodes_df, excluded_count = select_episodes(episodes_in_df, max_missing_fraction)
    if episodes_df.empty:
        raise ValueError("no episodes to export")
    frames_df, meta_df = build_frames_and_episodes(
        clean_dir, episodes_df, rows_per_file
    )

    version_name = f"{DATASET_NAME}_v{next_version_number(export_root):04d}"
    dataset_dir = export_root / version_name
    dataset_dir.mkdir(
        parents=True, exist_ok=False
    )  # never overwrite an existing version

    # data files
    episode_file_numbers = (
        meta_df["data_chunk_index"] * FILES_PER_CHUNK + meta_df["data_file_index"]
    ).to_numpy()
    frame_file_indices = np.repeat(
        episode_file_numbers, meta_df["length"].to_numpy()
    )  # one file number per frame
    for file_number in sorted(set(frame_file_indices)):
        path = (
            dataset_dir
            / f"data/chunk-{file_number // FILES_PER_CHUNK:03d}/file-{file_number % FILES_PER_CHUNK:03d}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        frames_df[frame_file_indices == file_number].to_parquet(path, index=False)

    # meta files
    labels = sorted(episodes_df["label"].unique())
    tasks_df = pd.DataFrame(
        {
            "task_index": range(len(labels)),
            "task": [LABEL_TO_TASK.get(label, label) for label in labels],
        }
    )
    (dataset_dir / "meta/episodes/chunk-000").mkdir(parents=True)
    meta_df.to_parquet(
        dataset_dir / "meta/episodes/chunk-000/file-000.parquet", index=False
    )
    tasks_df.to_parquet(dataset_dir / "meta/tasks.parquet", index=False)
    write_json(dataset_dir / "meta/stats.json", compute_stats(frames_df))
    write_json(
        dataset_dir / "meta/info.json",
        build_info(version_name, frames_df, meta_df, len(labels), rows_per_file),
    )

    # manifest: written last, so it can list the hash of everything else
    recording_ids = sorted(meta_df["recording_id"].unique())
    manifest = {
        "dataset_name": DATASET_NAME,
        "dataset_version": version_name,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline": get_pipeline_version(repo_dir),
        "settings": {
            "fps": FPS,
            "rows_per_file": rows_per_file,
            "max_missing_fraction": max_missing_fraction,
        },
        "counts": {
            "episodes_in": len(episodes_in_df),
            "episodes_excluded_missing_data": excluded_count,
            "episodes_exported": len(meta_df),
            "frames_exported": len(frames_df),
            "recordings_used": len(recording_ids),
        },
        "inputs": {
            "episodes_table": {
                "path": episodes_path.as_posix(),
                "sha256": sha256_of_file(episodes_path),
            },
            "clean_tables": {
                f"{recording_id}.parquet": sha256_of_file(
                    clean_dir / f"{recording_id}.parquet"
                )
                for recording_id in recording_ids
            },
        },
        "files": {
            relative_path: {
                "sha256": sha256_of_file(dataset_dir / relative_path),
                "bytes": (dataset_dir / relative_path).stat().st_size,
            }
            for relative_path in list_files(dataset_dir)
        },
    }
    write_json(dataset_dir / "manifest.json", manifest)
    return dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--clean", default="data/clean")
    parser.add_argument("--episodes", default="data/anonymized/episodes.parquet")
    parser.add_argument("--out", default="data/export")
    parser.add_argument(
        "--verify",
        help="check an existing dataset folder against its manifest and stop",
    )
    args = parser.parse_args()

    if args.verify:
        problems = verify_dataset(Path(args.verify))
        print("OK: nothing changed" if not problems else "\n".join(problems))
        raise SystemExit(1 if problems else 0)

    dataset_dir = export_dataset(Path(args.clean), Path(args.episodes), Path(args.out))
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in ("dataset_version", "schema_version", "pipeline", "counts")
            },
            indent=2,
        )
    )
    print(f"files: {len(manifest['files'])}")
    print(f"written to {dataset_dir}")


if __name__ == "__main__":
    main()
