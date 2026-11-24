import json

import pandas as pd

from exo.generate import Config, generate_recording, write_recording


def make_small_recording(tmp_path, seed=1):
    config = Config(seed=seed, n_recordings=2, duration_s=30.0, out_dir=str(tmp_path))
    streams, metadata = generate_recording(0, config)
    return config, streams, metadata


def test_files_are_written(tmp_path):
    config, streams, metadata = make_small_recording(tmp_path)
    recording_dir = write_recording(streams, metadata, tmp_path)
    for file_name in [
        "trunk_imu.csv",
        "cable_drive.csv",
        "forearm_sensors.csv",
        "metadata.json",
    ]:
        assert (recording_dir / file_name).exists()
    saved_metadata = json.loads((recording_dir / "metadata.json").read_text())
    assert saved_metadata["worker_id"] == "W001"


def test_sample_counts_match_rates(tmp_path):
    config, streams, _ = make_small_recording(tmp_path)
    assert (
        len(streams["forearm_sensors"]) == 30 * config.forearm_hz
    )  # no dropouts injected here
    assert len(streams["trunk_imu"]) <= 30 * config.trunk_imu_hz  # dropouts remove rows
    assert len(streams["cable_drive"]) <= 30 * config.cable_hz


def test_same_seed_gives_same_data(tmp_path):
    _, first_run, _ = make_small_recording(tmp_path, seed=7)
    _, second_run, _ = make_small_recording(tmp_path, seed=7)
    pd.testing.assert_frame_equal(first_run["cable_drive"], second_run["cable_drive"])


def test_ground_truth_is_recorded(tmp_path):
    _, _, metadata = make_small_recording(tmp_path)
    ground_truth = metadata["ground_truth"]
    assert 0.025 <= ground_truth["cable_offset_s"] <= 0.045
    assert any(fault["type"] == "dropout" for fault in ground_truth["faults"])
