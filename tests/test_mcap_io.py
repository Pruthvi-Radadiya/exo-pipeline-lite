import numpy as np
import pandas as pd

from exo.generate import Config, generate_recording, write_recording
from exo.mcap_io import describe_mcap, read_mcap, write_mcap


def make_mcap(tmp_path):
    config = Config(seed=3, n_recordings=1, duration_s=20.0, out_dir=str(tmp_path))
    streams, metadata = generate_recording(0, config)
    recording_dir = write_recording(streams, metadata, tmp_path / "raw")
    mcap_path = tmp_path / "mcap" / "rec_000.mcap"
    write_mcap(recording_dir, mcap_path)
    return recording_dir, mcap_path


def test_round_trip_keeps_every_stream(tmp_path):
    recording_dir, mcap_path = make_mcap(tmp_path)
    streams, _ = read_mcap(mcap_path)
    assert set(streams) == {"trunk_imu", "cable_drive", "forearm_sensors"}
    for stream_name, stream_df in streams.items():
        csv_df = pd.read_csv(recording_dir / f"{stream_name}.csv")
        assert len(stream_df) == len(csv_df)
        assert list(stream_df.columns) == list(csv_df.columns)
        np.testing.assert_allclose(stream_df.to_numpy(), csv_df.to_numpy(), atol=1e-6)


def test_file_metadata_has_ids_but_no_ground_truth(tmp_path):
    _, mcap_path = make_mcap(tmp_path)
    _, file_metadata = read_mcap(mcap_path)
    assert file_metadata["worker_id"] == "W001"
    assert file_metadata["device_serial"] == "LU20-1000"
    assert "ground_truth" not in file_metadata


def test_describe_runs(tmp_path, capsys):
    _, mcap_path = make_mcap(tmp_path)
    describe_mcap(mcap_path)
    assert "/cable_drive" in capsys.readouterr().out
