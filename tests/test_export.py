import json

import numpy as np
import pandas as pd
import pytest

from exo.anonymize import IDENTIFYING_COLUMNS
from exo.export import (
    MAX_MISSING_FRACTION,
    SENSOR_COLUMNS,
    assign_data_files,
    cut_frames,
    export_dataset,
    get_pipeline_version,
    next_version_number,
    select_episodes,
    sha256_of_file,
    verify_dataset,
)
from exo.generate import Config, generate_recording
from exo.qc import run_quality_checks
from exo.segment import segment_recording
from exo.sync import align_streams


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """One cleaned recording and its episode table, made once for all tests in this file."""
    root = tmp_path_factory.mktemp("export_inputs")
    streams, _ = generate_recording(0, Config(seed=5, n_recordings=1, duration_s=90.0))
    aligned_df, _ = align_streams(streams)
    _, cleaned_df = run_quality_checks(aligned_df)
    (root / "clean").mkdir()
    cleaned_df.to_parquet(root / "clean" / "rec_000.parquet", index=False)
    episodes_df = segment_recording(cleaned_df, "rec_000")
    episodes_df["worker_pseudonym"] = "Ptest00000000"
    episodes_df["recorded_month"] = "2026-10"
    episodes_path = root / "episodes.parquet"
    episodes_df.to_parquet(episodes_path, index=False)
    return root / "clean", episodes_path, episodes_df


def test_sha256_of_a_known_text(tmp_path):
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    assert (
        sha256_of_file(path)
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_next_version_number(tmp_path):
    assert next_version_number(tmp_path / "does_not_exist_yet") == 1
    for name in [
        "exo_dataset_v0001",
        "exo_dataset_v0003",
        "something_else",
        "exo_dataset_vABCD",
    ]:
        (tmp_path / name).mkdir()
    assert next_version_number(tmp_path) == 4


def test_episodes_are_never_split_over_two_files():
    assert assign_data_files([400, 400, 400], rows_per_file=1000) == [0, 0, 1]
    assert assign_data_files([1500, 100], rows_per_file=1000) == [
        0,
        1,
    ]  # one long episode still fits in one file


def test_select_episodes_drops_episodes_with_too_much_missing_data():
    episodes_df = pd.DataFrame(
        {"episode_id": ["a", "b", "c"], "missing_fraction": [0.0, 0.05, 0.2]}
    )
    kept_df, excluded_count = select_episodes(episodes_df, max_missing_fraction=0.05)
    assert list(kept_df["episode_id"]) == ["a", "b"]
    assert excluded_count == 1


def test_cut_frames_gives_an_episode_with_its_own_clock():
    times_s = np.round(np.arange(0.0, 20.0, 0.01), 2)
    table = pd.DataFrame(
        {"timestamp_s": times_s, **{column: 1.0 for column in SENSOR_COLUMNS}}
    )
    frames_df = cut_frames(table, start_s=5.0, end_s=7.5)
    assert len(frames_df) == 251  # 5.00 up to and including 7.50
    assert frames_df["timestamp"].iloc[0] == 0.0
    assert abs(frames_df["timestamp"].iloc[-1] - 2.5) < 1e-9
    np.testing.assert_allclose(np.diff(frames_df["timestamp"]), 0.01, atol=1e-9)


def test_export_writes_consistent_files(prepared, tmp_path):
    clean_dir, episodes_path, episodes_df = prepared
    dataset_dir = export_dataset(
        clean_dir, episodes_path, tmp_path / "export", repo_dir=tmp_path
    )

    for relative_path in [
        "data/chunk-000/file-000.parquet",
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
        "manifest.json",
    ]:
        assert (dataset_dir / relative_path).exists()

    frames_df = pd.read_parquet(dataset_dir / "data/chunk-000/file-000.parquet")
    meta_df = pd.read_parquet(dataset_dir / "meta/episodes/chunk-000/file-000.parquet")
    info = json.loads((dataset_dir / "meta/info.json").read_text())

    kept_df, _ = select_episodes(
        episodes_df, MAX_MISSING_FRACTION
    )  # episodes with too much missing data are left out
    assert len(meta_df) == len(kept_df)
    assert info["total_frames"] == len(frames_df) == int(meta_df["length"].sum())
    np.testing.assert_array_equal(
        frames_df["index"], np.arange(len(frames_df))
    )  # numbered 0, 1, 2, ...
    assert (
        (meta_df["dataset_to_index"] - meta_df["dataset_from_index"])
        == meta_df["length"]
    ).all()
    for column in SENSOR_COLUMNS:
        assert frames_df[column].dtype == np.float32
    for episode in meta_df.itertuples():  # every episode finds its own frames
        rows = frames_df.iloc[episode.dataset_from_index : episode.dataset_to_index]
        assert (rows["episode_index"] == episode.episode_index).all()


def test_big_datasets_are_split_into_several_files(prepared, tmp_path):
    clean_dir, episodes_path, _ = prepared
    dataset_dir = export_dataset(
        clean_dir,
        episodes_path,
        tmp_path / "export",
        repo_dir=tmp_path,
        rows_per_file=700,
    )
    data_files = sorted((dataset_dir / "data/chunk-000").glob("file-*.parquet"))
    assert len(data_files) >= 2
    episodes_per_file = [
        set(pd.read_parquet(path)["episode_index"]) for path in data_files
    ]
    for first, second in zip(episodes_per_file, episodes_per_file[1:]):
        assert first.isdisjoint(second)  # no episode appears in two files


def test_verify_finds_changed_missing_and_unlisted_files(prepared, tmp_path):
    clean_dir, episodes_path, _ = prepared
    dataset_dir = export_dataset(
        clean_dir, episodes_path, tmp_path / "export", repo_dir=tmp_path
    )
    assert verify_dataset(dataset_dir) == []

    with open(dataset_dir / "data/chunk-000/file-000.parquet", "ab") as file:
        file.write(b"x")
    (dataset_dir / "meta/tasks.parquet").unlink()
    (dataset_dir / "notes.txt").write_text("stray file")
    problems = verify_dataset(dataset_dir)
    assert "changed file: data/chunk-000/file-000.parquet" in problems
    assert "missing file: meta/tasks.parquet" in problems
    assert "file that is not in the manifest: notes.txt" in problems


def test_every_export_is_a_new_version_with_the_same_data(prepared, tmp_path):
    clean_dir, episodes_path, _ = prepared
    first_dir = export_dataset(
        clean_dir, episodes_path, tmp_path / "export", repo_dir=tmp_path
    )
    second_dir = export_dataset(
        clean_dir, episodes_path, tmp_path / "export", repo_dir=tmp_path
    )
    assert first_dir.name.endswith("_v0001") and second_dir.name.endswith("_v0002")
    assert verify_dataset(first_dir) == []  # the old version was not touched

    first_files = json.loads((first_dir / "manifest.json").read_text())["files"]
    second_files = json.loads((second_dir / "manifest.json").read_text())["files"]
    assert (
        first_files["data/chunk-000/file-000.parquet"]["sha256"]
        == second_files["data/chunk-000/file-000.parquet"]["sha256"]
    )


def test_episodes_with_too_much_missing_data_are_left_out_and_counted(
    prepared, tmp_path
):
    clean_dir, _, episodes_df = prepared
    kept_df, already_excluded_count = select_episodes(episodes_df, MAX_MISSING_FRACTION)
    damaged_df = episodes_df.copy()
    first_good_row = episodes_df.index[
        episodes_df["missing_fraction"] <= MAX_MISSING_FRACTION
    ][0]
    damaged_df.loc[first_good_row, "missing_fraction"] = (
        0.5  # damage one episode that was good before
    )
    damaged_path = tmp_path / "damaged_episodes.parquet"
    damaged_df.to_parquet(damaged_path, index=False)

    dataset_dir = export_dataset(
        clean_dir, damaged_path, tmp_path / "export", repo_dir=tmp_path
    )
    counts = json.loads((dataset_dir / "manifest.json").read_text())["counts"]
    assert counts["episodes_excluded_missing_data"] == already_excluded_count + 1
    assert counts["episodes_exported"] == len(kept_df) - 1


def test_nothing_that_identifies_a_person_is_exported(prepared, tmp_path):
    clean_dir, episodes_path, _ = prepared
    dataset_dir = export_dataset(
        clean_dir, episodes_path, tmp_path / "export", repo_dir=tmp_path
    )
    for parquet_path in dataset_dir.rglob("*.parquet"):
        columns = set(pd.read_parquet(parquet_path).columns)
        assert columns.isdisjoint(IDENTIFYING_COLUMNS)


def test_pipeline_version_outside_of_git_is_unknown(tmp_path):
    assert get_pipeline_version(tmp_path) == {"commit": "unknown", "dirty": True}
