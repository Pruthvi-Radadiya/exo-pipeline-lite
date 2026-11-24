import shutil

import numpy as np
import pytest

pytest.importorskip("pyspark")
pytestmark = pytest.mark.skipif(
    shutil.which("java") is None, reason="Java is not installed"
)

from pyspark.sql import functions as F

from exo.export import SENSOR_COLUMNS, export_dataset
from exo.generate import Config, generate_recording
from exo.qc import run_quality_checks
from exo.segment import segment_recording
from exo.spark_stats import (
    check_dataset_is_intact,
    compute_episode_stats,
    find_latest_dataset,
    make_spark_session,
    results_match,
    run_pandas_job,
    run_spark_job,
)
from exo.sync import align_streams


@pytest.fixture(scope="module")
def spark():
    session = make_spark_session()
    yield session
    session.stop()


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    """A small exported dataset (one recording), made once for the whole file."""
    root = tmp_path_factory.mktemp("spark_inputs")
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
    return export_dataset(root / "clean", episodes_path, root / "export", repo_dir=root)


def test_spark_treats_nan_as_the_biggest_number_but_our_stats_skip_it(spark):
    columns = ["episode_index", "timestamp"] + SENSOR_COLUMNS
    rows = [
        (0, 0.00, *[1.0] * len(SENSOR_COLUMNS)),
        (0, 0.01, *[float("nan")] * len(SENSOR_COLUMNS)),
        (0, 0.02, *[3.0] * len(SENSOR_COLUMNS)),
    ]
    frames_sdf = spark.createDataFrame(rows, columns)

    raw_maximum = frames_sdf.agg(F.max("left_cable_tension_n")).collect()[0][0]
    assert np.isnan(raw_maximum)  # the trap: a plain Spark max over a NaN gives NaN

    stats = compute_episode_stats(frames_sdf).toPandas().iloc[0]
    assert stats["peak_left_tension_n"] == 3.0
    assert stats["mean_left_forearm_angle_deg"] == 2.0
    assert stats["num_frames"] == 3
    assert stats["num_frames_with_missing_data"] == 1


def test_spark_gives_the_same_numbers_as_pandas(spark, dataset_dir):
    episode_stats_df, _ = run_spark_job(spark, dataset_dir)
    pandas_df = run_pandas_job(dataset_dir)
    assert len(episode_stats_df) == len(pandas_df) > 0
    assert results_match(episode_stats_df, pandas_df)
    assert (
        episode_stats_df["num_frames_with_missing_data"].sum() > 0
    )  # the test data really has gaps


def test_worker_stats_add_up(spark, dataset_dir):
    episode_stats_df, worker_stats_df = run_spark_job(spark, dataset_dir)
    assert list(worker_stats_df["worker_pseudonym"]) == ["Ptest00000000"]
    assert worker_stats_df["num_episodes"].sum() == len(episode_stats_df)
    assert (
        abs(
            worker_stats_df["mean_duration_s"].iloc[0]
            - episode_stats_df["duration_s"].mean()
        )
        < 1e-6
    )


def test_the_newest_dataset_version_is_found(dataset_dir):
    assert find_latest_dataset(dataset_dir.parent) == dataset_dir
    with pytest.raises(ValueError):
        find_latest_dataset(dataset_dir.parent / "does_not_exist")


def test_a_damaged_dataset_is_refused(dataset_dir, tmp_path):
    copy_dir = tmp_path / dataset_dir.name
    shutil.copytree(dataset_dir, copy_dir)
    check_dataset_is_intact(copy_dir)  # fine
    with open(copy_dir / "data/chunk-000/file-000.parquet", "ab") as file:
        file.write(b"x")
    with pytest.raises(ValueError, match="manifest"):
        check_dataset_is_intact(copy_dir)
