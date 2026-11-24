"""Step 9: statistics per episode with PySpark, in local mode (no cluster).

Reads the newest exported dataset (step 7) and computes, for every episode:
duration, number of frames, peak cable tensions, mean forearm angles and how many frames have missing data.
Then it averages these numbers per worker pseudonym.

Run:  python -m exo.spark_stats --export data/export --out data/stats --compare
"""

from __future__ import annotations

import argparse
import time
from functools import reduce
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from exo.export import DATASET_NAME, SENSOR_COLUMNS, next_version_number, verify_dataset

ABSOLUTE_TOLERANCE = (
    1e-3  # for comparing Spark and pandas results (the signals are float32)
)


def find_latest_dataset(export_root: Path) -> Path:
    latest_number = next_version_number(Path(export_root)) - 1
    if latest_number < 1:
        raise ValueError(f"no dataset version in {export_root}")
    return Path(export_root) / f"{DATASET_NAME}_v{latest_number:04d}"


def check_dataset_is_intact(dataset_dir: Path) -> None:
    problems = verify_dataset(dataset_dir)
    if problems:
        raise ValueError(
            "the dataset does not match its manifest: " + "; ".join(problems)
        )


def make_spark_session() -> SparkSession:
    spark = (
        SparkSession.builder.master(
            "local[*]"
        )  # use all cores of this computer, no cluster
        .appName("exo_episode_stats")
        .config(
            "spark.sql.shuffle.partitions", "4"
        )  # the default is 200, far too many for a small dataset
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def nan_to_null(column_name: str):
    """Spark counts NaN as a value (and as the biggest number!). pandas skips it. We turn it into null, which Spark skips.

    (In our Parquet files, missing values are stored as null already. But a NaN can come from other sources,
    so we make sure. It costs nothing.)
    """
    return F.when(F.isnan(column_name), F.lit(None)).otherwise(F.col(column_name))


def is_missing(column_name: str):
    return F.col(column_name).isNull() | F.isnan(column_name)


def compute_episode_stats(frames_sdf: DataFrame) -> DataFrame:
    """One row per episode, computed from the frames (a Spark DataFrame)."""
    any_signal_missing = reduce(
        lambda left, right: left | right,
        [is_missing(column) for column in SENSOR_COLUMNS],
    )
    return frames_sdf.groupBy("episode_index").agg(
        F.count("*").alias("num_frames"),
        F.max("timestamp").alias("duration_s"),
        F.max(nan_to_null("left_cable_tension_n")).alias("peak_left_tension_n"),
        F.max(nan_to_null("right_cable_tension_n")).alias("peak_right_tension_n"),
        F.avg(nan_to_null("left_forearm_angle_deg")).alias(
            "mean_left_forearm_angle_deg"
        ),
        F.avg(nan_to_null("right_forearm_angle_deg")).alias(
            "mean_right_forearm_angle_deg"
        ),
        F.sum(F.when(any_signal_missing, 1).otherwise(0)).alias(
            "num_frames_with_missing_data"
        ),
    )


def compute_worker_stats(episode_stats_sdf: DataFrame) -> DataFrame:
    """One row per worker pseudonym."""
    return (
        episode_stats_sdf.groupBy("worker_pseudonym")
        .agg(
            F.count("*").alias("num_episodes"),
            F.avg("duration_s").alias("mean_duration_s"),
            F.avg("peak_left_tension_n").alias("mean_peak_left_tension_n"),
            F.max("peak_left_tension_n").alias("max_peak_left_tension_n"),
        )
        .orderBy("worker_pseudonym")
    )


def run_spark_job(
    spark: SparkSession, dataset_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (episode stats, worker stats) as small pandas tables."""
    frames_sdf = spark.read.parquet(str(dataset_dir / "data/chunk-*/file-*.parquet"))
    episodes_sdf = spark.read.parquet(
        str(dataset_dir / "meta/episodes/chunk-*/file-*.parquet")
    ).select(
        "episode_index",
        "episode_id",
        "recording_id",
        "worker_pseudonym",
        "recorded_month",
    )
    episode_stats_sdf = compute_episode_stats(frames_sdf).join(
        episodes_sdf, on="episode_index", how="inner"
    )
    episode_stats_sdf = episode_stats_sdf.orderBy("episode_index")
    worker_stats_sdf = compute_worker_stats(episode_stats_sdf)
    return episode_stats_sdf.toPandas(), worker_stats_sdf.toPandas()


def run_pandas_job(dataset_dir: Path) -> pd.DataFrame:
    """The same episode statistics with plain pandas. Used to check Spark, and to compare the time."""
    frame_paths = sorted((dataset_dir / "data").glob("chunk-*/file-*.parquet"))
    frames_df = pd.concat(
        [pd.read_parquet(path) for path in frame_paths], ignore_index=True
    )
    meta_paths = sorted((dataset_dir / "meta/episodes").glob("chunk-*/file-*.parquet"))
    meta_df = pd.concat(
        [pd.read_parquet(path) for path in meta_paths], ignore_index=True
    )

    grouped = frames_df.groupby("episode_index")
    stats_df = pd.DataFrame(
        {
            "num_frames": grouped.size(),
            "duration_s": grouped["timestamp"].max(),
            "peak_left_tension_n": grouped["left_cable_tension_n"].max(),
            "peak_right_tension_n": grouped["right_cable_tension_n"].max(),
            "mean_left_forearm_angle_deg": grouped["left_forearm_angle_deg"].mean(),
            "mean_right_forearm_angle_deg": grouped["right_forearm_angle_deg"].mean(),
            "num_frames_with_missing_data": frames_df[SENSOR_COLUMNS]
            .isna()
            .any(axis=1)
            .groupby(frames_df["episode_index"])
            .sum(),
        }
    ).reset_index()
    merged_df = stats_df.merge(
        meta_df[
            [
                "episode_index",
                "episode_id",
                "recording_id",
                "worker_pseudonym",
                "recorded_month",
            ]
        ],
        on="episode_index",
    )
    return merged_df.sort_values("episode_index").reset_index(drop=True)


def results_match(spark_df: pd.DataFrame, pandas_df: pd.DataFrame) -> bool:
    columns = [column for column in pandas_df.columns if column in spark_df.columns]
    spark_sorted = spark_df.sort_values("episode_index").reset_index(drop=True)[columns]
    pandas_sorted = pandas_df.sort_values("episode_index").reset_index(drop=True)[
        columns
    ]
    for column in columns:
        if pd.api.types.is_numeric_dtype(pandas_sorted[column]):
            differences = (
                spark_sorted[column].astype(float) - pandas_sorted[column].astype(float)
            ).abs()
            if not (differences <= ABSOLUTE_TOLERANCE).all():
                return False
        elif not (
            spark_sorted[column] == pandas_sorted[column]
        ).all():  # text columns must be identical
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export", default="data/export", help="folder with the dataset versions"
    )
    parser.add_argument("--out", default="data/stats")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="also run pandas, compare results and time",
    )
    args = parser.parse_args()

    dataset_dir = find_latest_dataset(Path(args.export))
    check_dataset_is_intact(dataset_dir)
    print(f"dataset: {dataset_dir.name}")

    start = time.perf_counter()
    spark = make_spark_session()
    session_ready = time.perf_counter()
    episode_stats_df, worker_stats_df = run_spark_job(spark, dataset_dir)
    spark_done = time.perf_counter()
    spark.stop()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_stats_df.to_parquet(out_dir / "episode_stats.parquet", index=False)
    episode_stats_df.to_csv(out_dir / "episode_stats.csv", index=False)
    worker_stats_df.to_csv(out_dir / "worker_stats.csv", index=False)
    print(
        f"{len(episode_stats_df)} episodes, {len(worker_stats_df)} workers written to {out_dir}"
    )
    print(
        f"spark: {session_ready - start:.1f} s to start, {spark_done - session_ready:.1f} s for the job"
    )

    if args.compare:
        pandas_start = time.perf_counter()
        pandas_df = run_pandas_job(dataset_dir)
        pandas_seconds = time.perf_counter() - pandas_start
        print(f"pandas: {pandas_seconds:.2f} s for the same job")
        print(f"same results: {results_match(episode_stats_df, pandas_df)}")


if __name__ == "__main__":
    main()
