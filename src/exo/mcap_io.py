"""Step 2: write recordings into MCAP files and read them back.

Run:  python -m exo.mcap_io --raw data/raw --out data/mcap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from mcap.reader import make_reader
from mcap.writer import Writer

STREAM_NAMES = ["trunk_imu", "cable_drive", "forearm_sensors"]
NANOSECONDS_PER_SECOND = 1_000_000_000

# Only these fields of metadata.json go into the MCAP file.
# ground_truth stays out: a real device does not know its own clock error.
FILE_METADATA_KEYS = [
    "recording_id",
    "worker_id",
    "device_serial",
    "site",
    "recorded_at",
    "task",
]


def seconds_to_nanoseconds(time_s: float) -> int:
    return int(round(time_s * NANOSECONDS_PER_SECOND))


def make_json_schema(column_names: list[str]) -> dict:
    """Describe a message: an object where every field is a number."""
    return {
        "type": "object",
        "properties": {column_name: {"type": "number"} for column_name in column_names},
    }


def write_mcap(recording_dir: Path, mcap_path: Path) -> None:
    """Read the CSV files of one recording and write one MCAP file."""
    recording_dir = Path(recording_dir)
    mcap_path = Path(mcap_path)
    mcap_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((recording_dir / "metadata.json").read_text())

    with open(mcap_path, "wb") as mcap_file:
        writer = Writer(mcap_file)
        writer.start()

        for stream_name in STREAM_NAMES:
            stream_df = pd.read_csv(recording_dir / f"{stream_name}.csv")
            value_columns = [
                column for column in stream_df.columns if column != "timestamp_s"
            ]

            schema_id = writer.register_schema(
                name=f"exo/{stream_name}",
                encoding="jsonschema",
                data=json.dumps(make_json_schema(value_columns)).encode(),
            )
            channel_id = writer.register_channel(
                topic=f"/{stream_name}",
                message_encoding="json",
                schema_id=schema_id,
            )

            timestamps_s = stream_df["timestamp_s"].to_numpy()
            value_rows = stream_df[value_columns].to_dict("records")
            for time_s, value_row in zip(timestamps_s, value_rows):
                log_time_ns = seconds_to_nanoseconds(time_s)
                writer.add_message(
                    channel_id=channel_id,
                    log_time=log_time_ns,
                    publish_time=log_time_ns,
                    data=json.dumps(value_row).encode(),
                )

        file_metadata = {key: str(metadata[key]) for key in FILE_METADATA_KEYS}
        writer.add_metadata(name="recording", data=file_metadata)
        writer.finish()


def read_mcap(mcap_path: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Load an MCAP file back into one DataFrame per stream, plus the file metadata."""
    rows_by_stream: dict[str, list[dict]] = {}
    file_metadata: dict[str, str] = {}

    with open(mcap_path, "rb") as mcap_file:
        reader = make_reader(mcap_file)
        for _schema, channel, message in reader.iter_messages():
            stream_name = channel.topic.lstrip("/")
            row = json.loads(message.data)
            row["timestamp_s"] = message.log_time / NANOSECONDS_PER_SECOND
            rows_by_stream.setdefault(stream_name, []).append(row)
        for metadata_record in reader.iter_metadata():
            if metadata_record.name == "recording":
                file_metadata = dict(metadata_record.metadata)

    streams = {}
    for stream_name, rows in rows_by_stream.items():
        stream_df = pd.DataFrame(rows)
        value_columns = [
            column for column in stream_df.columns if column != "timestamp_s"
        ]
        streams[stream_name] = stream_df[["timestamp_s"] + value_columns]
    return streams, file_metadata


def describe_mcap(mcap_path: Path) -> None:
    """Print topics, message counts and the time range (like a mini `mcap info`)."""
    with open(mcap_path, "rb") as mcap_file:
        summary = make_reader(mcap_file).get_summary()
    statistics = summary.statistics
    start_s = statistics.message_start_time / NANOSECONDS_PER_SECOND
    end_s = statistics.message_end_time / NANOSECONDS_PER_SECOND
    print(
        f"{mcap_path}  messages={statistics.message_count}  time={start_s:.2f}s to {end_s:.2f}s"
    )
    for channel_id, channel in summary.channels.items():
        count = statistics.channel_message_counts[channel_id]
        print(f"  {channel.topic:<18} {count:>6} messages")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", default="data/raw", help="folder with rec_000, rec_001, ..."
    )
    parser.add_argument("--out", default="data/mcap", help="folder for the .mcap files")
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    recording_dirs = sorted(
        path for path in raw_dir.iterdir() if (path / "metadata.json").exists()
    )
    for recording_dir in recording_dirs:
        mcap_path = out_dir / f"{recording_dir.name}.mcap"
        write_mcap(recording_dir, mcap_path)
        size_mb = mcap_path.stat().st_size / 1_000_000
        print(f"{mcap_path}  {size_mb:.1f} MB")
    print(f"done: {len(recording_dirs)} files in {out_dir}")


if __name__ == "__main__":
    main()
