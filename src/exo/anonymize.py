"""Step 6: pseudonymization and consent handling.

What goes in:   the episode table (step 5), the identifying metadata stored in the
                MCAP files (step 2), and a consent table.
What comes out: an episode table without anything that points to a person, plus a
                separate private key table. Episodes of workers without consent are
                left out.

Run:  export EXO_PSEUDONYM_KEY="a long secret text"
      python -m exo.anonymize --mcap data/mcap --episodes data/episodes/episodes.parquet \\
          --consent data/private/consent.csv --out data/anonymized --private data/private
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path

import pandas as pd
from mcap.reader import make_reader

IDENTIFYING_COLUMNS = ["worker_id", "device_serial", "site", "recorded_at"]
SECRET_KEY_ENV_NAME = "EXO_PSEUDONYM_KEY"
MIN_SECRET_KEY_LENGTH = 16
PSEUDONYM_PREFIX = "P"
PSEUDONYM_HEX_LENGTH = 12


def read_secret_key() -> str:
    secret_key = os.environ.get(SECRET_KEY_ENV_NAME, "")
    if len(secret_key) < MIN_SECRET_KEY_LENGTH:
        raise ValueError(
            f"set the environment variable {SECRET_KEY_ENV_NAME} to a secret text of at least "
            f"{MIN_SECRET_KEY_LENGTH} characters"
        )
    return secret_key


def make_pseudonym(worker_id: str, secret_key: str) -> str:
    """The same worker and the same key always give the same pseudonym. Without the key you cannot compute it."""
    if len(secret_key) < MIN_SECRET_KEY_LENGTH:
        raise ValueError("the secret key is too short")
    digest = hmac.new(
        secret_key.encode(), worker_id.encode(), hashlib.sha256
    ).hexdigest()
    return PSEUDONYM_PREFIX + digest[:PSEUDONYM_HEX_LENGTH]


def to_recorded_month(recorded_at: str) -> str:
    """'2026-10-08T08:00:00Z' becomes '2026-10': the exact day is not needed."""
    return recorded_at[:7]


def read_recording_metadata(mcap_path: Path) -> dict[str, str]:
    """Only the small metadata record of an MCAP file, not the messages."""
    with open(mcap_path, "rb") as mcap_file:
        for metadata_record in make_reader(mcap_file).iter_metadata():
            if metadata_record.name == "recording":
                return dict(metadata_record.metadata)
    raise ValueError(f"no recording metadata in {mcap_path}")


def build_identity_registry(mcap_dir: Path) -> pd.DataFrame:
    """One row per recording with everything that identifies a person. This table stays private."""
    rows = []
    for mcap_path in sorted(Path(mcap_dir).glob("*.mcap")):
        metadata = read_recording_metadata(mcap_path)
        rows.append(
            {
                "recording_id": metadata["recording_id"],
                **{column: metadata[column] for column in IDENTIFYING_COLUMNS},
            }
        )
    return pd.DataFrame(rows)


def apply_consent(
    episodes_df: pd.DataFrame, consent_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split episodes into (with consent, without consent). No consent record means no consent."""
    consent_df = consent_df.copy()
    consent_df["consent_given"] = (
        consent_df["consent_given"].astype(str).str.lower().isin(["true", "1", "yes"])
    )
    merged_df = episodes_df.merge(
        consent_df[["worker_id", "consent_given"]], on="worker_id", how="left"
    )
    has_consent = merged_df["consent_given"].fillna(False).astype(bool)
    return merged_df[has_consent].drop(columns="consent_given"), merged_df[
        ~has_consent
    ].drop(columns="consent_given")


def find_leaked_identifiers(
    exported_df: pd.DataFrame, identity_registry_df: pd.DataFrame
) -> list[str]:
    """Look for any identifying text in any cell or column name of the exported table."""
    secret_texts = set()
    for column in IDENTIFYING_COLUMNS:
        secret_texts.update(identity_registry_df[column].astype(str).unique())

    leaks = []
    for column in exported_df.columns:
        cell_texts = exported_df[column].astype(str)
        for secret_text in sorted(secret_texts):
            if (
                secret_text in column
                or cell_texts.str.contains(secret_text, regex=False).any()
            ):
                leaks.append(f"{secret_text!r} appears in column {column!r}")
    return leaks


def anonymize(
    episodes_df: pd.DataFrame,
    identity_registry_df: pd.DataFrame,
    consent_df: pd.DataFrame,
    secret_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (anonymized episodes, key table, report)."""
    with_identity_df = episodes_df.merge(
        identity_registry_df, on="recording_id", how="left"
    )
    if with_identity_df["worker_id"].isna().any():
        raise ValueError(
            "some episodes belong to a recording without identity metadata"
        )

    kept_df, excluded_df = apply_consent(with_identity_df, consent_df)

    kept_df = kept_df.copy()
    kept_df["worker_pseudonym"] = kept_df["worker_id"].map(
        lambda worker_id: make_pseudonym(worker_id, secret_key)
    )
    kept_df["recorded_month"] = kept_df["recorded_at"].map(to_recorded_month)

    key_table_df = (
        kept_df[["worker_pseudonym", "worker_id"]]
        .drop_duplicates()
        .sort_values("worker_pseudonym")
        .reset_index(drop=True)
    )
    anonymized_df = kept_df.drop(columns=IDENTIFYING_COLUMNS)
    front_columns = ["episode_id", "recording_id", "worker_pseudonym", "recorded_month"]
    anonymized_df = anonymized_df[
        front_columns + [c for c in anonymized_df.columns if c not in front_columns]
    ]

    leaks = find_leaked_identifiers(anonymized_df, identity_registry_df)
    if leaks:
        raise ValueError("identifying data in the export: " + "; ".join(leaks))

    report = {
        "episodes_in": len(episodes_df),
        "episodes_without_consent": len(excluded_df),
        "episodes_exported": len(anonymized_df),
        "workers_exported": len(key_table_df),
        "workers_without_consent": int(excluded_df["worker_id"].nunique()),
        "removed_columns": IDENTIFYING_COLUMNS,
        "leaked_identifiers": len(leaks),
    }
    return anonymized_df.reset_index(drop=True), key_table_df, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mcap", default="data/mcap")
    parser.add_argument("--episodes", default="data/episodes/episodes.parquet")
    parser.add_argument("--consent", default="data/private/consent.csv")
    parser.add_argument(
        "--out", default="data/anonymized", help="folder that may be shared"
    )
    parser.add_argument(
        "--private", default="data/private", help="folder that must NEVER be shared"
    )
    args = parser.parse_args()

    out_dir, private_dir = Path(args.out), Path(args.private)
    out_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    identity_registry_df = build_identity_registry(Path(args.mcap))
    anonymized_df, key_table_df, report = anonymize(
        pd.read_parquet(args.episodes),
        identity_registry_df,
        pd.read_csv(args.consent),
        read_secret_key(),
    )

    identity_registry_df.to_csv(private_dir / "identity_registry.csv", index=False)
    key_table_df.to_csv(private_dir / "key_table.csv", index=False)
    anonymized_df.to_parquet(out_dir / "episodes.parquet", index=False)
    anonymized_df.to_csv(out_dir / "episodes.csv", index=False)
    (out_dir / "anonymization_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"shared:  {out_dir}")
    print(f"private: {private_dir}  (never share this folder)")


if __name__ == "__main__":
    main()
