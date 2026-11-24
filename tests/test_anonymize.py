import re

import pandas as pd
import pytest

from exo.anonymize import (
    anonymize,
    apply_consent,
    build_identity_registry,
    find_leaked_identifiers,
    make_pseudonym,
    read_secret_key,
    to_recorded_month,
)
from exo.generate import Config, generate_recording, write_recording
from exo.mcap_io import write_mcap

SECRET_KEY = "a-long-secret-text-for-the-tests"


def make_registry():
    return pd.DataFrame(
        {
            "recording_id": ["rec_a", "rec_b", "rec_c", "rec_d"],
            "worker_id": ["W001", "W002", "W003", "W001"],
            "device_serial": ["LU20-1", "LU20-2", "LU20-3", "LU20-4"],
            "site": [
                "Warehouse Munich",
                "Hub Hamburg",
                "Plant Ingolstadt",
                "Warehouse Munich",
            ],
            "recorded_at": [
                "2026-10-01T08:00:00Z",
                "2026-10-02T08:00:00Z",
                "2026-10-03T08:00:00Z",
                "2026-11-04T08:00:00Z",
            ],
        }
    )


def make_episodes():
    return pd.DataFrame(
        {
            "episode_id": ["rec_a_ep00", "rec_b_ep00", "rec_c_ep00", "rec_d_ep00"],
            "recording_id": ["rec_a", "rec_b", "rec_c", "rec_d"],
            "start_s": [1.0, 2.0, 3.0, 4.0],
            "label": ["lift"] * 4,
        }
    )


def make_consent():
    return pd.DataFrame(
        {
            "worker_id": ["W001", "W002", "W003"],
            "consent_given": [True, True, False],
            "changed_on": ["2026-09-15"] * 3,
        }
    )


def test_pseudonym_is_stable_and_hides_the_worker():
    pseudonym = make_pseudonym("W001", SECRET_KEY)
    assert pseudonym == make_pseudonym(
        "W001", SECRET_KEY
    )  # same worker, same key: same pseudonym
    assert re.fullmatch(r"P[0-9a-f]{12}", pseudonym)
    assert "W001" not in pseudonym
    assert pseudonym != make_pseudonym("W002", SECRET_KEY)  # different workers differ
    assert pseudonym != make_pseudonym(
        "W001", SECRET_KEY + "-other"
    )  # a different key gives different pseudonyms


def test_short_or_missing_keys_are_rejected(monkeypatch):
    with pytest.raises(ValueError):
        make_pseudonym("W001", "short")
    monkeypatch.delenv("EXO_PSEUDONYM_KEY", raising=False)
    with pytest.raises(ValueError):
        read_secret_key()
    monkeypatch.setenv("EXO_PSEUDONYM_KEY", SECRET_KEY)
    assert read_secret_key() == SECRET_KEY


def test_recorded_at_becomes_a_month():
    assert to_recorded_month("2026-10-08T08:00:00Z") == "2026-10"


def test_consent_gate_keeps_only_workers_with_consent():
    episodes_df = make_episodes().merge(
        make_registry()[["recording_id", "worker_id"]], on="recording_id"
    )
    episodes_df.loc[len(episodes_df)] = [
        "rec_x_ep00",
        "rec_x",
        9.0,
        "lift",
        "W999",
    ]  # a worker with no consent record at all
    kept_df, excluded_df = apply_consent(episodes_df, make_consent())
    assert set(kept_df["worker_id"]) == {"W001", "W002"}
    assert set(excluded_df["worker_id"]) == {"W003", "W999"}


def test_leak_finder_finds_identifiers_in_cells_and_column_names():
    registry_df = make_registry()
    leaky_df = pd.DataFrame({"episode_id": ["W001_ep00"], "Hub Hamburg": [1]})
    leaks = find_leaked_identifiers(leaky_df, registry_df)
    assert any("W001" in leak for leak in leaks)
    assert any("Hub Hamburg" in leak for leak in leaks)
    assert find_leaked_identifiers(make_episodes(), registry_df) == []


def test_anonymize_removes_identifiers_and_applies_consent():
    anonymized_df, key_table_df, report = anonymize(
        make_episodes(), make_registry(), make_consent(), SECRET_KEY
    )

    assert list(anonymized_df["recording_id"]) == [
        "rec_a",
        "rec_b",
        "rec_d",
    ]  # rec_c had no consent
    for identifying_column in ["worker_id", "device_serial", "site", "recorded_at"]:
        assert identifying_column not in anonymized_df.columns
    assert list(anonymized_df["recorded_month"]) == ["2026-10", "2026-10", "2026-11"]

    pseudonyms = list(anonymized_df["worker_pseudonym"])
    assert pseudonyms[0] == pseudonyms[2]  # rec_a and rec_d are the same worker
    assert pseudonyms[0] != pseudonyms[1]

    assert set(key_table_df["worker_id"]) == {
        "W001",
        "W002",
    }  # no pseudonym for the worker without consent
    assert report["episodes_in"] == 4
    assert report["episodes_exported"] == 3
    assert report["episodes_without_consent"] == 1
    assert report["workers_exported"] == 2
    assert report["leaked_identifiers"] == 0


def test_anonymize_refuses_to_export_when_an_identifier_hides_in_another_column():
    episodes_df = make_episodes()
    episodes_df["episode_id"] = [
        "W001_rec_a_ep00",
        "rec_b_ep00",
        "rec_c_ep00",
        "rec_d_ep00",
    ]  # a worker id inside an id
    with pytest.raises(ValueError, match="identifying data"):
        anonymize(episodes_df, make_registry(), make_consent(), SECRET_KEY)


def test_identity_registry_is_read_from_the_mcap_files(tmp_path):
    config = Config(seed=3, n_recordings=1, duration_s=20.0)
    streams, metadata = generate_recording(0, config)
    recording_dir = write_recording(streams, metadata, tmp_path / "raw")
    write_mcap(recording_dir, tmp_path / "mcap" / "rec_000.mcap")

    registry_df = build_identity_registry(tmp_path / "mcap")
    assert list(registry_df.columns) == [
        "recording_id",
        "worker_id",
        "device_serial",
        "site",
        "recorded_at",
    ]
    assert registry_df.iloc[0]["worker_id"] == "W001"
    assert registry_df.iloc[0]["device_serial"] == "LU20-1000"
