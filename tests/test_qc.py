import numpy as np
import pandas as pd

from exo.generate import Config, generate_recording
from exo.qc import (
    find_disagreements,
    find_dropouts,
    find_flatlines,
    find_runs,
    find_spikes,
    neighbor_median,
    run_quality_checks,
)
from exo.sync import align_streams


def make_table(duration_s=40.0, seed=0):
    """A clean fake table with all 9 columns: a lift every 8 seconds, plus a little noise."""
    random_gen = np.random.default_rng(seed)
    times_s = np.round(np.arange(0.0, duration_s, 0.01), 2)
    lift_level = np.clip(np.sin(2 * np.pi * times_s / 8), 0, None) ** 2

    def noise(size):
        return random_gen.normal(0, size, len(times_s))

    return pd.DataFrame(
        {
            "timestamp_s": times_s,
            "trunk_gyro_y_dps": 20 * np.gradient(lift_level, times_s) + noise(0.5),
            "trunk_acc_x_ms2": 1.0 + noise(0.05),
            "trunk_acc_z_ms2": 9.8 + noise(0.05),
            "left_cable_tension_n": 70 * lift_level + noise(0.7),
            "right_cable_tension_n": 65 * lift_level + noise(0.7),
            "left_cable_travel_mm": 200 * lift_level + noise(0.3),
            "right_cable_travel_mm": 200 * lift_level + noise(0.3),
            "left_forearm_angle_deg": 5 + 80 * lift_level + noise(0.3),
            "right_forearm_angle_deg": 5 + 80 * lift_level + noise(0.3),
        }
    )


def test_find_runs():
    assert find_runs(np.array([False, True, True, False, True])) == [(1, 3), (4, 5)]
    assert find_runs(np.array([False, False])) == []
    assert find_runs(np.array([True, True, True])) == [(0, 3)]


def test_neighbor_median_does_not_count_the_sample_itself():
    values = np.linspace(0.0, 10.0, 41)
    values[20] = 500.0  # a spike
    assert (
        abs(neighbor_median(values)[20] - 5.0) < 0.3
    )  # the neighbors still say "about 5"


def test_one_spike_is_found_and_a_clean_table_has_none():
    table = make_table()
    clean_spikes, _ = find_spikes(table, "left_cable_tension_n")
    assert clean_spikes == []
    table.loc[1500, "left_cable_tension_n"] += 200.0  # at 15.0 s
    spikes, is_spike = find_spikes(table, "left_cable_tension_n")
    assert len(spikes) == 1
    assert abs(spikes[0]["timestamp_s"] - 15.0) < 0.02
    assert is_spike.sum() == 1


def test_a_dead_flat_half_does_not_create_false_spikes():
    table = make_table()
    table.loc[2000:, "right_forearm_angle_deg"] = 5.0  # stuck from 20 s on
    spikes, _ = find_spikes(table, "right_forearm_angle_deg")
    assert spikes == []


def test_flatline_needs_to_be_long_enough():
    table = make_table()
    table.loc[1000:1099, "right_cable_travel_mm"] = 3.0  # 1 s: too short to count
    assert find_flatlines(table) == []
    table.loc[2000:2299, "right_cable_travel_mm"] = 3.0  # 3 s: a stuck sensor
    flatlines = find_flatlines(table)
    assert len(flatlines) == 1
    assert flatlines[0]["column"] == "right_cable_travel_mm"
    assert abs(flatlines[0]["start_s"] - 20.0) < 0.02


def test_dropout_is_found_with_its_stream():
    table = make_table()
    table.loc[1000:1049, ["left_cable_tension_n", "right_cable_tension_n"]] = (
        np.nan
    )  # 0.5 s hole
    dropouts = find_dropouts(table)
    assert len(dropouts) == 1
    assert dropouts[0]["stream"] == "cable_drive"
    assert abs(dropouts[0]["length_s"] - 0.5) < 0.02


def test_disagreement_is_found_when_one_arm_stops_moving():
    table = make_table()
    assert find_disagreements(table) == []
    table.loc[2000:, "right_forearm_angle_deg"] = 5.0
    disagreements = find_disagreements(table)
    assert len(disagreements) == 1
    assert disagreements[0]["weak_side"] == "right"
    assert disagreements[0]["right"] == "right_forearm_angle_deg"


def test_decisions_pass_warn_fail():
    clean_report, _ = run_quality_checks(make_table())
    assert clean_report["status"] == "pass"

    spiky_table = make_table()
    spiky_table.loc[1500, "left_cable_tension_n"] += 200.0
    spiky_report, cleaned_df = run_quality_checks(spiky_table)
    assert spiky_report["status"] == "warn"
    assert np.isnan(
        cleaned_df.loc[1500, "left_cable_tension_n"]
    )  # the spike is blanked out
    assert not np.isnan(
        spiky_table.loc[1500, "left_cable_tension_n"]
    )  # the input is untouched

    stuck_table = make_table()
    stuck_table.loc[2000:, "right_forearm_angle_deg"] = 5.0
    stuck_report, _ = run_quality_checks(stuck_table)
    assert stuck_report["status"] == "fail"
    assert "right_forearm_angle_deg" not in stuck_report["healthy_columns"]
    assert "left_forearm_angle_deg" in stuck_report["healthy_columns"]


def test_generated_recordings_get_the_expected_decision():
    config = Config(seed=5, n_recordings=1, duration_s=90.0)
    healthy_streams, _ = generate_recording(0, config)
    healthy_report, _ = run_quality_checks(align_streams(healthy_streams)[0])
    assert (
        healthy_report["status"] == "warn"
    )  # dropouts and spikes were injected, but nothing is broken

    stuck_streams, _ = generate_recording(
        7, config
    )  # recording 7 gets a stuck right forearm sensor
    stuck_report, _ = run_quality_checks(align_streams(stuck_streams)[0])
    assert stuck_report["status"] == "fail"
    assert stuck_report["checks"]["flatlines"][0]["column"] == "right_forearm_angle_deg"
