import numpy as np
import pandas as pd

from exo.generate import Config, generate_recording
from exo.qc import run_quality_checks
from exo.segment import drop_short_runs, merge_close_runs, segment_recording
from exo.sync import align_streams


def make_table(lifts, duration_s=60.0, amplitude_mm=200.0, seed=0):
    """A fake clean table. `lifts` is a list of (start_s, duration_s)."""
    random_gen = np.random.default_rng(seed)
    times_s = np.round(np.arange(0.0, duration_s, 0.01), 2)
    lift_level = np.zeros(len(times_s))
    for start_s, lift_duration_s in lifts:
        progress = (times_s - start_s) / lift_duration_s
        is_lifting = (progress > 0) & (progress < 1)
        lift_level[is_lifting] = np.sin(np.pi * progress[is_lifting]) ** 2

    def noise(size):
        return random_gen.normal(0, size, len(times_s))

    return pd.DataFrame(
        {
            "timestamp_s": times_s,
            "trunk_gyro_y_dps": noise(0.5),
            "trunk_acc_x_ms2": 1.0 + noise(0.05),
            "trunk_acc_z_ms2": 9.8 + noise(0.05),
            "left_cable_tension_n": 70 * lift_level + noise(0.7),
            "right_cable_tension_n": 65 * lift_level + noise(0.7),
            "left_cable_travel_mm": amplitude_mm * lift_level + noise(0.3),
            "right_cable_travel_mm": amplitude_mm * lift_level + noise(0.3),
            "left_forearm_angle_deg": 5 + 80 * lift_level + noise(0.3),
            "right_forearm_angle_deg": 5 + 80 * lift_level + noise(0.3),
        }
    )


def test_merge_close_runs_joins_only_close_neighbors():
    runs = [
        (0, 10),
        (15, 20),
        (500, 510),
    ]  # the first two are 0.05 s apart, the third is far away
    assert merge_close_runs(runs, step_s=0.01, max_gap_s=1.0) == [(0, 20), (500, 510)]


def test_drop_short_runs_removes_blips():
    runs = [(0, 50), (100, 300)]  # 0.5 s and 2.0 s
    assert drop_short_runs(runs, step_s=0.01, min_duration_s=1.0) == [(100, 300)]


def test_three_lifts_give_three_episodes_with_padding():
    table = make_table([(5.0, 3.0), (15.0, 3.0), (30.0, 3.0)])
    episodes_df = segment_recording(table, "rec_test")
    assert len(episodes_df) == 3
    assert list(episodes_df["episode_id"]) == [
        "rec_test_ep00",
        "rec_test_ep01",
        "rec_test_ep02",
    ]
    first = episodes_df.iloc[0]
    assert (
        abs((first["lift_start_s"] + first["lift_end_s"]) / 2 - 6.5) < 0.05
    )  # center of the first lift
    assert (
        abs((first["lift_start_s"] - first["start_s"]) - 0.5) < 0.02
    )  # 0.5 s of padding before
    assert abs((first["end_s"] - first["lift_end_s"]) - 0.5) < 0.02  # and after
    assert first["label"] == "lift"
    assert first["missing_fraction"] == 0.0


def test_blips_and_small_bumps_are_not_lifts():
    table = make_table([(5.0, 3.0)])
    table.loc[2000:2029, ["left_cable_travel_mm", "right_cable_travel_mm"]] = (
        50.0  # 0.3 s blip at 20 s
    )
    table.loc[4000:4400, ["left_cable_travel_mm", "right_cable_travel_mm"]] = (
        12.0  # 4 s, but under the threshold
    )
    assert len(segment_recording(table, "rec_test")) == 1


def test_a_short_hole_inside_a_lift_does_not_split_it():
    table = make_table([(5.0, 3.0)])
    table.loc[620:680, ["left_cable_travel_mm", "right_cable_travel_mm"]] = (
        np.nan
    )  # 0.6 s hole near the top
    episodes_df = segment_recording(table, "rec_test")
    assert len(episodes_df) == 1
    assert (
        episodes_df.iloc[0]["missing_fraction"] > 0.0
    )  # but the episode says that it has missing data


def test_padding_is_cut_at_the_end_of_the_recording():
    table = make_table([(56.0, 3.0)], duration_s=60.0)
    episode = segment_recording(table, "rec_test").iloc[0]
    assert episode["end_s"] <= table["timestamp_s"].iloc[-1]


def test_generated_recording_gives_one_episode_per_true_lift():
    config = Config(seed=5, n_recordings=1, duration_s=90.0)
    streams, metadata = generate_recording(0, config)
    aligned_df, _ = align_streams(streams)
    _, cleaned_df = run_quality_checks(aligned_df)
    episodes_df = segment_recording(cleaned_df, "rec_000")

    true_lifts = metadata["ground_truth"]["lifts"]
    assert len(episodes_df) == len(true_lifts)
    true_centers_s = [lift["start_s"] + lift["duration_s"] / 2 for lift in true_lifts]
    found_centers_s = (
        (episodes_df["lift_start_s"] + episodes_df["lift_end_s"]) / 2
    ).to_list()
    for true_center_s, found_center_s in zip(true_centers_s, found_centers_s):
        assert abs(true_center_s - found_center_s) < 0.5
