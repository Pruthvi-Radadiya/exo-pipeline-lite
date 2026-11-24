"""Show the episodes found in one recording, and zoom into one of them.

Run:  python scripts/plot_segments.py rec_000 0
      python scripts/plot_segments.py rec_005 8     (a lift that overlaps a dropout)
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from exo.segment import LIFT_THRESHOLD_MM, make_activity_signal


def plot_segments(recording_id: str, zoom_episode_index: int) -> None:
    clean_df = pd.read_parquet(f"data/clean/{recording_id}.parquet")
    episodes_df = pd.read_parquet("data/episodes/episodes.parquet")
    episodes_df = episodes_df[episodes_df["recording_id"] == recording_id].reset_index(
        drop=True
    )
    times_s = clean_df["timestamp_s"]
    activity_mm = make_activity_signal(clean_df)

    figure, (overview_axis, zoom_axis) = plt.subplots(2, 1, figsize=(11, 8))

    # 1. the whole recording with all episodes
    overview_axis.plot(
        times_s, activity_mm, label="cable pulled in (average of left and right)"
    )
    overview_axis.axhline(
        LIFT_THRESHOLD_MM,
        color="tab:red",
        linestyle="--",
        label=f"lift threshold ({LIFT_THRESHOLD_MM:.0f} mm)",
    )
    for index, episode in episodes_df.iterrows():
        overview_axis.axvspan(
            episode["start_s"], episode["end_s"], color="tab:green", alpha=0.25
        )
        overview_axis.text(
            (episode["start_s"] + episode["end_s"]) / 2,
            245,
            str(index),
            ha="center",
            fontsize=8,
        )
    overview_axis.set_xlabel("time [s]")
    overview_axis.set_ylabel("cable pulled in [mm]")
    overview_axis.set_title(
        f"1. {recording_id}: {len(episodes_df)} episodes (green), numbered"
    )
    overview_axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2
    )  # below the plot, so it hides nothing
    overview_axis.set_ylim(-10, 260)

    # 2. zoom into one episode
    episode = episodes_df.iloc[zoom_episode_index]
    margin_s = 1.0
    in_zoom = (times_s >= episode["start_s"] - margin_s) & (
        times_s <= episode["end_s"] + margin_s
    )
    zoom_axis.plot(times_s[in_zoom], activity_mm[in_zoom], label="cable pulled in [mm]")
    zoom_axis.plot(
        times_s[in_zoom],
        clean_df.loc[in_zoom, "left_cable_tension_n"],
        label="left cable tension [N]",
    )
    zoom_axis.axhline(LIFT_THRESHOLD_MM, color="tab:red", linestyle="--")
    zoom_axis.axvspan(
        episode["start_s"],
        episode["end_s"],
        color="tab:green",
        alpha=0.25,
        label="episode (with padding)",
    )
    zoom_axis.axvline(
        episode["lift_start_s"],
        color="black",
        linestyle=":",
        label="lift start / end found",
    )
    zoom_axis.axvline(episode["lift_end_s"], color="black", linestyle=":")
    zoom_axis.set_xlabel("time [s]")
    zoom_axis.set_title(
        f"2. Zoom into {episode['episode_id']}: missing data inside = {episode['missing_fraction']:.0%}"
    )
    zoom_axis.legend(loc="upper right")

    figure.tight_layout()
    output_path = Path("docs") / f"{recording_id}_episodes.png"
    figure.savefig(output_path, dpi=110)
    print(f"saved {output_path}")


if __name__ == "__main__":
    plot_segments(
        sys.argv[1] if len(sys.argv) > 1 else "rec_000",
        int(sys.argv[2]) if len(sys.argv) > 2 else 0,
    )
