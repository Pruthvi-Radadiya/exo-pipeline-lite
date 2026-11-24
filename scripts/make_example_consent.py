"""Create a FAKE consent table for the project. In real life this comes from the consent system.

Run:  python scripts/make_example_consent.py
"""

from pathlib import Path

import pandas as pd

WORKERS_WITHOUT_CONSENT = ["W005"]  # this worker withdrew consent


def main() -> None:
    worker_ids = [f"W{number:03d}" for number in range(1, 7)]
    consent_df = pd.DataFrame(
        {
            "worker_id": worker_ids,
            "consent_given": [
                worker_id not in WORKERS_WITHOUT_CONSENT for worker_id in worker_ids
            ],
            "changed_on": ["2026-09-15"] * len(worker_ids),
        }
    )
    output_path = Path("data/private/consent.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    consent_df.to_csv(output_path, index=False)
    print(consent_df.to_string(index=False))
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
