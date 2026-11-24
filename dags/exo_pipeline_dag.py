"""The whole exo-pipeline-lite as one Airflow DAG.

generate -> write_mcap -> sync_and_calibrate -> quality_checks -> segment
                                                                     |
make_example_consent --------------------------------------------> anonymize
                                                                     |
                                              check_export_storage (fails once on purpose)
                                                                     |
                                                                   export -> verify_export

Trigger it by hand in the Airflow UI. It does not run on a schedule.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param, get_current_context, task

# Where the project lives inside the Airflow container (see docker-compose.yaml).
PROJECT_DIR = os.environ.get("EXO_PROJECT_DIR", "/opt/airflow/project")

default_args = {
    "owner": "pruthvi",
    "retries": 2,  # try each task up to 3 times in total
    "retry_delay": timedelta(seconds=10),
    "execution_timeout": timedelta(
        minutes=10
    ),  # a task that runs longer than this is stopped
}


def run_in_project(
    task_id: str, command: str, needs_secret_key: bool = False
) -> BashOperator:
    """A task that runs a shell command inside the project folder."""
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd {PROJECT_DIR} && {command}",
        append_env=needs_secret_key,  # keep the container's environment, so EXO_PSEUDONYM_KEY is visible
    )


with (
    DAG(
        dag_id="exo_pipeline",
        description="synthetic exoskeleton data: generate, sync, QC, segment, anonymize, export",
        schedule=None,  # only run when somebody triggers it
        start_date=datetime(2026, 1, 1),
        catchup=False,
        max_active_runs=1,  # never two runs at the same time (they would write the same folders)
        default_args=default_args,
        params={
            "fail_once": Param(
                True,
                type="boolean",
                description="make the storage check fail on its first try",
            )
        },
        tags=["exo", "interview-project"],
    ) as dag
):
    # Every task first deletes its own output folder, so that running it again (a retry!) gives the same result.
    generate = run_in_project(
        "generate",
        "rm -rf data/raw && python -m exo.generate --out data/raw --n 20",
    )
    write_mcap = run_in_project(
        "write_mcap",
        "rm -rf data/mcap && python -m exo.mcap_io --raw data/raw --out data/mcap",
    )
    sync_and_calibrate = run_in_project(
        "sync_and_calibrate",
        "rm -rf data/synced && python -m exo.sync --mcap data/mcap --out data/synced",
    )
    quality_checks = run_in_project(
        "quality_checks",
        "rm -rf data/qc data/clean && python -m exo.qc --synced data/synced --out data/qc --clean data/clean",
    )
    segment = run_in_project(
        "segment",
        "rm -rf data/episodes && python -m exo.segment --clean data/clean --out data/episodes",
    )
    make_example_consent = run_in_project(
        "make_example_consent",
        "python scripts/make_example_consent.py",  # FAKE consent table. In real life it comes from a consent system.
    )
    anonymize = run_in_project(
        "anonymize",
        "rm -rf data/anonymized && python -m exo.anonymize",
        needs_secret_key=True,
    )


@task(retries=2, retry_delay=timedelta(seconds=10))
def check_export_storage() -> str:
    """Pretend to check that the storage is reachable. Fails on the first try, to show a retry."""
    context = get_current_context()
    try_number = context["ti"].try_number
    if context["params"]["fail_once"] and try_number == 1:
        raise RuntimeError(
            "simulated storage hiccup (this is on purpose, Airflow will retry)"
        )
    return f"storage ok on try {try_number}"


export = run_in_project("export", "python -m exo.export")
verify_export = run_in_project(
    "verify_export",
    'latest=$(ls -d data/export/exo_dataset_v* | sort | tail -1) && python -m exo.export --verify "$latest"',
)

generate >> write_mcap >> sync_and_calibrate >> quality_checks >> segment
[segment, make_example_consent] >> anonymize
anonymize >> check_export_storage() >> export >> verify_export
