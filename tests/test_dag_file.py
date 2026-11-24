import importlib.util
import re
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
DAG_FILE = PROJECT_DIR / "dags" / "exo_pipeline_dag.py"


def test_every_module_and_script_in_the_dag_exists():
    """Cheap check that needs no Airflow: the DAG only calls code that is really there."""
    dag_text = DAG_FILE.read_text()
    modules = re.findall(r"python -m exo\.(\w+)", dag_text)
    scripts = re.findall(r"python (scripts/\w+\.py)", dag_text)
    assert modules and scripts
    for module in modules:
        assert (PROJECT_DIR / "src" / "exo" / f"{module}.py").exists(), module
    for script in scripts:
        assert (PROJECT_DIR / script).exists(), script


def test_the_secret_key_is_not_written_into_the_dag():
    dag_text = DAG_FILE.read_text()
    assert not re.search(r"EXO_PSEUDONYM_KEY\s*=\s*[\"']", dag_text)
    assert (
        "append_env=needs_secret_key" in dag_text
    )  # the key comes from the container's environment


def test_the_dag_loads_and_has_the_expected_tasks():
    """Only runs when Airflow is installed (for example inside the Docker image)."""
    pytest.importorskip("airflow")
    spec = importlib.util.spec_from_file_location("exo_pipeline_dag", DAG_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dag = module.dag
    assert set(dag.task_dict) == {
        "generate",
        "write_mcap",
        "sync_and_calibrate",
        "quality_checks",
        "segment",
        "make_example_consent",
        "anonymize",
        "check_export_storage",
        "export",
        "verify_export",
    }
    assert dag.default_args["retries"] >= 1
    assert dag.catchup is False
    assert dag.max_active_runs == 1
    assert dag.task_dict["anonymize"].upstream_task_ids == {
        "segment",
        "make_example_consent",
    }
    assert dag.task_dict["export"].upstream_task_ids == {"check_export_storage"}
