#!/usr/bin/env bash
# Run the whole exo-pipeline-lite from the command line (steps 1 to 7), then print the results.
# Usage:  export EXO_PSEUDONYM_KEY="a long secret text"  &&  bash scripts/run_pipeline.sh
set -euo pipefail
: "${EXO_PSEUDONYM_KEY:?set EXO_PSEUDONYM_KEY first (a long secret text)}"

# start from clean folders, so that an old result can never end up in the new one
# (data/export is kept: every run adds a NEW dataset version)
rm -rf data/raw data/mcap data/synced data/qc data/clean data/episodes data/anonymized

python -m exo.generate --out data/raw --n 20
python -m exo.mcap_io --raw data/raw --out data/mcap
python -m exo.sync --mcap data/mcap --out data/synced
python -m exo.qc --synced data/synced --out data/qc --clean data/clean
python -m exo.segment --clean data/clean --out data/episodes
python scripts/make_example_consent.py
python -m exo.anonymize
python -m exo.export
python -m exo.export --verify "$(ls -d data/export/exo_dataset_v* | sort | tail -1)"

echo
echo "===== results ====="
python scripts/evaluate_sync.py | tail -1
python scripts/evaluate_qc.py | tail -3
python scripts/evaluate_segment.py | tail -3