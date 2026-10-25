#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python -m unittest experiments.samo.test_samo -v
python -m experiments.samo.run --cancers brca --folds 3 --epochs 1 --methods baseline --out results/samo_smoke --skip-existing
python -m experiments.samo.run --cancers brca kirc stad --folds 3 --epochs 50 --out results/samo_survival --skip-existing
python -m experiments.samo.summarize --results results/samo_survival --out experiments/samo/RESULTS_SUMMARY.csv
