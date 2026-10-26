#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python -m unittest experiments.samo.test_samo experiments.csamo.test_csamo -v
python -m experiments.csamo.run --cancers brca --folds 3 --epochs 6 --out results/csamo_smoke --skip-existing
python -m experiments.csamo.run --cancers brca kirc stad --folds 3 --epochs 50 --out results/csamo_survival --skip-existing
python -m experiments.samo.summarize --results results/csamo_survival --out experiments/csamo/RESULTS_SUMMARY.csv
