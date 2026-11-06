#!/bin/bash
# RA-SAMO v4: Validation-directed probe selection
# Run on EC2 from repository root: bash experiments/samo/run_v4.sh
#
# 25 runs: 1 method x 5 cancers x 5 folds x 50 epochs
# Expected time: ~3-4 hours on g4dn.xlarge

set -euo pipefail

cd "$(dirname "$0")/../.."

echo "=== RA-SAMO v4 Experiment ==="
echo "Start: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Host: $(hostname)"
echo ""

PYTHON="${PYTHON:-python3}"

$PYTHON -m experiments.samo.run \
  --cancers brca kirc luad lusc stad \
  --methods ra_samo_v4 \
  --folds 5 \
  --epochs 50 \
  --seed 42 \
  --out results/ra_samo_v4_50ep \
  --skip-existing \
  --modality-diagnostics

echo ""
echo "=== Done ==="
echo "End: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Results: results/ra_samo_v4_50ep/"
