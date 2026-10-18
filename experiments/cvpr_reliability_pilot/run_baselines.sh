#!/usr/bin/env bash
# Run matched pilot baselines on three cancer cohorts.
# Example: bash experiments/cvpr_reliability_pilot/run_baselines.sh
# Override: EPOCHS=30 FOLDS="0" CANCERS="brca" bash .../run_baselines.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
EPOCHS="${EPOCHS:-100}"
FOLDS="${FOLDS:-0 1 2}"
CANCERS="${CANCERS:-brca kirc luad}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/cvpr_reliability_pilot/baselines}"
mkdir -p "$OUTPUT_ROOT"
for cancer in $CANCERS; do
  config="configs/${cancer}.yaml"
  if [[ ! -f "$config" ]]; then
    echo "ERROR: missing $config; inspect cancer config filenames" >&2
    exit 1
  fi
  echo "=== $cancer | epochs=$EPOCHS | folds=$FOLDS ==="
  # Existing runner: same split and initial weights for both methods.
  python -m experiments.phase1_optimizer_validation.run_phase1 \
    --config "$config" \
    --output "$OUTPUT_ROOT/$cancer" \
    --epochs "$EPOCHS" \
    --folds $FOLDS \
    --methods adamw disentangle_aux_only \
    2>&1 | tee "$OUTPUT_ROOT/$cancer.log"
done
echo "Completed launcher. Inspect per-fold JSON and manifest for failures."
