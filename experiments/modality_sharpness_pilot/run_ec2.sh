#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-python3}"
CANCER="${CANCER:-brca}"
EPOCHS="${EPOCHS:-10}"
for FOLD in 0 1 2 3 4; do
  "$PYTHON" experiments/modality_sharpness_pilot/run_pilot.py \
    --config "configs/${CANCER}.yaml" \
    --output "experiments/modality_sharpness_pilot/outputs/${CANCER}/fold_${FOLD}" \
    --fold "$FOLD" --epochs "$EPOCHS" \
    --optimizers sam msam block_fixed block_adaptive
done
