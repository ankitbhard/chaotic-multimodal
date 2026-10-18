#!/bin/bash
# Phase 2: Baseline experiments (3-fold x 100 epochs)
# Methods: adamw, aux_cox, mod_dropout
# Cancers: BRCA, KIRC, LUAD
#
# Usage:
#   bash experiments/cvpr_reliability_pilot/run_baselines.sh
#   EPOCHS=30 CANCERS="brca" bash experiments/cvpr_reliability_pilot/run_baselines.sh
set -euo pipefail

PYTHON="${PYTHON:-/opt/pytorch/bin/python3}"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/chaotic-multimodal-v2}"
EPOCHS="${EPOCHS:-100}"
FOLDS="${FOLDS:-3}"
CANCERS="${CANCERS:-brca kirc luad}"
cd "$PROJECT_ROOT"

echo "=== Phase 2: Baseline Experiments ==="
echo "Start: $(date)"
echo "Epochs: $EPOCHS | Folds: $FOLDS | Cancers: $CANCERS"

for cancer in $CANCERS; do
    config="configs/${cancer}.yaml"
    if [[ ! -f "$config" ]]; then
        echo "ERROR: missing $config" >&2
        exit 1
    fi

    echo ""
    echo "=============================="
    echo "  Cancer: $cancer"
    echo "=============================="

    mkdir -p "results/cvpr_reliability_pilot/${cancer}"

    $PYTHON -m experiments.cvpr_reliability_pilot.train \
        --config "$config" \
        --methods adamw aux_cox mod_dropout \
        --folds "$FOLDS" --epochs "$EPOCHS" \
        --save_dir "results/cvpr_reliability_pilot/${cancer}" \
        --skip_existing \
        2>&1 | tee "results/cvpr_reliability_pilot/${cancer}/baselines.log"
done

echo ""
echo "=== Phase 2 Complete: $(date) ==="
