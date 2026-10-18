#!/bin/bash
# Full pilot: all 6 methods × 3 folds × 100 epochs + robustness evaluation
# Cancers: BRCA, KIRC, LUAD
set -e

PYTHON="/opt/pytorch/bin/python3"
PROJECT_ROOT="$HOME/chaotic-multimodal-v2"
cd "$PROJECT_ROOT"

echo "=== CVPR Reliability Pilot — Full Run ==="
echo "Start: $(date)"

# Phase 2+4: Training all methods
for cancer in brca kirc luad; do
    echo ""
    echo "=============================="
    echo "  Training: $cancer"
    echo "=============================="

    $PYTHON -m experiments.cvpr_reliability_pilot.train \
        --config configs/${cancer}.yaml \
        --methods all \
        --folds 3 --epochs 100 \
        --save_dir results/cvpr_reliability_pilot/${cancer} \
        --skip_existing \
        2>&1 | tee results/cvpr_reliability_pilot/${cancer}/pilot.log
done

# Phase 5: Robustness evaluation
echo ""
echo "=== Robustness Evaluation ==="
for cancer in brca kirc luad; do
    echo ""
    echo "  Robustness: $cancer"

    $PYTHON -m experiments.cvpr_reliability_pilot.evaluate \
        --config configs/${cancer}.yaml \
        --results_dir results/cvpr_reliability_pilot/${cancer} \
        --methods all --folds 3 \
        2>&1 | tee results/cvpr_reliability_pilot/${cancer}/robustness.log
done

# Phase 6: Analysis
echo ""
echo "=== Statistical Analysis ==="
$PYTHON -m experiments.cvpr_reliability_pilot.analyze_results \
    --results_base results/cvpr_reliability_pilot \
    --cancers brca kirc luad \
    --folds 3 \
    2>&1 | tee results/cvpr_reliability_pilot/analysis.log

echo ""
echo "=== Pilot Complete: $(date) ==="
