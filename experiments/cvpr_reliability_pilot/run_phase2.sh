#!/bin/bash
# Phase 2: Isolate reliability supervision effect
# Methods: learned_gating_aux, pcrf_aux, pcrf_aux_warmup
# Cancers: BRCA, KIRC, LUAD
# 3 methods × 3 cancers × 3 folds = 27 runs

set -e
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python3}"
SAVE_BASE="results/cvpr_reliability_pilot/phase2"

echo "============================================================"
echo "CVPR Reliability Pilot — Phase 2"
echo "Isolating reliability supervision effect"
echo "Methods: learned_gating_aux, pcrf_aux, pcrf_aux_warmup"
echo "Cancers: BRCA, KIRC, LUAD"
echo "============================================================"

for CANCER in brca kirc luad; do
    echo ""
    echo "============================================================"
    echo "  ${CANCER^^} — Phase 2 (3 methods × 3 folds × 100 epochs)"
    echo "============================================================"
    $PYTHON -m experiments.cvpr_reliability_pilot.train \
        --config configs/${CANCER}.yaml \
        --methods learned_gating_aux pcrf_aux pcrf_aux_warmup \
        --folds 3 --epochs 100 \
        --save_dir ${SAVE_BASE}/${CANCER} \
        --skip_existing
done

echo ""
echo "============================================================"
echo "All Phase 2 training complete!"
echo "Results saved to: ${SAVE_BASE}/"
echo "============================================================"
