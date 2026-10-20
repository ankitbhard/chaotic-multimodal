#!/bin/bash
# Phase 2 Rerun with full training diagnostics
# Methods: learned_gating_aux, pcrf_aux, pcrf_aux_warmup
# Cancers: BRCA, KIRC, LUAD
# 3 methods × 3 cancers × 3 folds = 27 runs + diagnostics CSVs

set -e
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python3}"
SAVE_BASE="results/cvpr_reliability_pilot/phase2_diagnostics"

echo "============================================================"
echo "CVPR Reliability Pilot — Phase 2 (with diagnostics)"
echo "Captures 18 categories of optimizer/training metrics"
echo "Methods: learned_gating_aux, pcrf_aux, pcrf_aux_warmup"
echo "Cancers: BRCA, KIRC, LUAD"
echo "============================================================"

for CANCER in brca kirc luad; do
    echo ""
    echo "============================================================"
    echo "  ${CANCER^^} — Phase 2 diagnostics (3 methods × 3 folds × 100 epochs)"
    echo "============================================================"
    $PYTHON -m experiments.cvpr_reliability_pilot.train \
        --config configs/${CANCER}.yaml \
        --methods learned_gating_aux pcrf_aux pcrf_aux_warmup \
        --folds 3 --epochs 100 \
        --save_dir ${SAVE_BASE}/${CANCER} \
        --diagnostics \
        --hessian_every 10 \
        --skip_existing
done

echo ""
echo "============================================================"
echo "All Phase 2 diagnostics training complete!"
echo "Results saved to: ${SAVE_BASE}/"
echo "Diagnostic CSVs: *_step_diagnostics.csv, *_epoch_diagnostics.csv, *_hessian_diagnostics.csv"
echo "============================================================"
