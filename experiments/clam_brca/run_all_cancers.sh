#!/bin/bash
# Run CLAM 2x3 experiment on all 5 cancers (5 folds each)
# 5 cancers x 5 folds x 6 methods = 150 runs
# CLAM runs ~700s each, MeanPool ~10s each → ~50 + 0.25 = ~50h per cancer
# Total: ~15-20h with 3 CLAM methods per fold dominating
#
# Usage: bash experiments/clam_brca/run_all_cancers.sh

set -e
cd /home/ubuntu/chaotic-multimodal-v2

PYTHON=/opt/pytorch/bin/python3

echo "=========================================="
echo "  CLAM 2x3 All Cancers (5 folds each)"
echo "=========================================="

for CANCER in brca kirc luad lusc stad; do
    CONFIG="configs/${CANCER}_clam.yaml"
    echo ""
    echo "=========================================="
    echo "  Starting: ${CANCER} (config: ${CONFIG})"
    echo "  $(date)"
    echo "=========================================="

    $PYTHON -m experiments.clam_brca.train \
        --config "$CONFIG" \
        --skip-existing \
        2>&1 | tee -a "logs/clam_${CANCER}.log"

    echo ""
    echo "  Finished: ${CANCER} at $(date)"
done

echo ""
echo "=========================================="
echo "  All cancers complete: $(date)"
echo "=========================================="
