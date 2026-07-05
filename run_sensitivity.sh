#!/bin/bash
# Sensitivity sweep: 7 unique configs × 2 cohorts (BRCA, LUAD) = 14 runs
# Default: theta=0.10, window=10, alpha=0.5
# Vary one parameter at a time

PYTHON=/opt/pytorch/bin/python3
SCRIPT=train.py
SEED=42
OUTDIR=results_sensitivity

# 7 unique (theta, window, alpha) combinations
OPTS=(
    "adam_sens_t0.05_w10_a0.5"
    "adam_sens_t0.10_w10_a0.5"
    "adam_sens_t0.15_w10_a0.5"
    "adam_sens_t0.10_w5_a0.5"
    "adam_sens_t0.10_w20_a0.5"
    "adam_sens_t0.10_w10_a0.25"
    "adam_sens_t0.10_w10_a1.0"
)

mkdir -p logs

for cancer in brca luad; do
    for opt in "${OPTS[@]}"; do
        echo "=== Running $cancer / $opt ==="
        $PYTHON $SCRIPT \
            --config "configs/${cancer}.yaml" \
            --optimizers "$opt" \
            --seed $SEED \
            --save_dir "$OUTDIR/${cancer}" \
            2>&1 | tee "logs/sensitivity_${cancer}_${opt}.log"
    done
done

echo "=== Sensitivity sweep complete ==="
