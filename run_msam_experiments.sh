#!/bin/bash
# run_msam_experiments.sh — Run all M-SAM paper experiments on EC2.
#
# Total new GPU runs: 183
#   - Main comparison: 100 (4 opts x 5 cancers x 5 folds)
#   - Ablation: 50 (2 opts x 5 cancers x 5 folds)
#   - Sensitivity: 27 (rho + alpha sweeps on 3 cancers, fold 0)
#   - Loss landscape: 6 (3 opts x 2 cancers, with checkpoints)
#
# Estimated time: ~10-12 hours on single GPU.
#
# Usage:
#   bash run_msam_experiments.sh           # Run all
#   bash run_msam_experiments.sh main      # Main comparison only
#   bash run_msam_experiments.sh ablation  # Ablation only
#   bash run_msam_experiments.sh sensitivity
#   bash run_msam_experiments.sh landscape

set -e

PYTHON="/opt/pytorch/bin/python3"
SCRIPT="train_msam_survival.py"
BASE_DIR="results_msam_paper"
CANCERS="brca kirc luad lusc stad"
N_FOLDS=5

echo "============================================================"
echo "  M-SAM Paper Experiments"
echo "  Start: $(date)"
echo "============================================================"

# ── Phase 2A: Main Comparison ────────────────────────────────────────────────
run_main() {
    echo ""
    echo "============================================================"
    echo "  PHASE 2A: Main Comparison (100 runs)"
    echo "  Optimizers: cosine_ge, sam, msam, msam_chaotic"
    echo "============================================================"

    for cancer in $CANCERS; do
        echo ""
        echo ">>> Cancer: $cancer"
        $PYTHON $SCRIPT \
            --config configs/${cancer}.yaml \
            --optimizers cosine_ge sam msam msam_chaotic \
            --save_dir ${BASE_DIR}/${cancer} \
            --n_folds $N_FOLDS \
            2>&1 | tee -a logs/msam_main_${cancer}.log
    done

    echo ""
    echo "  Phase 2A complete: $(date)"
}

# ── Phase 2B: Ablation ───────────────────────────────────────────────────────
run_ablation() {
    echo ""
    echo "============================================================"
    echo "  PHASE 2B: Ablation (50 runs)"
    echo "  Optimizers: msam_no_ogm, msam_uniform"
    echo "============================================================"

    for cancer in $CANCERS; do
        echo ""
        echo ">>> Cancer: $cancer"
        $PYTHON $SCRIPT \
            --config configs/${cancer}.yaml \
            --optimizers msam_no_ogm msam_uniform \
            --save_dir ${BASE_DIR}/${cancer} \
            --n_folds $N_FOLDS \
            2>&1 | tee -a logs/msam_ablation_${cancer}.log
    done

    echo ""
    echo "  Phase 2B complete: $(date)"
}

# ── Phase 2C: Sensitivity Sweeps ─────────────────────────────────────────────
run_sensitivity() {
    echo ""
    echo "============================================================"
    echo "  PHASE 2C: Sensitivity Sweeps (27 runs)"
    echo "  rho: {0.01, 0.02, 0.05, 0.10, 0.20} x 3 cancers"
    echo "  alpha: {0.1, 0.25, 0.5, 1.0} x 3 cancers"
    echo "============================================================"

    SENS_CANCERS="brca kirc luad"

    # Rho sweep (fold 0 only)
    for cancer in $SENS_CANCERS; do
        for rho in 0.01 0.02 0.05 0.10 0.20; do
            echo ">>> $cancer rho=$rho"
            $PYTHON $SCRIPT \
                --config configs/${cancer}.yaml \
                --optimizers msam \
                --save_dir ${BASE_DIR}_sensitivity/${cancer}/rho_${rho} \
                --n_folds $N_FOLDS --fold 0 \
                --rho $rho \
                2>&1 | tee -a logs/msam_sens_${cancer}_rho${rho}.log
        done
    done

    # Alpha sweep (fold 0 only)
    for cancer in $SENS_CANCERS; do
        for alpha in 0.1 0.25 0.5 1.0; do
            echo ">>> $cancer alpha=$alpha"
            $PYTHON $SCRIPT \
                --config configs/${cancer}.yaml \
                --optimizers msam \
                --save_dir ${BASE_DIR}_sensitivity/${cancer}/alpha_${alpha} \
                --n_folds $N_FOLDS --fold 0 \
                --alpha $alpha \
                2>&1 | tee -a logs/msam_sens_${cancer}_alpha${alpha}.log
        done
    done

    echo ""
    echo "  Phase 2C complete: $(date)"
}

# ── Phase 2D: Loss Landscape Checkpoints ─────────────────────────────────────
run_landscape() {
    echo ""
    echo "============================================================"
    echo "  PHASE 2D: Loss Landscape Checkpoints (6 runs)"
    echo "  cosine, sam, msam on BRCA + KIRC (fold 0)"
    echo "============================================================"

    for cancer in brca kirc; do
        echo ">>> $cancer — saving checkpoints"
        $PYTHON $SCRIPT \
            --config configs/${cancer}.yaml \
            --optimizers cosine_ge sam msam \
            --save_dir ${BASE_DIR}_landscape/${cancer} \
            --n_folds $N_FOLDS --fold 0 \
            --save_checkpoints \
            2>&1 | tee -a logs/msam_landscape_${cancer}.log
    done

    echo ""
    echo "  Phase 2D complete: $(date)"
}

# ── Main dispatch ────────────────────────────────────────────────────────────
mkdir -p logs

case "${1:-all}" in
    main)        run_main ;;
    ablation)    run_ablation ;;
    sensitivity) run_sensitivity ;;
    landscape)   run_landscape ;;
    all)
        run_main
        run_ablation
        run_sensitivity
        run_landscape
        ;;
    *)
        echo "Usage: $0 {all|main|ablation|sensitivity|landscape}"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE: $(date)"
echo "============================================================"
