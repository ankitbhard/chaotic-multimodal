#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Experiment B (C-index): 5-fold cross-validation
#
# Methods : all 17 optimizers
# Cancers : brca, kirc, luad, lusc, stad
# Folds   : 0 1 2 3 4
# Total   : 17 × 5 × 5 = 425 runs
#
# Results: results_cindex_exp_b/<cancer>/fold_<k>/<opt>_survival.json
# Logs   : logs_cindex_exp_b/<cancer>_fold<k>_<opt>.log
#
# Usage:
#   tmux new -s cindex_b
#   bash run_cindex_exp_b.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
set -o pipefail

CANCERS="brca kirc luad lusc stad"
OPTS="sgd sgd_mom adam adadelta cosine sgd_ge sgd_mom_ge adam_ge adadelta_ge cosine_ge chaotic_no_ge chaotic chaotic_adadelta chaotic_adaptive adam_adaptive sgd_mom_adaptive"
N_FOLDS=5
EPOCHS=200
SAVE_BASE="results_cindex_exp_b"
LOG_DIR="logs_cindex_exp_b"

PYTHON="${PYTHON:-/opt/pytorch/bin/python3}"

mkdir -p "$LOG_DIR"
cd "$(dirname "$0")"

START_TIME=$(date +%s)
TOTAL_RUNS=0
DONE_RUNS=0
FAILED_RUNS=()

# Count total
for cancer in $CANCERS; do for fold in $(seq 0 $((N_FOLDS - 1))); do for opt in $OPTS; do
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
done; done; done

echo "════════════════════════════════════════════════════════════"
echo "  Experiment B (C-index) — 5-fold CV"
echo "  Started : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Cancers : $CANCERS"
echo "  Opts    : 17 optimizers"
echo "  Folds   : $N_FOLDS"
echo "  Total   : $TOTAL_RUNS runs"
echo "  Epochs  : $EPOCHS"
echo "════════════════════════════════════════════════════════════"

RUN_NUM=0
for cancer in $CANCERS; do
    CONFIG="configs/${cancer}.yaml"
    if [[ ! -f "$CONFIG" ]]; then
        echo "[SKIP] Config not found: $CONFIG"; continue
    fi

    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "  Cancer: ${cancer^^}"
    echo "╚══════════════════════════════════════╝"

    for fold in $(seq 0 $((N_FOLDS - 1))); do
        SAVE_DIR="${SAVE_BASE}/${cancer}/fold_${fold}"
        mkdir -p "$SAVE_DIR"

        echo "  ── fold ${fold} ──"

        for opt in $OPTS; do
            RUN_NUM=$((RUN_NUM + 1))
            LOG_FILE="${LOG_DIR}/${cancer}_fold${fold}_${opt}.log"
            SURV_FILE="${SAVE_DIR}/${opt}_survival.json"

            if [[ -f "$SURV_FILE" ]] && $PYTHON -c "
import json, sys
h = json.load(open('$SURV_FILE'))
sys.exit(0 if 'test_cindex' in h else 1)
" 2>/dev/null; then
                DONE_RUNS=$((DONE_RUNS + 1))
                continue
            fi

            RUN_START=$(date +%s)
            echo -n "    [${RUN_NUM}/${TOTAL_RUNS}] ${opt} ... "

            if $PYTHON train_survival.py \
                --config "$CONFIG" \
                --optimizers "$opt" \
                --save_dir "$SAVE_DIR" \
                --epochs "$EPOCHS" \
                --fold "$fold" \
                --n_folds "$N_FOLDS" \
                > "$LOG_FILE" 2>&1; then

                RUN_END=$(date +%s)
                ELAPSED=$((RUN_END - RUN_START))
                CINDEX=$($PYTHON -c "
import json
h = json.load(open('$SURV_FILE'))
print(f\"{h.get('test_cindex',0):.4f}\")
" 2>/dev/null || echo "?")
                echo "✓ cindex=${CINDEX}  [${ELAPSED}s]"
                DONE_RUNS=$((DONE_RUNS + 1))
            else
                echo "✗ FAILED"
                FAILED_RUNS+=("${cancer}/fold${fold}/${opt}")
            fi

        done  # opts
    done  # folds
done  # cancers

# ── Summary ───────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
WALL=$((END_TIME - START_TIME))
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Experiment B (C-index) COMPLETE"
echo "  Wall time : $((WALL/3600))h $(( (WALL%3600)/60 ))m $((WALL%60))s"
echo "  Runs done : $DONE_RUNS / $TOTAL_RUNS"
[[ ${#FAILED_RUNS[@]} -gt 0 ]] && printf "  FAILED    : %s\n" "${FAILED_RUNS[@]}"
echo "════════════════════════════════════════════════════════════"

# ── Aggregation ──────────────────────────────────────────────────────────────
echo ""
$PYTHON - <<'PYEOF'
import json, os, numpy as np

save_base = "results_cindex_exp_b"
cancers   = ["brca", "kirc", "luad", "lusc", "stad"]
opts      = ["sgd", "sgd_mom", "adam", "adadelta", "cosine",
             "sgd_ge", "sgd_mom_ge", "adam_ge", "adadelta_ge", "cosine_ge",
             "chaotic_no_ge", "chaotic", "chaotic_adadelta",
             "chaotic_adaptive", "adam_adaptive", "sgd_mom_adaptive"]
n_folds   = 5

print("C-index Results (mean ± std across 5 folds):")
print(f"\n{'Optimizer':<22} {'BRCA':>12} {'KIRC':>12} {'LUAD':>12} {'LUSC':>12} {'STAD':>12} {'Avg':>12}")
print("-" * 94)

for opt in opts:
    row = [opt]
    all_means = []
    for cancer in cancers:
        cidxs = []
        for fold in range(n_folds):
            p = os.path.join(save_base, cancer, f"fold_{fold}", f"{opt}_survival.json")
            if os.path.exists(p):
                try:
                    h = json.load(open(p))
                    ci = h.get("test_cindex")
                    if ci is not None:
                        cidxs.append(ci)
                except:
                    pass
        if cidxs:
            m, s = np.mean(cidxs), np.std(cidxs)
            row.append(f"{m:.3f}±{s:.3f}")
            all_means.append(m)
        else:
            row.append("---")
    if all_means:
        row.append(f"{np.mean(all_means):.3f}")
    else:
        row.append("---")
    print(f"{row[0]:<22} {row[1]:>12} {row[2]:>12} {row[3]:>12} {row[4]:>12} {row[5]:>12} {row[6]:>12}")
PYEOF
