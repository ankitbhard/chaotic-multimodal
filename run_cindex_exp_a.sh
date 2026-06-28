#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Experiment A (C-index): Multi-seed robustness
#
# Methods : all 17 optimizers
# Seeds   : 42 123 456 789 999
# Cancers : brca, kirc, luad, lusc, stad
# Total   : 17 × 5 × 5 = 425 runs
#
# Results: results_cindex_exp_a/<cancer>/seed_<seed>/<opt>_survival.json
# Logs   : logs_cindex_exp_a/<cancer>_seed<seed>_<opt>.log
#
# Usage:
#   tmux new -s cindex_a
#   bash run_cindex_exp_a.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e
set -o pipefail

CANCERS="brca kirc luad lusc stad"
OPTS="sgd sgd_mom adam adadelta cosine sgd_ge sgd_mom_ge adam_ge adadelta_ge cosine_ge chaotic_no_ge chaotic chaotic_adadelta chaotic_adaptive adam_adaptive sgd_mom_adaptive"
SEEDS="42 123 456 789 999"
EPOCHS=200
SAVE_BASE="results_cindex_exp_a"
LOG_DIR="logs_cindex_exp_a"

PYTHON="${PYTHON:-/opt/pytorch/bin/python3}"

mkdir -p "$LOG_DIR"
cd "$(dirname "$0")"

START_TIME=$(date +%s)
TOTAL_RUNS=0
DONE_RUNS=0
FAILED_RUNS=()

# Count total
for cancer in $CANCERS; do for seed in $SEEDS; do for opt in $OPTS; do
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
done; done; done

echo "════════════════════════════════════════════════════════════"
echo "  Experiment A (C-index) — Multi-seed robustness"
echo "  Started : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Cancers : $CANCERS"
echo "  Opts    : 17 optimizers"
echo "  Seeds   : $SEEDS"
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

    for seed in $SEEDS; do
        SAVE_DIR="${SAVE_BASE}/${cancer}/seed_${seed}"
        mkdir -p "$SAVE_DIR"

        echo "  ── seed ${seed} ──"

        for opt in $OPTS; do
            RUN_NUM=$((RUN_NUM + 1))
            LOG_FILE="${LOG_DIR}/${cancer}_seed${seed}_${opt}.log"
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
                --seed "$seed" \
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
                FAILED_RUNS+=("${cancer}/seed${seed}/${opt}")
            fi

        done  # opts
    done  # seeds
done  # cancers

# ── Summary ───────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
WALL=$((END_TIME - START_TIME))
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Experiment A (C-index) COMPLETE"
echo "  Wall time : $((WALL/3600))h $(( (WALL%3600)/60 ))m $((WALL%60))s"
echo "  Runs done : $DONE_RUNS / $TOTAL_RUNS"
[[ ${#FAILED_RUNS[@]} -gt 0 ]] && printf "  FAILED    : %s\n" "${FAILED_RUNS[@]}"
echo "════════════════════════════════════════════════════════════"

# ── Aggregation ──────────────────────────────────────────────────────────────
echo ""
$PYTHON - <<'PYEOF'
import json, os, numpy as np

save_base = "results_cindex_exp_a"
cancers   = ["brca", "kirc", "luad", "lusc", "stad"]
opts      = ["sgd", "sgd_mom", "adam", "adadelta", "cosine",
             "sgd_ge", "sgd_mom_ge", "adam_ge", "adadelta_ge", "cosine_ge",
             "chaotic_no_ge", "chaotic", "chaotic_adadelta",
             "chaotic_adaptive", "adam_adaptive", "sgd_mom_adaptive"]
seeds     = [42, 123, 456, 789, 999]

print("C-index Results (mean ± std across 5 seeds):")
print(f"\n{'Optimizer':<22} {'BRCA':>12} {'KIRC':>12} {'LUAD':>12} {'LUSC':>12} {'STAD':>12} {'Avg':>12}")
print("-" * 94)

for opt in opts:
    row = [opt]
    all_means = []
    for cancer in cancers:
        cidxs = []
        for seed in seeds:
            p = os.path.join(save_base, cancer, f"seed_{seed}", f"{opt}_survival.json")
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
