#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all.sh  —  Full 12-optimizer × 6-cancer training sweep
#
# Optimizer groups (72 runs total):
#   Baseline (no OGM-GE, no chaotic LR):  sgd, sgd_mom, adam, adadelta, cosine
#   Chaotic LR only (no OGM-GE):          chaotic_no_ge
#   OGM-GE only (no chaotic LR):          sgd_ge, sgd_mom_ge, adam_ge, adadelta_ge, cosine_ge
#   Full method (chaotic LR + OGM-GE):    chaotic
#
# Results saved to: results_singlehead/<cancer>/
# Logs saved to:    logs/<cancer>_<opt>.log
#
# Usage:
#   tmux new -s training
#   bash run_all.sh
#   # Ctrl+B D to detach; tmux attach -t training to recheck
#
# Resume a single failed run:
#   bash run_all.sh --cancer brca --opt adam
# ─────────────────────────────────────────────────────────────────────────────

set -e
set -o pipefail

# ── Config ────────────────────────────────────────────────────────────────────

CANCERS="brca luad kirc stad lusc gbm"

MAIN_OPTS="sgd sgd_mom adam adadelta cosine chaotic_no_ge sgd_ge sgd_mom_ge adam_ge adadelta_ge cosine_ge chaotic"

EPOCHS=200
SAVE_BASE="results_singlehead"
LOG_DIR="logs"

# Python binary (use /opt/pytorch env on EC2)
PYTHON="${PYTHON:-/opt/pytorch/bin/python3}"

# ── Parse optional single-run args ────────────────────────────────────────────

SINGLE_CANCER=""
SINGLE_OPT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cancer) SINGLE_CANCER="$2"; shift 2 ;;
        --opt)    SINGLE_OPT="$2";    shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -n "$SINGLE_CANCER" && -n "$SINGLE_OPT" ]]; then
    CANCERS="$SINGLE_CANCER"
    MAIN_OPTS="$SINGLE_OPT"
    echo "=== Single-run mode: cancer=$SINGLE_CANCER  opt=$SINGLE_OPT ==="
fi

# ── Setup ─────────────────────────────────────────────────────────────────────

mkdir -p "$LOG_DIR"

START_TIME=$(date +%s)
TOTAL_RUNS=0
FAILED_RUNS=()

cd "$(dirname "$0")"

echo "════════════════════════════════════════════════════════════"
echo "  Pan-cancer survival sweep — single-head model"
echo "  Started : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Cancers : $CANCERS"
echo "  Opts    : $MAIN_OPTS"
echo "  Epochs  : $EPOCHS"
echo "  Save    : $SAVE_BASE/"
echo "════════════════════════════════════════════════════════════"

# ── Main loop ─────────────────────────────────────────────────────────────────

for cancer in $CANCERS; do

    CONFIG="configs/${cancer}.yaml"
    if [[ ! -f "$CONFIG" ]]; then
        echo "[SKIP] Config not found: $CONFIG"
        continue
    fi

    SAVE_DIR="${SAVE_BASE}/${cancer}"
    mkdir -p "$SAVE_DIR"

    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "  Cancer: ${cancer^^}"
    echo "╚══════════════════════════════════════╝"

    for opt in $MAIN_OPTS; do

        LOG_FILE="${LOG_DIR}/${cancer}_${opt}.log"
        HIST_FILE="${SAVE_DIR}/${opt}_history.json"

        # Skip already-completed runs (resume support)
        if [[ -f "$HIST_FILE" ]]; then
            # Check if test_survival_acc key exists (marks completed run)
            if $PYTHON -c "
import json, sys
h = json.load(open('$HIST_FILE'))
sys.exit(0 if 'test_survival_acc' in h else 1)
" 2>/dev/null; then
                echo "  [SKIP] ${cancer}/${opt} — already complete ($($PYTHON -c "
import json
h = json.load(open('$HIST_FILE'))
print(f\"acc={h['test_survival_acc']*100:.1f}%  auc={h.get('test_survival_auc', 0):.3f}\")
" 2>/dev/null || echo "see $HIST_FILE"))"
                TOTAL_RUNS=$((TOTAL_RUNS + 1))
                continue
            fi
        fi

        echo "  ── ${opt} ──"
        RUN_START=$(date +%s)

        if $PYTHON train.py \
            --config "$CONFIG" \
            --optimizers "$opt" \
            --save_dir "$SAVE_DIR" \
            --epochs "$EPOCHS" \
            2>&1 | tee "$LOG_FILE"; then

            RUN_END=$(date +%s)
            ELAPSED=$((RUN_END - RUN_START))

            # Extract final test metrics from saved JSON
            METRICS=$($PYTHON -c "
import json
h = json.load(open('$HIST_FILE'))
print(f\"acc={h.get('test_survival_acc',0)*100:.1f}%  \
f1={h.get('test_survival_f1',0):.3f}  \
auc={h.get('test_survival_auc',0):.3f}\")
" 2>/dev/null || echo "see $HIST_FILE")

            echo "  ✓ ${cancer}/${opt}  ${METRICS}  [${ELAPSED}s]"
            TOTAL_RUNS=$((TOTAL_RUNS + 1))

        else
            echo "  ✗ FAILED: ${cancer}/${opt} — check $LOG_FILE"
            FAILED_RUNS+=("${cancer}/${opt}")
            TOTAL_RUNS=$((TOTAL_RUNS + 1))
        fi

    done  # opts

done  # cancers

# ── Summary ───────────────────────────────────────────────────────────────────

END_TIME=$(date +%s)
WALL=$((END_TIME - START_TIME))
HH=$((WALL / 3600))
MM=$(( (WALL % 3600) / 60 ))
SS=$((WALL % 60))

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  SWEEP COMPLETE"
echo "  Finished : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Wall time: ${HH}h ${MM}m ${SS}s"
echo "  Runs done: $TOTAL_RUNS"
if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
    echo "  FAILED   : ${#FAILED_RUNS[@]}"
    for r in "${FAILED_RUNS[@]}"; do
        echo "             ✗ $r"
    done
else
    echo "  All runs succeeded."
fi
echo "════════════════════════════════════════════════════════════"

# ── Print results table ───────────────────────────────────────────────────────

echo ""
echo "Results table (test metrics):"
echo ""
$PYTHON - <<'PYEOF'
import json, os, glob

save_base = "results_singlehead"
cancers   = ["brca", "luad", "kirc", "stad", "lusc", "gbm"]
opts      = ["sgd", "sgd_mom", "adam", "adadelta", "cosine",
             "chaotic_no_ge",
             "sgd_ge", "sgd_mom_ge", "adam_ge", "adadelta_ge", "cosine_ge",
             "chaotic"]

# Header
cols = ["optimizer"] + [f"{c[:4].upper()} acc" for c in cancers] + ["avg acc"] + [f"{c[:4].upper()} auc" for c in cancers] + ["avg auc"]
print("  " + "  ".join(f"{c:>10}" for c in cols))
print("  " + "-" * (12 * len(cols)))

for opt in opts:
    accs, aucs = [], []
    row_acc, row_auc = [], []
    for cancer in cancers:
        p = os.path.join(save_base, cancer, f"{opt}_history.json")
        if os.path.exists(p):
            h = json.load(open(p))
            a = h.get("test_survival_acc", float("nan"))
            u = h.get("test_survival_auc", float("nan"))
        else:
            a, u = float("nan"), float("nan")
        accs.append(a)
        aucs.append(u)
        row_acc.append(f"{a*100:>6.1f}%" if a == a else "     ---")
        row_auc.append(f"{u:>7.3f}"      if u == u else "      ---")

    valid_accs = [x for x in accs if x == x]
    valid_aucs = [x for x in aucs if x == x]
    avg_acc = sum(valid_accs)/len(valid_accs) if valid_accs else float("nan")
    avg_auc = sum(valid_aucs)/len(valid_aucs) if valid_aucs else float("nan")

    vals = ([f"{opt:>14}"]
            + row_acc
            + [f"{avg_acc*100:>6.1f}%" if avg_acc==avg_acc else "     ---"]
            + row_auc
            + [f"{avg_auc:>7.3f}"      if avg_auc==avg_auc else "      ---"])
    print("  " + "  ".join(f"{v:>10}" for v in vals))

PYEOF

echo ""
echo "Full per-epoch histories in: ${SAVE_BASE}/<cancer>/<opt>_history.json"
echo "Logs in: ${LOG_DIR}/"
