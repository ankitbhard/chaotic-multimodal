#!/usr/bin/env bash
# SGC-AdamW experiment orchestrator
# Usage: bash experiments/sgc_adamw/run_all.sh [--smoke] [--skip-existing]
#
# Phase 1: 7 optimizers × 2 cancers × 3 folds = 42 runs
# Expected runtime: ~30-60 min on T4 (data cached after first cancer/fold)
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-/opt/pytorch/bin/python3}"
OUT="results/sgc_adamw_kirc_stad"
EXTRA_ARGS=""

for arg in "$@"; do
    case "$arg" in
        --smoke) EXTRA_ARGS="--epochs 2 --cancers kirc --optimizers adamw_b999 sgc_drift_alignment" ;;
        --skip-existing) EXTRA_ARGS="$EXTRA_ARGS --skip-existing" ;;
    esac
done

echo "=== SGC-AdamW Experiment Matrix ==="
echo "Output: $OUT"
echo "Python: $PYTHON"
echo "Extra: $EXTRA_ARGS"
echo "Start: $(date)"

# Run unit tests first
echo "=== Running unit tests ==="
$PYTHON -m pytest experiments/sgc_adamw/test_sgc.py -v --tb=short 2>&1 | tail -20
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "UNIT TESTS FAILED — aborting"
    exit 1
fi
echo "=== Tests passed ==="

# Phase 1: Full matrix
echo "=== Phase 1: 7 optimizers × KIRC/STAD × 3 folds ==="
$PYTHON -m experiments.sgc_adamw.run_sgc \
    --cancers kirc stad \
    --folds 3 \
    --epochs 50 \
    --out "$OUT" \
    --log-every 50 \
    --curvature-interval 10 \
    $EXTRA_ARGS

echo "=== Phase 1 complete: $(date) ==="

# Summary
echo "=== Results ==="
find "$OUT" -name summary.json | while read f; do
    cancer=$(echo "$f" | sed 's|.*/\(kirc\|stad\)/.*|\1|')
    fold=$(echo "$f" | sed 's|.*fold_\([0-9]*\).*|\1|')
    opt=$(echo "$f" | sed 's|.*/fold_[0-9]*/\([^/]*\)/.*|\1|')
    score=$($PYTHON -c "import json; print(json.load(open('$f'))['test_score'])")
    printf "%-6s fold=%s %-25s test_c=%.4f\n" "$cancer" "$fold" "$opt" "$score"
done | sort

echo "=== ALL DONE: $(date) ==="
