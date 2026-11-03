#!/usr/bin/env bash
# Unattended EC2 run: CLAM Attention MIL 2x3 experiment.
# Powers off on completion. Run from repository root.
#
# Usage:
#   bash experiments/clam_brca/run_ec2.sh              # full 6x3
#   SMOKE=1 bash experiments/clam_brca/run_ec2.sh      # 2-epoch smoke test
set -Eeuo pipefail
cd "$(git rev-parse --show-toplevel)"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
LOGDIR="run_logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/clam_brca_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

finish() {
  code=$?
  trap - EXIT
  echo "Finished with exit status $code at $(date -u). Log: $LOG"
  sync
  if [[ "${SHUTDOWN_ON_EXIT:-1}" == "1" ]]; then
    echo "Shutting down EC2 guest in 2 minutes."
    sudo -n shutdown -h +2 || echo "WARNING: shutdown failed."
  fi
  exit "$code"
}
trap finish EXIT

echo "Branch: $(git branch --show-current); commit: $(git rev-parse HEAD)"
echo "Host: $(hostname)"

# CUDA check
/opt/pytorch/bin/python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA GPU not available"
print("CUDA:", torch.cuda.get_device_name(0))
PY

# Config check
test -s configs/brca_clam.yaml || { echo "Missing configs/brca_clam.yaml"; exit 1; }

# Unit tests (CPU only, fast)
echo "Running unit tests..."
/opt/pytorch/bin/python3 -m pytest experiments/clam_brca/test_clam.py -v --tb=short

if [[ "${SMOKE:-0}" == "1" ]]; then
    echo "SMOKE TEST: 2 epochs, 1 fold, 2 methods"
    /opt/pytorch/bin/python3 -m experiments.clam_brca.train \
        --config configs/brca_clam.yaml \
        --methods clam_adamw clam_ra_samo \
        --folds 1 --epochs 2 --skip-existing
else
    echo "FULL RUN: 6 methods x 3 folds x 50 epochs"
    /opt/pytorch/bin/python3 -m experiments.clam_brca.train \
        --config configs/brca_clam.yaml \
        --methods meanpool_adamw meanpool_aux_adamw meanpool_ra_samo \
                  clam_adamw clam_aux_adamw clam_ra_samo \
        --skip-existing
fi

echo "Generating analysis..."
/opt/pytorch/bin/python3 -m experiments.clam_brca.analyze \
    --results-dir results/clam_brca

echo "SUCCESS: CLAM BRCA experiment complete."
