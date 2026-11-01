#!/usr/bin/env bash
# Unattended EC2 benchmark for Information Acquisition Diagnostics.
# 3 cancers × 3 methods × 3 folds = 27 runs, 50 epochs each.
# Shuts down EC2 on completion or failure.
set -Eeuo pipefail
cd "$(git rev-parse --show-toplevel)"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
OUT="${OUT:-results/information_acquisition_diagnostics}"
LOGDIR="${LOGDIR:-run_logs}"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/info_acq_$(date -u +%Y%m%dT%H%M%SZ).log"
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
echo "Host: $(hostname); output: $OUT"

# Verify CUDA
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA GPU not available"
print("CUDA:", torch.cuda.get_device_name(0))
PY

# Verify configs
for cancer in brca kirc luad; do
  test -s "configs/$cancer.yaml" || { echo "Missing configs/$cancer.yaml"; exit 1; }
done

# Run unit tests
python -m pytest experiments/samo/test_information_diagnostics.py -v

echo "=== 27-run information acquisition benchmark ==="
echo "=== 3 cancers × 3 methods × 3 folds × 50 epochs ==="
python -m experiments.samo.run_info_acquisition \
  --cancers brca kirc luad \
  --methods baseline adaptive_aux rank \
  --folds 3 --epochs 50 --seed 42 \
  --out "$OUT" --skip-existing

echo "SUCCESS: all 27 information acquisition runs finished."
