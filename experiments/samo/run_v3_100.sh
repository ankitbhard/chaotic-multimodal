#!/usr/bin/env bash
# Unattended EC2 benchmark for RA-SAMO v3 (4 variants × 5 cancers × 5 folds).
# Shuts down EC2 on completion or failure.
set -Eeuo pipefail
cd "$(git rev-parse --show-toplevel)"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
OUT="${OUT:-results/ra_samo_v3_50ep}"
LOGDIR="${LOGDIR:-run_logs}"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/v3_unattended_$(date -u +%Y%m%dT%H%M%SZ).log"
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
for cancer in brca kirc luad lusc stad; do
  test -s "configs/$cancer.yaml" || { echo "Missing configs/$cancer.yaml"; exit 1; }
done

# Run all tests
python -m unittest \
  experiments.samo.test_censor_aware_controller \
  experiments.samo.test_progress_controller \
  experiments.samo.test_representation_gradients -v

echo "=== 100-run benchmark: 4 v3 variants × 5 cancers × 5 folds × 50 epochs ==="
python -m experiments.samo.run \
  --cancers brca kirc luad lusc stad \
  --methods ra_samo_v3_0 ra_samo_v3_1 ra_samo_v3_2 ra_samo_v3_3 \
  --folds 5 --epochs 50 --seed 42 \
  --out "$OUT" --skip-existing --modality-diagnostics

echo "SUCCESS: all 100 v3 runs finished."
