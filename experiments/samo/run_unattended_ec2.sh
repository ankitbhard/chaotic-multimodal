#!/usr/bin/env bash
# Unattended EC2 benchmark. Run from repository root on a dedicated instance.
# WARNING: By default powers off the EC2 guest when the job exits, even on error.
set -Eeuo pipefail
cd "$(git rev-parse --show-toplevel)"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
OUT="${OUT:-results/samo_all_cancers_50ep}"
LOGDIR="${LOGDIR:-run_logs}"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/unattended_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1

finish() {
  code=$?
  trap - EXIT
  echo "Finished with exit status $code at $(date -u). Log: $LOG"
  sync
  if [[ "${SHUTDOWN_ON_EXIT:-1}" == "1" ]]; then
    echo "Shutting down EC2 guest in 2 minutes; check EC2 instance-initiated shutdown behavior (stop vs terminate)."
    sudo -n shutdown -h +2 || echo "WARNING: shutdown failed (check sudo permissions)."
  fi
  exit "$code"
}
trap finish EXIT

echo "Branch: $(git branch --show-current); commit: $(git rev-parse HEAD)"
echo "Host: $(hostname); output: $OUT"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA GPU not available; refusing expensive CPU-only run"
print("CUDA:", torch.cuda.get_device_name(0))
PY
# Verify required data configuration files exist before launching expensive runs.
for cancer in brca kirc luad lusc stad; do
  test -s "configs/$cancer.yaml" || { echo "Missing configs/$cancer.yaml"; exit 1; }
done
python -m unittest experiments.samo.test_progress_controller experiments.samo.test_representation_gradients experiments.samo.test_history_controller

echo "PHASE 1: All existing methods, 5 cancers x 5 folds x 50 epochs."
python -m experiments.samo.run \
  --cancers brca kirc luad lusc stad \
  --methods baseline aux rank aux_rank adaptive_aux samo ra_samo ra_samo_no_proj ra_samo_v1 \
  --folds 5 --epochs 50 --seed 42 \
  --out "$OUT" --skip-existing --modality-diagnostics

echo "PHASE 2: RA-SAMO v2 leave-one-cancer-out historical controller."
# Never train the v2 controller on trajectories from its target cancer.
for cancer in brca kirc luad lusc stad; do
  history=()
  for source in brca kirc luad lusc stad; do
    [[ "$source" == "$cancer" ]] && continue
    for fold in 0 1 2 3 4; do
      for method in samo ra_samo_v1; do
        file="$OUT/$source/fold_$fold/$method/epochs.csv"
        test -s "$file" || { echo "Missing history: $file"; exit 1; }
        history+=("$file")
      done
    done
  done
  echo "V2 target=$cancer; source histories=${#history[@]} (other cancers only)"
  python -m experiments.samo.run \
    --cancers "$cancer" --methods ra_samo_v2 \
    --folds 5 --epochs 50 --seed 42 \
    --out "$OUT" --skip-existing --modality-diagnostics \
    --history-files "${history[@]}"
done

echo "SUCCESS: all existing-method and v2 runs finished."
