#!/usr/bin/env bash
# Run only after completing the readiness checklist in V3_AGENT_HANDOFF.md.
set -Eeuo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT="${OUT:-results/ra_samo_v3_50ep}"
python -m unittest experiments.samo.test_censor_aware_controller
python -m experiments.samo.run --cancers brca kirc luad lusc stad \
  --methods ra_samo_v3_0 ra_samo_v3_1 ra_samo_v3_2 ra_samo_v3_3 \
  --folds 5 --epochs 50 --seed 42 --out "$OUT" --skip-existing --modality-diagnostics
