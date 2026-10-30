#!/usr/bin/env bash
# Run from repo root. Data paths must be configured in configs/*.yaml.
set -euo pipefail
python -m experiments.samo.run \
  --cancers brca kirc luad lusc stad \
  --methods baseline aux rank aux_rank adaptive_aux samo ra_samo ra_samo_no_proj ra_samo_v1 \
  --folds 5 --epochs 50 --seed 42 \
  --out results/samo_all_cancers_50ep \
  --skip-existing --modality-diagnostics
