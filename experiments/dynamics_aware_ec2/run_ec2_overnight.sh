#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
python -m experiments.dynamics_aware_ec2.run_dynamic --task both --epochs 50 --folds 3 --seeds 42 43 44 --skip-existing "$@"
