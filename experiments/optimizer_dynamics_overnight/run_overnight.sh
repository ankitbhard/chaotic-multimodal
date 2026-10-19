#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
mkdir -p logs/optimizer_dynamics_overnight
echo "Starting Adam/AdamW overnight dynamics experiment at $(date)"
python -m experiments.optimizer_dynamics_overnight.run --task survival --cancers brca kirc luad lusc stad --folds 3 --epochs 50 --log-every 50 --skip-existing 2>&1 | tee -a logs/optimizer_dynamics_overnight/survival.log
python -m experiments.optimizer_dynamics_overnight.run --task classification --epochs 50 --seeds 42 43 44 --batch-size 128 --log-every 250 --skip-existing 2>&1 | tee -a logs/optimizer_dynamics_overnight/classification.log
echo "Finished at $(date)"
