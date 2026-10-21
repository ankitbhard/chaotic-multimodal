# Dynamics-Aware AdamW: 18-run EC2 experiment

This folder is isolated from the original Adam/AdamW benchmark. It reuses the original data loading, models, splits, scoring, and cosine learning-rate schedule. Run it from the repository root on branch `experiments/adam-adamw-overnight`.

## GPU preflight

```bash
python -m py_compile experiments/dynamics_aware_ec2/*.py
python -c 'import torch; assert torch.cuda.is_available()'
python -m experiments.dynamics_aware_ec2.run_dynamic --task survival --cancers brca --folds 1 --epochs 1 --out results/dynamics_aware_smoke
```

WARNING: The smoke-test `--folds 1` may not be valid for cross-validation. Prefer `--folds 3` and expect three short runs.

## Full benchmark (50 epochs)

```bash
nohup bash experiments/dynamics_aware_ec2/run_ec2_overnight.sh > dynamics_aware_overnight.log 2>&1 &
tail -f dynamics_aware_overnight.log
```

Outputs: `results/dynamics_aware_adamw/`, 15 cancer-fold runs + 3 CIFAR-10 seed runs. Existing baseline results are not modified.

Important: original diagnostics use constant beta2 bias correction, so the dynamic runner sets `v_mean` and `effective_lr_p10/p50/p90` to NaN. Update them before moment-level comparisons. This is a prototype; benchmark improvement is not established. Before publishing results, audit survival splits, validation selection, and convergence targets. Check training logs for GPU errors and data-path availability.
