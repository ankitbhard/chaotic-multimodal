# SGC-AdamW KIRC/STAD experiment — execution instructions

## Status and source
This branch was created from `experiments/adam-adamw-overnight`, which contains the exact original implementation:
- `experiments/dynamics_aware_ec2/dynamics_optimizer.py` — DynamicsAwareAdamW
- `experiments/dynamics_aware_ec2/run_dynamic.py` — dynamic experiment launcher
- `experiments/optimizer_dynamics_overnight/run.py` — original matched survival/classification training pipeline

The prior SGC prototype is supplied separately in the conversation attachment `sgc_adamw_agent_package.zip`. **The prototype has not been merged or benchmarked yet.** Do not substitute the unrelated LLM-supervisor branch for this experiment.

## Implementation
1. Add `experiments/sgc_adamw/sgc_optimizer.py`, starting from the attached SGC prototype. Keep original DynamicsAwareAdamW byte-for-byte unchanged. Add configurable drift, gradient-alignment, and optional curvature gates, plus a disabled-adaptation mode that matches torch.optim.AdamW.
2. Use per-parameter (or explicitly documented per-group) state, correct cumulative-product variable-beta2 bias correction, decoupled weight decay, sparse-gradient rejection, finite-value checks, and complete state_dict checkpoint restoration. Never overwrite historical experiment directories.
3. For optional curvature gating, measure gradients at theta_t and theta_{t-1} on **the same minibatch**, at a configurable sparse interval; never interpret different-minibatch gradient differences as clean curvature. Record extra forward/backward cost. Ensure no accidental optimizer state changes during the probe and restore model state/mode/RNG as necessary. Make no-curvature the default ablation.
4. Add `experiments/sgc_adamw/run_sgc.py` reusing `experiments.optimizer_dynamics_overnight.run` data loaders, models, splits, metrics, seeds, scheduler and selection logic. Prefer an explicit optimizer factory/refactor over global torch.optim.AdamW monkey-patching; preserve legacy behavior with regression tests.
5. Fix dynamic diagnostics: existing `diagnostics()` computes `v/(1-beta2**step)`, which is wrong for variable beta2. Use `state["beta2_product"]` for corrected v, and log per-tensor beta2, drift EMA, alignment EMA, curvature estimate, update norms and elapsed time. If group values aggregate parameters, document the weighting and handle missing values.
6. Ensure comparison runs do not share checkpoints or overwrite `results/dynamics_aware_adamw`. New outputs under `results/sgc_adamw_kirc_stad/`.

## Test plan (mandatory before GPU)
- Disabled adaptation vs torch.optim.AdamW: multiple steps, deterministic gradients, decoupled WD, beta1/beta2 correction, exact/near-exact updates.
- Variable beta2 cumulative product bias correction: hand-computed two/three-step test.
- Independent parameter groups; missing gradients; sparse gradients; CPU/CUDA when available; zero/NaN/Inf handling; no state leakage.
- Save/reload optimizer state_dict and reproduce next update exactly.
- Curvature probe uses same batch, restores state, no hidden extra updates, and logs compute overhead.
- 10-step smoke test plus one small end-to-end survival run before the full matrix.

## Experiment matrix
Datasets: KIRC, STAD. Folds: 0,1,2. Epochs: 50. Seeds, data partitions, initializations, loss, batch size, LR=0.001, WD=0.0001, cosine schedule identical to previous benchmark. Match training-step budgets.
Optimizers:
- AdamW beta2=0.999
- AdamW beta2=0.95 and beta2=0.98 (fixed beta2 ablations)
- Original DynamicsAwareAdamW beta2_min=0.95, sensitivity=3, change_ema=0.9
- SGC drift-only
- SGC drift+alignment
- SGC drift+alignment+same-minibatch curvature
Record curvature probe frequency and full extra compute.

## Report
- Paired test C-index by fold and mean/std; validation-selection policy; no test-based hyperparameter tuning.
- Epochs/steps AND wall-clock seconds to the same predeclared absolute validation C-index thresholds, with not-reached outcomes clearly marked.
- Validation learning curves and area under quality-vs-compute curve.
- beta2 trajectories, drift/alignment/curvature, update norms, effective LR distributions, overhead.
- Git diff, tests and failures, environment/GPU, commands, result paths, and uncertainty. Do not claim universal improvement from 3 folds.
- Stop after KIRC/STAD; request approval before wider cancer/LLM experiments.

## Suggested execution order
1. Inspect attached prototype and original runner; prepare integration PR.
2. Run unit tests and CPU smoke.
3. Verify EC2 GPU/data paths; run a short KIRC smoke.
4. Run paired KIRC/STAD 50-epoch matrix only after smoke passes.
5. Publish evidence-based report; do not modify main.
