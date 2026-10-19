# Adam and AdamW overnight training dynamics

Isolated experiment branch: `experiments/adam-adamw-overnight`. No existing PCRF scripts are modified.

## EC2 commands

```bash
cd ~/chaotic-multimodal
git fetch origin
git switch experiments/adam-adamw-overnight
python -m py_compile experiments/optimizer_dynamics_overnight/run.py
nohup bash experiments/optimizer_dynamics_overnight/run_overnight.sh > overnight.log 2>&1 &
tail -f overnight.log
```

The launcher runs **five cancer cohorts** (BRCA/KIRC/LUAD/LUSC/STAD), 3 folds each, Adam and AdamW, then **CIFAR-10** ResNet-18, 3 seeds, Adam and AdamW. Each run trains for **50 epochs** (36 runs total). This can exceed one night on a T4.

**Before the full run**, check GPU, datasets, `torchvision`, and execute a small smoke run:

```bash
python -c 'import torch,torchvision; print(torch.__version__,torch.cuda.is_available(),torchvision.__version__)'
python -m experiments.optimizer_dynamics_overnight.run --task survival --cancers brca --folds 2 --epochs 1 --log-every 1 --out results/optimizer_dynamics_smoke
```

The full launcher does not run the smoke test automatically. It exits on an error and logs progress. Results go to `results/optimizer_dynamics_overnight/`. Completed runs are skipped when restarted, but interrupted runs restart from epoch 1.

Step-level CSV includes gradient norm, first and second Adam moments, effective LR percentiles, parameter update norms, update-to-weight ratio, gradient-update cosine, sampled update-to-update cosine, and first-order predicted change. Epoch-level CSV includes training loss, validation loss, validation accuracy/C-index and learning rate. Best-validation checkpoint is evaluated on the held-out test set.

**Caveats:** This is a dynamics investigation, not tuned optimizer benchmarking. Adam's weight decay is coupled; AdamW's is decoupled. For clean optimizer comparisons, run a zero-weight-decay control. Hessian curvature, gradient variance, and measured post-update loss are not included. Survival validation loss uses mini-batch Cox likelihood and should not be treated as a full-risk-set likelihood. Classification training augmentation is random and not cached identically between optimizer runs.
