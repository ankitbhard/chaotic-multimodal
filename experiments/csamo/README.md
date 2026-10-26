# C-SAMO: Concordance-guided objective selection

**Implementation committed, NOT executed on EC2.** This branch is isolated from the original SAMO experiments.

C-SAMO trains a single multimodal auxiliary-head model with standard AdamW. It periodically tests four candidate **one-step training updates** from the same live model and optimizer state: fusion Cox, fusion + auxiliary Cox, fusion + ranking, fusion + auxiliary + ranking. Each hypothetical candidate is evaluated on a controller-only validation subset. The winning objective (subject to a minimum C-index improvement threshold) is used for the next training interval. Candidate updates are discarded; the live optimizer/model are not modified during selection.

- Initial mode: aux_rank
- Warmup: 5 epochs; decision interval: 5 epochs
- Selection threshold: +0.01 C-index over current candidate
- Weights: 0.1 for image Cox and gene Cox (when active); 0.1 for ranking (when active)
- Fixed random split of the **original validation dataset** into controller (50%) and checkpoint (50%) partitions, disjoint within each fold.
- Checkpoints selected by checkpoint-validation C-index only; test set used once after selecting best checkpoint.
- Same train/test fold builder, AdamW, cosine LR, and 50-epoch budget as original SAMO.
- 3 cancers x 3 folds = 9 full runs. **Each controller decision requires four deep-copied model+optimizer one-step updates and four validation evaluations**, so runtime/memory may be significantly higher.

Run from repository root:
```bash
git fetch origin
git switch experiment/csamo-adaptive
git pull --ff-only origin experiment/csamo-adaptive
bash experiments/csamo/run_all.sh
```

The script runs unit tests, a six-epoch BRCA smoke run (three folds, first controller decision), nine 50-epoch runs, then summarizes results.

To run only one fold-equivalent smoke experiment, the runner currently interprets --folds as the **total cross-validation fold count**. Do not change it to 1, as the dataset splitter may require multiple folds.

Outputs: `results/csamo_survival/<cancer>/fold_<n>/csamo/{epochs.csv,decisions.csv,summary.json}` and `experiments/csamo/RESULTS_SUMMARY.csv`.

## Scientific limitations and mandatory checks

- This is an **experimental validation-guided controller**, not an established improvement. Results have not been generated or verified.
- Controller validation is repeatedly queried; adaptive overfitting remains possible. For publication, repeat across seeds and use nested evaluation / independent external cohorts.
- Validation is halved, which changes checkpoint selection relative to the original SAMO benchmark. To make **strictly matched comparisons**, rerun fixed-objective baselines with the same controller/checkpoint split, same auxiliary architecture and same checkpoint protocol.
- One-step C-index differences may be noisy or tied; a +0.01 switching threshold limits but does not eliminate this problem.
- Counterfactuals use one training minibatch and AdamW state; they are not multi-epoch forecasts.
- Extra auxiliary heads are present even when disabled. Parameter count differs from original fusion-only baseline.
- Check original Cox tie/risk-set semantics, data splits, event counts per validation subset, and model shared-weight initialization before interpreting results.
- If the GPU cannot fit four sequential deep copies or model copying fails, stop and revise the lookahead implementation.
