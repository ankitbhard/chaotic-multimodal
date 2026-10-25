# SAMO: Survival-Aware Multi-Objective Optimization

**Implementation staged; benchmarks have NOT been run.** Independent branch `experiment/samo-survival-optimization` from `experiments/adam-adamw-overnight`. Existing experiments are unchanged.

Components: event-normalized Breslow Cox loss; censoring-aware pairwise logistic ranking loss (only earlier observed events are comparable); image/gene auxiliary Cox heads; bounded EMA loss-ratio adaptive objective weights. Adaptive weighting is a heuristic, not a proven optimizer.

Six methods: `baseline`, `aux`, `rank`, `aux_rank`, `adaptive_aux`, `samo`. All use AdamW and the original cosine LR schedule. Three cohorts (BRCA, KIRC, STAD), three folds, 50 epochs = 54 runs.

Run later, after other jobs finish:

```bash
bash experiments/samo/run_all.sh
```

For tests only: `python -m unittest experiments.samo.test_samo -v`.

**Important methodological differences:** Original benchmark uses a different Cox loss normalization (mean over all batch samples). This experiment uses event-normalized Cox loss for all methods, so its `baseline` must be used for internal comparisons; historical scores are not directly interchangeable. Auxiliary variants add head parameters and are initialized with matching original-model tensors where names/shapes agree. The simple controller uses training loss ratios, not gradient balancing. No results or scientific improvement are claimed.

Outputs under `results/samo_survival/`. The script starts a one-epoch BRCA smoke test before the 54-run matrix. It does not provision GPUs or launch cloud instances. Existing config YAMLs must point to available data.

Known follow-up validation before full execution: review Cox risk-set/tie conventions, inspect model parameter-key matching, test one actual dataset batch, and confirm comparable fold splits and C-index orientation. The script is not a substitute for an audited end-to-end run.
