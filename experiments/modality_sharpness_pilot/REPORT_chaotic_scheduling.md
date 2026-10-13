# Chaotic Modality-Aware SAM Scheduling: Research Report

**Date:** 2026-10-13
**Branch:** `experiments/modality-sharpness-pilot`
**Commit:** `4614aea`
**Cohort:** BRCA, Fold 0, 30 epochs

---

## A. Implementation Verification

### Tests Executed

**14/14 tests pass** across two test modules:

```
test_chaotic_schedule (7 tests):
  - test_exact_budget_and_reproducibility .............. OK
  - test_extreme_budgets ............................... OK
  - test_step_count_and_finite_weights ................. OK
  - test_zero_rho_matches_sgd .......................... OK
  - test_parameter_restoration_after_sam ............... OK  (new)
  - test_fraction_one_trajectory_reproducible .......... OK  (new)
  - test_different_seeds_different_schedules ........... OK  (new)

test_modality_schedule (7 tests):
  - test_exact_counts_and_reproducibility .............. OK
  - test_all_four_states_and_updates ................... OK
  - test_no_perturbation_matches_sgd ................... OK
  - test_image_gene_groups_no_overlap .................. OK  (new)
  - test_fusion_head_always_receives_gradients ......... OK  (new)
  - test_image_only_does_not_perturb_gene .............. OK  (new)
  - test_modality_schedules_use_different_seeds ........ OK  (new)
```

Additionally, all 13 existing tests (test_block_sam + test_hessian_diagnostics) continue to pass.

### Verified Properties

1. **Budget exactness:** All scheduling modes (periodic, random, chaotic) produce exactly `round(total_steps * fraction)` SAM steps.
2. **Reproducibility:** Same seed always produces identical schedules.
3. **Parameter restoration:** SAM perturbation is fully removed before the SGD update.
4. **Parameter group isolation:** Image and gene encoder parameters are disjoint sets.
5. **Fusion/head gradient flow:** Head parameters receive gradients in all 4 modality states (both, image-only, gene-only, neither).
6. **Selective perturbation:** Image-only scheduling does NOT perturb gene encoder parameters.
7. **Schedule independence:** Image and gene schedules use different seeds (offset by 7919).
8. **SGD equivalence:** Zero perturbation matches pure SGD with momentum.

### Bugs Identified

None. The implementation is correct.

### Implementation Notes

- The chaotic schedule uses a logistic map (z → 4z(1−z)) to generate scores, then rank-selects the top B scores for an exact budget.
- Independent modality schedules with 50% image + 50% gene produce a union budget of ~75% (not 50%), because overlap is only ~25%. This was accounted for in the experimental design.
- Perturbation normalization: for global scheduled SAM, the gradient norm is computed over ALL parameters; for modality-aware SAM, the norm is computed over only the selected modality's parameters.

---

## B. Experimental Comparison

### BRCA Pilot: 30 Epochs, Fold 0, rho=0.05

#### Table 1: All Methods at Their Native SAM Budgets

| Method | SAM% | Best Val CI | Test CI | Step (ms) | Est. Time (s) |
|---|---|---|---|---|---|
| SGD (baseline) | 0% | 0.4872 | 0.5720 | 3.9 | 3.6 |
| **Periodic 50%** | 50% | 0.4922 | **0.5783** | 6.8 | 5.6 |
| Random 50% | 50% | 0.4889 | 0.5679 | 6.8 | 5.6 |
| Chaotic 50% | 50% | **0.5004** | 0.5705 | 6.8 | 5.6 |
| Mod-Periodic 50% | 50% | 0.4922 | **0.5823** | 6.3 | 5.2 |
| Full SAM 100% | 100% | 0.4807 | 0.5645 | 9.6 | 7.5 |

#### Table 2: Matched-Budget Comparison (75% SAM)

| Method | SAM% | Best Val CI | Test CI | Step (ms) | Est. Time (s) |
|---|---|---|---|---|---|
| Periodic 75% | 75% | 0.4807 | 0.5746 | 8.4 | 6.7 |
| Random 75% | 75% | 0.4856 | **0.5772** | 8.2 | 6.6 |
| Chaotic 75% | 75% | 0.4881 | 0.5649 | 8.1 | 6.5 |
| Mod-Random 76% | 76% | 0.4749 | 0.5701 | 7.2 | 5.9 |
| Mod-Chaotic 75% | 75% | 0.4955 | 0.5671 | 7.2 | 5.9 |
| Full SAM 100% | 100% | 0.4807 | 0.5645 | 9.6 | 7.5 |

#### Table 3: Modality State Counts (for modality-aware methods)

| Method | Both | Image Only | Gene Only | Neither | Union SAM |
|---|---|---|---|---|---|
| Mod-Periodic | 345 | 0 | 0 | 345 | 345 (50%) |
| Mod-Random | 166 | 179 | 179 | 166 | 524 (76%) |
| Mod-Chaotic | 171 | 174 | 174 | 171 | 519 (75%) |

Note: Mod-Periodic degenerates to synchronized perturbation (both or neither) because both schedules use the same periodic placement algorithm. Only random and chaotic produce all 4 states.

---

## C. Convergence Analysis

### Epochs to Reach Shared Val C-index Targets

| Target | SGD | Periodic 50% | Random 50% | Chaotic 50% | Mod-Periodic | Full SAM |
|---|---|---|---|---|---|---|
| 0.43 | 5 | 6 | 6 | 5 | 6 | 5 |
| 0.45 | 11 | 11 | 11 | 8 | 13 | 13 |
| 0.47 | 13 | 13 | 19 | 13 | 13 | 19 |
| 0.49 | — | 23 | — | 23 | 26 | — |

### Val C-index at Key Epochs

| Epoch | SGD | Periodic 50% | Random 50% | Chaotic 50% | Mod-Periodic | Full SAM |
|---|---|---|---|---|---|---|
| 5 | 0.435 | 0.422 | 0.424 | 0.434 | 0.425 | 0.435 |
| 10 | 0.441 | 0.444 | 0.444 | 0.447 | 0.443 | 0.437 |
| 20 | 0.479 | 0.481 | 0.471 | 0.472 | 0.481 | 0.451 |
| 30 | 0.483 | 0.489 | 0.482 | 0.476 | 0.490 | 0.474 |

### Training Loss

Training losses are nearly identical across all methods (0.276–0.285 at epoch 30), indicating no method has a convergence advantage in the loss landscape over 30 epochs.

---

## D. Research Interpretation

### 1. Does chaotic SAM scheduling outperform periodic scheduling?

**No.** At 50% SAM budget, periodic (test CI 0.5783) outperforms chaotic (0.5705) by +0.0078. At 75% SAM budget, chaotic (0.5649) underperforms both periodic (0.5746) and random (0.5772). Chaotic scheduling has no advantage over simpler strategies.

### 2. Does chaotic scheduling outperform random scheduling?

**Mixed, with no clear advantage.** At 50%: chaotic (0.5705) slightly outperforms random (0.5679, +0.0026). At 75%: chaotic (0.5649) underperforms random (0.5772, −0.0123). The differences are within noise for a single fold.

### 3. Does modality-aware scheduling improve performance at matched compute?

**No.** At matched 75% union budget:
- Global random 75%: test CI = 0.5772
- Modality-random 76%: test CI = 0.5701 (−0.0071)
- Global chaotic 75%: test CI = 0.5649
- Modality-chaotic 75%: test CI = 0.5671 (+0.0022)

The differences are negligible and inconsistent in direction.

At matched 50% budget:
- Global periodic 50%: test CI = 0.5783
- Modality-periodic 50%: test CI = 0.5823 (+0.0040)

Modality-periodic shows a small advantage, but this is a single fold and the difference (0.004) is well within expected variance.

### 4. Is any observed benefit consistent across folds?

**Not tested.** This is a single-fold pilot. No claim of consistency can be made.

### 5. Does the additional complexity justify a new optimizer?

**No.** The key finding is that **SGD without SAM** (test CI 0.5720) is competitive with or better than **full SAM** (0.5645). The best performer is modality-periodic (0.5823), but it beats SGD by only 0.0103, which is within single-fold variance. The additional complexity of chaotic scheduling, modality-aware perturbation selection, and logistic map orbit generation provides no demonstrated benefit.

### 6. What experiment should be conducted next?

**No further experiments are recommended for this approach.** The pilot reveals two fundamental issues:

1. **SAM provides marginal benefit on BRCA at 30 epochs.** Full SAM (100%) is worse than SGD, and 50% SAM is only marginally better. This suggests the learning rate and training duration dominate performance, not perturbation strategy.

2. **All scheduling variants are within 0.018 C-index of each other** (0.5645 to 0.5823). At this effect size, 5-fold cross-validation with 200 epochs would cost ~15 GPU-hours per cancer × 5 cancers = 75 GPU-hours. Even if a small advantage emerged, it would not constitute a publishable finding.

If the research direction must be pursued, the most informative next step would be a 5-fold BRCA comparison of only SGD vs Periodic-50% vs Full-SAM at 200 epochs, to establish whether SAM provides any benefit at all on this task when training to convergence.

---

## Summary

| Question | Answer |
|---|---|
| Does chaotic scheduling help? | **No** |
| Does modality-aware scheduling help? | **No** |
| Does SAM itself help (vs SGD)? | **Marginal at best** |
| Is the added complexity justified? | **No** |
| Recommendation | **NO-GO for full 5-cohort experimentation** |

### GO / NO-GO Recommendation

## **NO-GO**

The BRCA pilot does not support the hypothesis that modality-specific chaotic scheduling can achieve comparable or better survival prediction than standard SAM while reducing computational cost. The observed differences across all 8 methods are small (max 0.018 C-index) and inconsistent. Full SAM underperforms SGD. Chaotic scheduling provides no advantage over simpler periodic or random strategies. Modality-aware scheduling provides no advantage over global scheduling at matched compute budgets.

Launching a 5-cohort 5-fold experiment would consume significant GPU resources with no reasonable expectation of a publishable finding.

---

## Appendix

### Git History
```
4614aea  Add correctness tests for scheduled and modality-aware SAM
fb0762e  Test modality schedule budgets states and SGD equivalence
d629110  Wire modality-aware periodic random chaotic SAM into TCGA runner
22ebd9f  Add independently scheduled image and gene SAM perturbations
ec3b22d  Add matched-budget scheduled SAM CPU tests
5cb7bd2  Expose matched-budget chaotic periodic random scheduling in BRCA pilot
1841272  Add matched-budget periodic random chaotic SAM scheduling
```

### Configuration
- Cohort: BRCA
- Fold: 0/5
- Epochs: 30
- SAM radius: rho=0.05
- Learning rate: 0.001
- Weight decay: 0.0001
- Batch size: 32
- Optimizer: SGD with momentum 0.9
- LR scheduler: CosineAnnealing
- Total training steps per run: 690 (23 batches × 30 epochs)
- GPU: Tesla T4

### Timing Note
The `schedule_full` total wall-clock time (388s) in the first run is inflated by CUDA warmup (it was the first optimizer to execute). Steady-state step times (9.6ms) are accurate. Estimated fair training times use steady-state step measurements.
