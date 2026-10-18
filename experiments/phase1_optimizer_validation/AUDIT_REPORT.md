# Phase 1 Audit Report — Next-Generation Multimodal Optimization

**Date:** 2026-10-17
**Branch:** `experiments/modality-sharpness-pilot`
**Commit:** `7f913babf1c061cf5701006e238e5a84bfccd42a`
**Auditor:** Claude (automated audit)

---

## Summary

This audit examines the correctness of data handling, the Cox survival objective,
the C-index metric, and all optimizer implementations used in the exploratory
30-epoch BRCA fold-0 experiment (results in `results_nextgen_adamw/`).

| Severity | Count | Description |
|----------|-------|-------------|
| **Critical** | 1 | LR scheduler–adaptive controller conflict creates unfair comparison |
| **Major** | 3 | Broken local `joint_model.py`, disentanglement changes architecture, zero-event batch behavior |
| **Minor** | 3 | Silent embedding fallback, n_top_genes discrepancy, no checkpoint resume |
| **Informational** | 4 | Various observations |

---

## CRITICAL Findings

### C1. Cosine scheduler is ineffective for adaptive optimizers — unfair comparison

**Severity:** CRITICAL
**Component:** `experiments/next_gen_optimization/run_experiments.py` (lines 133–189)
**Affects:** `adaptive_loss_ema`, `adaptive_gradnorm`, all hybrid optimizers

**Description:**
The experiment runner creates a `CosineAnnealingScheduler` for every method
(line 135) and calls `scheduler.step()` after each batch (line 189). For the
AdamW baseline, this correctly anneals the LR from 0.001→0 over training.

However, for adaptive optimizers (`LossEMAAdaptiveOptimizer`, etc.), the
optimizer's `step()` method overwrites `param_groups["lr"]` with
`self.reference_lr * scale` (adaptive_step.py:117–119) before calling
`self.base.step()`. The value of `self.reference_lr` is set once at init
(adaptive_step.py:87) and never updated.

**Sequence per batch:**
1. `opt.step(closure)` → sets LR = `reference_lr * scale` → applies gradients
2. `scheduler.step()` → sets LR = cosine-scheduled value (immediately overwritten next step)

**Effect:**
- **AdamW baseline:** Gets cosine annealing (LR decays from 1e-3 to ~0)
- **Adaptive methods:** Run with flat reference LR=1e-3, modulated only by loss-EMA
- **The cosine scheduler does nothing for adaptive methods**

**Impact on reported results:**
The comparison is unfair. AdamW with cosine annealing is compared against
adaptive methods without cosine annealing. Any improvement could be attributed
to (a) the adaptive mechanism, or (b) the absence of aggressive LR decay,
or (c) a combination. This **confounds** the comparison entirely.

**Resolution options:**
1. Apply cosine annealing to `reference_lr` over time (adaptive modulation on top of scheduled LR)
2. Remove cosine annealing from all methods (compare with flat LR)
3. Run both configurations as explicit ablations

**Recommendation:** Option 3 — run all methods both with and without cosine
scheduling as a controlled ablation. For the primary benchmark, use option 1
(adaptive modulation on top of scheduled baseline) since this is the most
scientifically defensible comparison.

---

## MAJOR Findings

### M1. Working-copy `models/joint_model.py` is missing `JointModel` class

**Severity:** MAJOR
**Component:** `models/joint_model.py` (local working copy)

**Description:**
`git status` shows `M models/joint_model.py`. The working copy is missing
the entire `JointModel` class definition (lines 34–77 in git HEAD). The file
jumps from imports directly to `build_model()`, which references `JointModel`
in its return-type annotation, causing `NameError` on import.

This prevents importing `train_survival.py` in the local environment.
The EC2 instance may have a different version (pulled from git or separately
uploaded).

**Impact:** Any code that imports `train_survival` or `models.joint_model`
will fail locally. Tests were run using the `cox_loss_simple` workaround in
test_optimizers.py.

**Resolution:** Restore `joint_model.py` from git: `git checkout -- models/joint_model.py`

### M2. Disentanglement methods change the training objective — not a pure optimizer comparison

**Severity:** MAJOR
**Component:** `gradient_disentanglement.py`, `hybrid_optimizers.py`

**Description:**
`disentangle_aux`, `disentangle_balance`, and `hybrid_disentangle_adaptive`
add two auxiliary `nn.Module` heads (each: Linear(feat_dim→64)→ReLU→Linear(64→1))
and train with an augmented loss:

    L_total = L_fusion + 0.1 * (L_image_aux + L_gene_aux)

This is **not** a change to the optimizer — it is a change to the **loss function
and architecture**. The auxiliary heads add ~8,500 extra trainable parameters.

At test time, auxiliary heads are discarded and evaluation uses only the
fusion-based prediction, identical to the baseline.

**Impact on interpretation:**
Comparing `disentangle_balance` (C-index 0.6944) against AdamW (0.6669) is
**not** an optimizer comparison. It is a comparison of different training
objectives. Any improvement could be due to (a) the auxiliary supervision
signal, (b) the gradient balancing, (c) the different effective regularization,
or (d) random variation on a single fold.

**Resolution:** Clearly document this distinction. Do not present disentanglement
results as optimizer improvements. Run ablations separating auxiliary loss from
gradient balancing.

### M3. Cox loss behavior with zero-event batches

**Severity:** MAJOR
**Component:** `train_survival.py:cox_loss()` (lines 77–91)

**Description:**
When a batch contains zero events (`vital_status` all zeros), the loss computes:
```
loss = -torch.mean((risk - log_cum_sum) * events)
     = -torch.mean(0) = 0
```
This produces zero loss and zero gradients for that batch. The optimizer step
still executes, but weight decay is applied without any task-specific gradient.

With BRCA (146 events / 1052 patients = 13.9% event rate) and batch_size=32,
a zero-event batch occurs with probability (1-0.139)^32 ≈ 0.8% per batch.
Over 30 epochs × ~23 batches = ~690 steps, this is ~5.5 wasted steps.

**Impact:** Minimal in practice but creates inconsistency. Weight decay is
applied while no survival signal is received. Adaptive optimizers that track
loss EMA will see spurious zero-loss values, pulling the EMA down and briefly
inflating the LR multiplier.

**Resolution:** Guard the loss: if `events.sum() == 0`, return a small
constant or skip the optimizer step. For Phase 1, document but do not change
(to preserve reproducibility of existing experiments).

---

## MINOR Findings

### m1. Silent embedding fallback to zeros

**Severity:** MINOR
**Component:** `train_survival.py:_load_embedding()` (line 97–98)

**Description:**
If an embedding file is missing, the function returns `torch.zeros(image_dim)`
without logging a warning. Patients with zero-vector embeddings will still be
included in training.

**Impact:** In practice, BRCA matching is good (1052 patients matched). But
if some embeddings are silently missing, the model receives uninformative
image features for those patients, diluting the signal.

**Resolution:** Add a warning counter. Not blocking for Phase 1.

### m2. n_top_genes discrepancy

**Severity:** MINOR
**Component:** Manifest shows `n_top_genes: 20000` in config, but actual gene
count in training is n_genes=1000.

**Description:**
The `build_survival_dataloaders()` function selects top-K genes by variance.
The config says 20000, but training logs show "n_genes=1000". This may be
because only ~1000 gene columns exist in the BRCA CSV files after filtering
non-gene columns, or the CSV files contain exactly 1000 gene features.

**Impact:** Affects model capacity and comparison with other cohorts. Not a
correctness issue but should be documented.

### m3. No checkpoint resume support in next-gen runner

**Severity:** MINOR
**Component:** `experiments/next_gen_optimization/run_experiments.py`

**Description:**
If training is interrupted, there is no way to resume from a checkpoint.
The runner must restart from scratch. For 200-epoch runs, this is a risk.

**Resolution:** Add checkpoint saving every N epochs and resume logic for
Phase 1 runner.

---

## INFORMATIONAL Findings

### I1. AdamW and grad_diagnostic produce identical results

The manifest shows both `adamw` and `grad_diagnostic` achieve test C-index
0.6669139465875371 (identical to 16 decimal places) and best_val 0.654320987654321.
This is expected: `GradientDiagnosticOptimizer` logs conflict metrics but does
not modify gradients.

### I2. Risk score extraction from 2-output head

The model uses `Linear(256, 2)` as the survival head and extracts `output[:,1]`
as the risk score. This is a legacy design (originally for binary survival
classification). For Cox training, a single-output head would be more natural.
However, this is not a correctness issue — the second logit is a valid
unbounded risk score.

### I3. Per-batch scheduler stepping

The cosine scheduler is stepped per batch, not per epoch. With
`T_max = epochs * batches_per_epoch`, this produces smooth cosine decay.
This is correct but differs from the epoch-level scheduling used in
`train_survival.py`'s `train_one_optimizer()`.

### I4. Fold seed calculation

`seed = int(cfg.get("seed", 42)) + args.fold` — the model initialization
seed varies by fold. This means fold 0 uses seed 42, fold 1 uses seed 43, etc.
The data split seed is always 42 (passed directly to `build_survival_dataloaders`).
This is correct: same splits but different model initializations per fold.

---

## A1. Data Integrity Verification

| Check | Status | Notes |
|-------|--------|-------|
| Patient-level separation | **PASS** | StratifiedKFold on vital_status, 12-char barcode dedup |
| No patient overlap | **PASS** | sklearn's StratifiedKFold guarantees disjoint folds |
| Normalization fit on train only | **PASS** | StandardScaler.fit_transform(train), transform(val/test) |
| Patient-to-image matching | **PASS** | Barcode[:12] matches embedding filename[:12] |
| Multi-slide handling | **PASS** | `drop_duplicates(subset="patient_barcode")` keeps first |
| Censoring definition | **PASS** | vital_status=1 (died), 0 (censored); OS_months continuous |
| Identical folds across optimizers | **PASS** | Same seed (42) and fold index passed to all methods |

## A2. Cox Survival Objective Verification

| Check | Status | Notes |
|-------|--------|-------|
| Event encoding | **PASS** | 1=event (death), 0=censored |
| Risk direction | **PASS** | Higher risk_score → higher hazard (correct for Cox) |
| Sort order | **PASS** | Descending by OS_months (logcumsumexp accumulates risk set) |
| Tied events | **WARN** | Breslow approximation (standard, acceptable) |
| Zero-event batches | **MAJOR M3** | Returns zero loss; see finding M3 |
| C-index correctness | **PASS** | Standard Harrell's C-index implementation |

## A3. Optimizer Correctness Verification

| Check | AdamW | Loss-EMA | Disentangle | SAM/M-SAM |
|-------|-------|----------|-------------|-----------|
| Gradient zeroing | PASS | PASS | PASS | PASS |
| Parameter updates | PASS | PASS | PASS | PASS |
| Param groups | PASS | PASS (single group) | PASS (includes aux) | PASS |
| LR scheduling | PASS | **CRITICAL C1** | **CRITICAL C1** | PASS |
| Weight decay | PASS | PASS | PASS | PASS |
| Gradient clipping | N/A | N/A | N/A | N/A |
| Numerical stability | PASS | PASS (bounded scale) | PASS | PASS |

### Loss-EMA specific:
| Check | Status | Notes |
|-------|--------|-------|
| EMA equation | PASS | `ema = decay * ema + (1-decay) * val` |
| Loss value timing | PASS | Current batch loss used immediately |
| LR bounds | PASS | [0.5x, 1.5x] reference_lr |
| Warm-up | NONE | First step initializes EMA to current loss |
| AdamW compatibility | PASS | Wraps AdamW as base |
| Zero-event behavior | WARN | Zero loss → EMA drops → LR inflates briefly |
| External scheduler interaction | **CRITICAL C1** | Scheduler is ineffective |

### Disentanglement specific:
| Check | Status | Notes |
|-------|--------|-------|
| Auxiliary Cox objectives | PASS | Inline _cox_loss (same formula as main) |
| Gradient routing | PASS | Single backward through total loss |
| Parameter ownership | PASS | model params + aux_heads params in one optimizer |
| Gradient balancing | PASS | Geometric mean target, scales encoder grads |
| Aux heads at test time | PASS | Not used; evaluate() calls standard model.forward() |
| Same inference head | PASS | head_survival is identical |

---

## Recommendations for Phase 1

### Must fix before benchmark:
1. **C1:** Implement proper LR scheduling for adaptive methods (apply adaptive
   modulation on top of cosine-scheduled reference LR)
2. **M1:** Restore `joint_model.py` from git HEAD

### Must document:
3. **M2:** Disentanglement is a training-objective change, not a pure optimizer comparison
4. **M3:** Zero-event batch behavior (accept as-is for reproducibility)

### Nice to have:
5. **m3:** Add checkpoint saving/resume for 200-epoch runs
6. **m1:** Add embedding-missing warning

---

## Appendix: Original Pilot Configuration

```
Cancer: BRCA
Patients: 1052 (146 events, 13.9% event rate)
Split: fold 0/5 (715 train / 126 val / 211 test)
Epochs: 30
Base optimizer: AdamW
LR: 0.001
Weight decay: 0.0001
Batch size: 32
LR scheduler: CosineAnnealing (per-batch, T_max=690)
Model: JointModel (attention fusion, ~1.45M params)
Image dim: 1024 (UNI)
Gene features: 1000 (after variance selection)
Seed: 42
Initial weights SHA256: ac0a0e14cf84cf1a...
```
