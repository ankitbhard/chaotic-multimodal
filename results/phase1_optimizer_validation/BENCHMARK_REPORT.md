# Phase 1 Benchmark Report — Rigorous Optimizer Validation

**Date:** 2026-10-17
**Cancer:** BRCA (1052 patients, 146 events, 13.9% event rate)
**Folds:** 5-fold stratified CV by vital_status
**Epochs:** 200
**Base optimizer:** AdamW (lr=0.001, weight_decay=0.0001) for ALL methods
**LR Schedule:** Epoch-level cosine annealing (eta_min=0) unless otherwise noted
**Checkpoint:** Best validation C-index
**Model:** JointModel (attention fusion, 1.45M params)

---

## Critical Correction (Audit Finding C1)

The original pilot had a critical bug: adaptive optimizers used a **fixed reference LR**
while AdamW used cosine annealing. This confounded all comparisons. This benchmark corrects
the issue: ALL methods share the same epoch-level cosine schedule. Adaptive methods modulate
the **cosine-scheduled** LR (not a fixed reference). SAM/M-SAM also use AdamW base with
externally-managed cosine LR.

---

## Stage C: Primary Benchmark (5 Methods × 5 Folds × 200 Epochs)

| Method | Test C-index | Val C-index | Best Epoch | Time/fold |
|--------|-------------|-------------|------------|-----------|
| SAM | **0.6702 ± 0.0385** | 0.7031 ± 0.0246 | 48 | 57.4s |
| Adaptive LR (Loss-EMA) | 0.6635 ± 0.0256 | 0.7331 ± 0.0379 | 40 | 28.5s |
| AdamW (baseline) | 0.6519 ± 0.0474 | 0.7290 ± 0.0253 | 61 | 374.1s* |
| M-SAM | 0.6366 ± 0.0392 | 0.7243 ± 0.0165 | 28 | 419.8s |
| Disentangle + Balance | 0.6336 ± 0.0400 | 0.7565 ± 0.0347 | 61 | 42.5s |

*AdamW timing includes CUDA warmup (~370s) on fold 0; actual training is ~28s/fold.

### Per-Fold Test C-index

| Method | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|--------|--------|--------|--------|--------|--------|
| AdamW | 0.6717 | 0.5952 | 0.5959 | 0.7078 | 0.6890 |
| Adaptive LR | 0.6866 | 0.6181 | 0.6677 | 0.6885 | 0.6566 |
| Disentangle + Balance | 0.6484 | 0.6258 | 0.5738 | 0.6971 | 0.6228 |
| SAM | 0.6988 | 0.6411 | 0.6135 | 0.7201 | 0.6775 |
| M-SAM | 0.6246 | 0.6483 | 0.5733 | 0.6947 | 0.6419 |

### Statistical Analysis (Paired t-tests vs AdamW)

| Method | Δ C-index | t-statistic | p-value | Significant? |
|--------|-----------|-------------|---------|-------------|
| SAM | +0.0183 | 1.943 | 0.1239 | No |
| Adaptive LR (Loss-EMA) | +0.0116 | 0.635 | 0.5602 | No |
| Disentangle + Balance | -0.0183 | -1.186 | 0.3011 | No |
| M-SAM | -0.0154 | -0.835 | 0.4507 | No |

### Generalization Gap

| Method | Val CI | Test CI | Gap | Assessment |
|--------|--------|---------|-----|-----------|
| SAM | 0.7031 | 0.6702 | 0.033 | Best generalization |
| Adaptive LR | 0.7331 | 0.6635 | 0.070 | Good |
| AdamW | 0.7290 | 0.6519 | 0.077 | Moderate |
| M-SAM | 0.7243 | 0.6366 | 0.088 | Moderate |
| Disentangle | 0.7565 | 0.6336 | 0.123 | Overfitting |

---

## Stage D1: Disentanglement Component Ablation

Separating the two components of `disentangle_balance`:
- **Auxiliary heads:** Unimodal Cox prediction heads (training-only)
- **Gradient balancing:** Scale encoder gradients to geometric mean of their norms

| Method | Test C-index | Δ vs AdamW | p-value |
|--------|-------------|-----------|---------|
| AdamW (baseline) | 0.6519 ± 0.0474 | — | — |
| + Grad Balance only | 0.6526 ± 0.0430 | +0.0007 | 0.960 |
| **+ Aux heads only** | **0.6723 ± 0.0528** | **+0.0204** | **0.007*** |
| + Aux + Grad Balance (full) | 0.6336 ± 0.0400 | -0.0183 | 0.301 |

### Key Finding (D1)

**Auxiliary heads alone provide a statistically significant improvement (p=0.007).**
This is the only statistically significant result across all comparisons. However,
adding gradient balancing on top *reverses* the improvement (Δ = -0.039 from aux-only
to full disentangle). The gradient norm balancing appears to disrupt the beneficial
signal from auxiliary supervision.

**Interpretation:** Unimodal auxiliary heads act as an effective regularizer by forcing
each encoder to produce individually predictive features. Gradient balancing, however,
artificially constrains the optimizer and may prevent the model from naturally allocating
capacity where it's most useful (gene encoder may legitimately need larger gradients
on BRCA given the survival signal in genomic features).

---

## Stage D2: Adaptive LR Component Ablation

Separating cosine annealing from Loss-EMA adaptive modulation:

| Method | Test C-index | Δ vs Cosine | p-value |
|--------|-------------|-------------|---------|
| AdamW + Flat LR | 0.6383 ± 0.0506 | -0.0137 | 0.431 |
| AdamW + Cosine | 0.6519 ± 0.0474 | — | — |
| AdamW + Flat + Loss-EMA | 0.6543 ± 0.0327 | +0.0024 | 0.918 |
| AdamW + Cosine + Loss-EMA | 0.6635 ± 0.0256 | +0.0116 | 0.560 |

### Cross-Comparison

| Effect | Δ C-index | Direction |
|--------|-----------|-----------|
| Cosine annealing (without EMA) | +0.0137 | Helpful |
| Cosine annealing (with EMA) | +0.0092 | Helpful |
| Loss-EMA modulation (on flat LR) | +0.0160 | Helpful |
| Loss-EMA modulation (on cosine LR) | +0.0116 | Helpful |

### Key Finding (D2)

**Cosine annealing and Loss-EMA are complementary.** Both provide modest improvements
independently, and their combination yields the best mean performance (0.6635). Notably,
Loss-EMA consistently **reduces variance** across folds (std: 0.047→0.026 with cosine,
0.051→0.033 with flat), suggesting it stabilizes training. Neither effect is individually
significant at p<0.05 due to high fold-to-fold variance.

---

## Summary of Findings

### Ranking by Test C-index (5-fold mean)
1. **Aux heads only** — 0.6723 ± 0.0528 (p=0.007 vs AdamW)
2. SAM — 0.6702 ± 0.0385 (p=0.124)
3. Adaptive LR (Cosine + Loss-EMA) — 0.6635 ± 0.0256 (p=0.560)
4. Adaptive LR (Flat + Loss-EMA) — 0.6543 ± 0.0327
5. Grad Balance only — 0.6526 ± 0.0430
6. AdamW + Cosine (baseline) — 0.6519 ± 0.0474
7. AdamW + Flat — 0.6383 ± 0.0506
8. M-SAM — 0.6366 ± 0.0392
9. Disentangle + Balance — 0.6336 ± 0.0400

### Conclusions

1. **Auxiliary unimodal supervision is the most effective technique** (the only statistically
   significant improvement), but only when gradient balancing is NOT applied.

2. **SAM provides the best generalization** (smallest val-test gap at 0.033) and the highest
   test C-index among the primary benchmark methods, suggesting flat minima help generalization.

3. **Loss-EMA adaptive LR reduces training variance** and provides modest gains on top of
   cosine scheduling. It's a low-cost, low-risk addition to any training pipeline.

4. **Gradient norm balancing is harmful** on BRCA — it over-constrains the optimization.

5. **M-SAM (Shapley + modality-aware SAM + OGM-GE) underperforms vanilla SAM.** The
   additional components add complexity and computation without improving test performance.
   On BRCA, where both modalities contribute similarly, modality-aware perturbation may
   not be needed.

### Recommended Configuration for Future Experiments

Based on these results, the recommended training configuration is:

```
Base optimizer: AdamW
LR schedule: Cosine annealing (epoch-level)
Enhancement: Auxiliary unimodal Cox heads (NO gradient balancing)
Optional: Loss-EMA adaptive LR modulation (low cost, reduces variance)
Optional: SAM perturbation (if compute budget allows)
```

---

## Comparison with Original Pilot (Buggy LR Scheduling)

| Method | Original (30ep, buggy) | Corrected (30ep) | Benchmark (200ep) |
|--------|----------------------|-----------------|-------------------|
| AdamW | 0.6669 | 0.6172 | 0.6519 (mean) |
| Adaptive LR | 0.6918 | 0.5263 | 0.6635 (mean) |
| Disentangle | 0.6944 | 0.5935 | 0.6336 (mean) |

The original pilot **inflated adaptive method results** because they ran with flat LR=0.001
while AdamW had cosine decay (audit finding C1). After correction, the relative ordering
and magnitudes change substantially.

---

## Limitations

1. **Single cancer type** (BRCA) — results may not generalize to cancers with different
   modality balance (e.g., LUAD where image features are weaker)
2. **Small test sets** (~211 patients per fold) create high per-fold variance
3. **Low event rate** (13.9%) limits C-index statistical power
4. **No hyperparameter tuning** — method-specific parameters (ema_decay, rho, aux_weight)
   use default values and may be suboptimal
5. **M-SAM with AdamW base** — the original M-SAM was designed for SGD; the Shapley
   computation uses accuracy-based contribution estimates which may not align with
   Cox regression objectives

---

## Appendix: File Inventory

### Results
- `benchmark/` — Primary 5-method × 5-fold results (JSON + checkpoints)
- `ablation_d1/` — Disentanglement component ablation
- `ablation_d2/` — Adaptive LR component ablation
- `pilot/` — 30-epoch pilot reproduction
- `figures/` — Training curves, box plots, diagnostic plots

### Code
- `run_phase1.py` — Phase 1 experiment runner (fixes audit issue C1)
- `test_phase1.py` — 18 unit tests (Cox loss, cosine LR, controllers, etc.)
- `analyze_phase1.py` — Analysis and visualization script
- `AUDIT_REPORT.md` — Pre-experiment code audit
- `EXPERIMENT_PLAN.md` — Experiment design document
- `RUN_MATRIX.csv` — Full run matrix
