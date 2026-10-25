# SGC-AdamW KIRC/STAD Experiment Report

**Date:** 2026-10-25
**Branch:** `experiment/sgc-adamw-kirc-stad`
**Runs:** 42 (7 optimizers × 2 cancers × 3 folds)
**Tests passed:** 35/35 (34 CPU + 1 CUDA)

---

## 1. Objective

Evaluate whether adaptive second-moment decay (variable β₂) improves
survival prediction on KIRC and STAD compared to fixed-β₂ AdamW.
Three SGC-AdamW ablation modes are tested:

1. **Drift-only**: β₂ adapts based on squared-gradient drift (matches DynamicsAwareAdamW)
2. **Drift+Alignment**: adds gradient-momentum cosine alignment signal
3. **Drift+Alignment+Curvature**: adds same-minibatch curvature gating

---

## 2. Methods

| ID | Type | β₂ | Adaptive | Notes |
|---|---|---|---|---|
| adamw_b999 | torch.optim.AdamW | 0.999 | No | Standard baseline |
| adamw_b98 | torch.optim.AdamW | 0.98 | No | Lower fixed β₂ |
| adamw_b95 | torch.optim.AdamW | 0.95 | No | Lowest fixed β₂ |
| dynamics_aware | DynamicsAwareAdamW | 0.95–0.999 | Drift | Prior method |
| sgc_drift | SGCAdamW | 0.95–0.999 | Drift | Reimplementation |
| sgc_drift_alignment | SGCAdamW | 0.95–0.999 | Drift+Align | New |
| sgc_drift_align_curv | SGCAdamW | 0.95–0.999 | Drift+Align+Curv | New |

**Shared config:** lr=0.001, wd=0.0001, cosine annealing, 50 epochs,
3-fold stratified CV (seed=42), batch_size=32, attention fusion, dropout=0.3.

---

## 3. Results

### KIRC (510 patients)

| Optimizer | Test C-index | Fold 0 | Fold 1 | Fold 2 | Val C | Gap | Best Ep | Time |
|---|---|---|---|---|---|---|---|---|
| AdamW β₂=0.999 | 0.7123 ± 0.0248 | 0.7466 | 0.7012 | 0.6891 | 0.7747 | 0.062 | 14 | 196s |
| AdamW β₂=0.98 | 0.7004 ± 0.0363 | 0.7474 | 0.6949 | 0.6590 | 0.7700 | 0.070 | 4 | 238s |
| AdamW β₂=0.95 | 0.7022 ± 0.0119 | 0.7097 | 0.7115 | 0.6853 | 0.7690 | 0.067 | 23 | 231s |
| DynamicsAware | **0.7232 ± 0.0209** | 0.7488 | 0.7230 | 0.6977 | 0.7826 | 0.059 | 11 | 233s |
| SGC drift-only | **0.7232 ± 0.0209** | 0.7488 | 0.7230 | 0.6977 | 0.7826 | 0.059 | 11 | 234s |
| SGC drift+align | 0.7161 ± 0.0245 | 0.7496 | 0.7071 | 0.6917 | 0.7780 | 0.062 | 7 | 236s |
| SGC drift+align+curv | 0.7128 ± 0.0332 | 0.7496 | 0.7197 | 0.6691 | 0.7812 | 0.068 | 17 | 245s |

### STAD (379 patients)

| Optimizer | Test C-index | Fold 0 | Fold 1 | Fold 2 | Val C | Gap | Best Ep | Time |
|---|---|---|---|---|---|---|---|---|
| AdamW β₂=0.999 | **0.5711 ± 0.0099** | 0.5628 | 0.5654 | 0.5850 | 0.6142 | 0.043 | 32 | 41s |
| AdamW β₂=0.98 | 0.5540 ± 0.0350 | 0.5875 | 0.5057 | 0.5688 | 0.6220 | 0.068 | 8 | 40s |
| AdamW β₂=0.95 | 0.5628 ± 0.0174 | 0.5872 | 0.5482 | 0.5529 | 0.6091 | 0.046 | 17 | 41s |
| DynamicsAware | 0.5281 ± 0.0256 | 0.5488 | 0.4919 | 0.5435 | 0.6093 | 0.081 | 14 | 44s |
| SGC drift-only | 0.5281 ± 0.0256 | 0.5488 | 0.4919 | 0.5435 | 0.6093 | 0.081 | 14 | 41s |
| SGC drift+align | 0.5307 ± 0.0268 | 0.5649 | 0.4995 | 0.5276 | 0.6121 | 0.081 | 8 | 43s |
| SGC drift+align+curv | 0.5358 ± 0.0225 | 0.5631 | 0.5081 | 0.5363 | 0.6145 | 0.079 | 8 | 45s |

---

## 4. Paired Statistical Tests (vs AdamW β₂=0.999)

| Cancer | Optimizer | Δ C-index | t-stat | p-value | Direction |
|---|---|---|---|---|---|
| KIRC | AdamW β₂=0.98 | -0.0119 | -1.27 | 0.332 | - |
| KIRC | AdamW β₂=0.95 | -0.0102 | -0.73 | 0.543 | - |
| KIRC | DynamicsAware | +0.0108 | 1.88 | 0.201 | + |
| KIRC | SGC drift-only | +0.0108 | 1.88 | 0.201 | + |
| KIRC | SGC drift+align | +0.0038 | 3.74 | 0.065 | ~ |
| KIRC | SGC drift+align+curv | +0.0005 | 0.04 | 0.969 | ~ |
| STAD | AdamW β₂=0.98 | -0.0171 | -0.70 | 0.556 | - |
| STAD | AdamW β₂=0.95 | -0.0083 | -0.49 | 0.672 | - |
| STAD | DynamicsAware | -0.0430 | -2.50 | 0.129 | - |
| STAD | SGC drift-only | -0.0430 | -2.50 | 0.129 | - |
| STAD | SGC drift+align | -0.0404 | -1.89 | 0.200 | - |
| STAD | SGC drift+align+curv | -0.0352 | -1.96 | 0.189 | - |

---

## 5. SGC Drift-Only vs DynamicsAwareAdamW Validation

SGC drift-only mode should numerically match DynamicsAwareAdamW.

- **KIRC**: EXACT MATCH
  - Fold scores: ['0.7488', '0.7230', '0.6977']
- **STAD**: EXACT MATCH
  - Fold scores: ['0.5488', '0.4919', '0.5435']

**Validation status:** PASS

---

## 6. Ablation Analysis

### Does drift help? (sgc_drift vs adamw_b999)

- **KIRC**: Δ = +0.0108 (helps)
- **STAD**: Δ = -0.0430 (hurts)

### Does alignment help? (sgc_drift_alignment vs sgc_drift)

- **KIRC**: Δ = -0.0070 (hurts)
- **STAD**: Δ = +0.0026 (neutral)

### Does curvature help? (sgc_drift_align_curv vs sgc_drift_alignment)

- **KIRC**: Δ = -0.0033 (neutral)
- **STAD**: Δ = +0.0052 (helps)

### Fixed β₂ comparison (adamw_b95 vs adamw_b98 vs adamw_b999)

- **KIRC**: β₂=0.999=0.7123, β₂=0.98=0.7004, β₂=0.95=0.7022 → best: β₂=0.999
- **STAD**: β₂=0.999=0.5711, β₂=0.98=0.5540, β₂=0.95=0.5628 → best: β₂=0.999

---

## 7. Generalization Gap Analysis

| Cancer | Optimizer | Val C | Test C | Gap |
|---|---|---|---|---|
| KIRC | AdamW β₂=0.999 | 0.7747 | 0.7123 | 0.062 |
| KIRC | AdamW β₂=0.98 | 0.7700 | 0.7004 | 0.070 |
| KIRC | AdamW β₂=0.95 | 0.7690 | 0.7022 | 0.067 |
| KIRC | DynamicsAware | 0.7826 | 0.7232 | 0.059 |
| KIRC | SGC drift-only | 0.7826 | 0.7232 | 0.059 |
| KIRC | SGC drift+align | 0.7780 | 0.7161 | 0.062 |
| KIRC | SGC drift+align+curv | 0.7812 | 0.7128 | 0.068 |
| STAD | AdamW β₂=0.999 | 0.6142 | 0.5711 | 0.043 |
| STAD | AdamW β₂=0.98 | 0.6220 | 0.5540 | 0.068 |
| STAD | AdamW β₂=0.95 | 0.6091 | 0.5628 | 0.046 |
| STAD | DynamicsAware | 0.6093 | 0.5281 | 0.081 |
| STAD | SGC drift-only | 0.6093 | 0.5281 | 0.081 |
| STAD | SGC drift+align | 0.6121 | 0.5307 | 0.081 |
| STAD | SGC drift+align+curv | 0.6145 | 0.5358 | 0.079 |

---

## 8. Key Findings

1. **SGC drift-only exactly matches DynamicsAwareAdamW** — validates the
   reimplementation (identical fold scores on both cancers)

2. **Adaptive β₂ is cancer-specific:**
   - KIRC: drift adaptation helps (+0.011 over baseline)
   - STAD: drift adaptation hurts (−0.043 below baseline)

3. **Alignment signal provides no benefit** on either cancer
   - KIRC: −0.007 vs drift-only
   - STAD: +0.003 (noise-level)

4. **Curvature gating provides no benefit** on either cancer
   - KIRC: −0.003 vs drift+alignment
   - STAD: +0.005 (noise-level)

5. **Fixed β₂=0.999 is the most robust** across both cancers
   - KIRC: 0.7123 (second-best behind adaptive methods)
   - STAD: 0.5711 (best overall)

6. **Lower fixed β₂ doesn't help** — β₂=0.98 and β₂=0.95 are worse
   or comparable to β₂=0.999 on both cancers

7. **Adaptive methods over-correct on STAD** — the noisier dataset
   (379 patients, lower absolute C-index) is hurt by faster β₂ adaptation

---

## 9. Limitations and Caveats

- Only 3 folds — insufficient for p<0.05 significance at observed effect sizes
- Only 2 cancers — cannot generalize to other datasets
- Hyperparameters (drift_sensitivity=3.0, alignment_sensitivity=1.0) not tuned
- Curvature interval=10 may be too sparse or too frequent
- No test-based hyperparameter tuning performed (all based on validation selection)

---

## 10. Recommendation

**Do NOT proceed to wider experiments.** The SGC-AdamW additions
(alignment, curvature) provide no measurable benefit over the simpler
drift-only variant, which itself is cancer-specific (helps KIRC, hurts STAD).

**For the paper:** Fixed AdamW (β₂=0.999) remains the recommended baseline
optimizer. The adaptive β₂ idea is sound but requires per-dataset tuning
that undermines its value as a general-purpose optimizer improvement.

---

## 11. Environment

| Item | Value |
|---|---|
| GPU | Tesla T4 (15.6 GB) |
| PyTorch | 2.10.0+cu130 |
| Python | 3.13.12 |
| EC2 | i-03aed551c7468abb6 |
| Unit tests | 35/35 passed |
| Total training time | ~42 runs, data loading dominated |
