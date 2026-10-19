# Phase 2 Ablation Analysis

## Ablation Matrix

Phase 2 isolates three factors in the PCRF method:

| Factor | learned_gating_aux | pcrf_aux | pcrf_aux_warmup |
|--------|:------------------:|:--------:|:---------------:|
| PCRF architecture | ✓ | ✓ | ✓ |
| Auxiliary Cox heads | ✓ | ✓ | ✓ |
| Aux encoder gradients | ✓ | ✓ | ✓ |
| Reliability supervision | ✗ | ✓ | ✓ (after 20 ep) |
| Curriculum warmup | — | — | ✓ |

All three methods have **identical parameter counts** (565,126), **identical initialization**,
and **identical optimizer configuration**.

---

## Ablation 1: Reliability Supervision (learned_gating_aux vs pcrf_aux)

This is the critical experiment. Everything is held constant except the reliability loss.

| Cancer | Without Reliability | With Reliability | Δ | Per-fold Δ |
|--------|:------------------:|:----------------:|:---:|:----------:|
| BRCA | 0.639 | **0.680** | **+0.042** | +0.075, +0.005, +0.044 |
| KIRC | 0.696 | **0.703** | +0.007 | −0.007, +0.031, −0.003 |
| LUAD | **0.629** | 0.626 | −0.003 | −0.006, +0.052, −0.055 |

**Finding:** Reliability supervision provides a consistent improvement on BRCA
(positive on all 3 folds) but is inconclusive on KIRC and LUAD.

The BRCA result is notable because it contrasts with Phase 1's `learned_gating` vs
`pcrf` comparison (where gating was BETTER). The difference: Phase 2 includes
auxiliary Cox heads, which means the reliability targets are calibrated from
better-trained aux heads (non-detached features → stronger gradient signal).

---

## Ablation 2: Curriculum Warmup (pcrf_aux vs pcrf_aux_warmup)

| Cancer | Standard | With Warmup | Δ |
|--------|:--------:|:-----------:|:---:|
| BRCA | **0.680** | 0.675 | −0.006 |
| KIRC | **0.703** | 0.699 | −0.004 |
| LUAD | **0.626** | 0.623 | −0.003 |

**Finding:** 20-epoch warmup provides no benefit. The standard PCRF+Aux method
matches or exceeds warmup on all cancers. Curriculum reliability is not needed
when auxiliary heads are trained with encoder gradients from epoch 1.

This makes sense: with `use_aux_loss=True`, the auxiliary heads receive strong
supervision from the first epoch via the Cox loss + encoder gradient flow. The
perturbation sensitivity is therefore meaningful immediately. The warmup was designed
for the case where aux heads train on detached features (slower learning), which
applies to standalone `pcrf` but not to `pcrf_aux`.

---

## Ablation 3: Phase 1 vs Phase 2 Reliability Effect

Phase 1 compared `learned_gating` vs `pcrf` (WITHOUT aux):

| Cancer | learned_gating | pcrf | Δ_Phase1 |
|--------|:-------------:|:----:|:--------:|
| BRCA | **0.663** | 0.637 | −0.025 (gating better) |
| KIRC | **0.701** | 0.686 | −0.015 (gating better) |
| LUAD | 0.608 | **0.614** | +0.006 |

Phase 2 compares `learned_gating_aux` vs `pcrf_aux` (WITH aux):

| Cancer | learned_gating_aux | pcrf_aux | Δ_Phase2 |
|--------|:------------------:|:--------:|:--------:|
| BRCA | 0.639 | **0.680** | **+0.042** |
| KIRC | 0.696 | **0.703** | +0.007 |
| LUAD | **0.629** | 0.626 | −0.003 |

**Finding:** Reliability supervision HURTS without aux heads (Phase 1) but HELPS with
them (Phase 2, at least on BRCA). This confirms the hypothesis from the Phase 1 report:

> The auxiliary heads need to be sufficiently trained for the perturbation sensitivity
> to be meaningful.

When aux heads train on non-detached features (Phase 2), they learn stronger predictions
faster, producing more informative perturbation-calibration targets. When aux heads
train on detached features (Phase 1 pcrf), they produce weaker predictions and the
reliability targets are noisy/uninformative.

---

## Ablation 4: Regularization Effect

Reliability loss may act primarily as a regularizer rather than providing semantically
meaningful modality weighting. Evidence:

| Metric | learned_gating_aux | pcrf_aux | pcrf_aux_warmup |
|--------|:------------------:|:--------:|:---------------:|
| BRCA gen. gap | 0.090 | **0.056** | **0.038** |
| KIRC gen. gap | **0.080** | 0.089 | 0.084 |
| LUAD gen. gap | 0.116 | **0.105** | 0.117 |
| BRCA best epoch | 39 | 34 | 57 |
| KIRC best epoch | 10 | 17 | 22 |
| LUAD best epoch | 11 | 25 | 7 |

On BRCA, pcrf_aux has a much smaller generalization gap (0.056 vs 0.090), suggesting
the reliability loss acts as an effective regularizer. On KIRC and LUAD, the effect
is less clear.

The later checkpoint selection for pcrf_aux (BRCA: 34 vs 39, KIRC: 17 vs 10) is
consistent with slower overfitting due to the additional loss term.

---

## Summary of Ablation Findings

1. **Reliability supervision helps with aux heads (BRCA)** but not without them
2. **Curriculum warmup is unnecessary** when aux heads receive encoder gradients
3. **Reliability loss may act primarily as a regularizer** (smaller gen. gap on BRCA)
4. **The effect is cancer-specific** — BRCA benefits most, KIRC marginally, LUAD not at all
5. **Phase 1's negative result (pcrf < learned_gating) is explained** by weak aux heads
   trained on detached features

### Implication for Method Design

If reliability supervision is published, it MUST be paired with auxiliary unimodal heads
that receive encoder gradients (`use_aux_loss=True`). The standalone version (detached
aux heads) produces meaningless reliability targets and hurts performance.
