# CVPR Reliability Pilot — Phase 2 Results
**Generated:** 2026-10-18T21:06:46.096856
**Methods:** learned_gating_aux, pcrf_aux, pcrf_aux_warmup
**Cancers:** BRCA, KIRC, LUAD
**Folds:** 3


## BRCA
### Test C-index (mean ± std)
| Method | Test C-index | Val C-index | Best Epoch | Gap | Time | Warmup |
|--------|-------------|-------------|------------|-----|------|--------|
| learned_gating_aux | 0.6389 ± 0.0358 | 0.7286 | 39 | 0.090 | 240s | 0 |
| pcrf_aux | 0.6804 ± 0.0154 | 0.7367 | 34 | 0.056 | 20s | 0 |
| pcrf_aux_warmup | 0.6745 ± 0.0353 | 0.7129 | 57 | 0.038 | 20s | 20 |

### Per-Fold Test C-index
| Method | Fold 0 | Fold 1 | Fold 2 |
|--------|-------|-------|-------|
| learned_gating_aux | 0.5889 | 0.6709 | 0.6569 |
| pcrf_aux | 0.6640 | 0.6760 | 0.7011 |
| pcrf_aux_warmup | 0.6264 | 0.6870 | 0.7102 |

### Paired Comparisons
**PCRF+Aux vs Gating+Aux (reliability effect):**
- Δ C-index = +0.0415 (95% CI: [-0.0456, +0.1286])
- t = 2.049, p = 0.1769, Significant: No

**PCRF+Aux+Warmup vs Gating+Aux (curriculum reliability effect):**
- Δ C-index = +0.0357 (95% CI: [-0.0108, +0.0821])
- t = 3.302, p = 0.0807, Significant: No

**Warmup vs No-Warmup (curriculum effect):**
- Δ C-index = -0.0058 (95% CI: [-0.0743, +0.0626])
- t = -0.366, p = 0.7492, Significant: No


### Reliability Weights (test set)
| Method | Img Weight | Gene Weight |
|--------|-----------|------------|
| learned_gating_aux | 0.463 ± 0.096 | 0.537 ± 0.096 |
| pcrf_aux | 0.408 ± 0.021 | 0.592 ± 0.021 |
| pcrf_aux_warmup | 0.440 ± 0.013 | 0.560 ± 0.013 |

## KIRC
### Test C-index (mean ± std)
| Method | Test C-index | Val C-index | Best Epoch | Gap | Time | Warmup |
|--------|-------------|-------------|------------|-----|------|--------|
| learned_gating_aux | 0.6960 ± 0.0105 | 0.7761 | 10 | 0.080 | 156s | 0 |
| pcrf_aux | 0.7029 ± 0.0184 | 0.7916 | 17 | 0.089 | 10s | 0 |
| pcrf_aux_warmup | 0.6992 ± 0.0091 | 0.7832 | 22 | 0.084 | 10s | 20 |

### Per-Fold Test C-index
| Method | Fold 0 | Fold 1 | Fold 2 |
|--------|-------|-------|-------|
| learned_gating_aux | 0.7095 | 0.6944 | 0.6840 |
| pcrf_aux | 0.7025 | 0.7257 | 0.6807 |
| pcrf_aux_warmup | 0.6981 | 0.7108 | 0.6887 |

### Paired Comparisons
**PCRF+Aux vs Gating+Aux (reliability effect):**
- Δ C-index = +0.0070 (95% CI: [-0.0456, +0.0596])
- t = 0.571, p = 0.6254, Significant: No

**PCRF+Aux+Warmup vs Gating+Aux (curriculum reliability effect):**
- Δ C-index = +0.0032 (95% CI: [-0.0314, +0.0379])
- t = 0.402, p = 0.7265, Significant: No

**Warmup vs No-Warmup (curriculum effect):**
- Δ C-index = -0.0037 (95% CI: [-0.0323, +0.0248])
- t = -0.565, p = 0.6292, Significant: No


### Reliability Weights (test set)
| Method | Img Weight | Gene Weight |
|--------|-----------|------------|
| learned_gating_aux | 0.463 ± 0.088 | 0.537 ± 0.088 |
| pcrf_aux | 0.461 ± 0.072 | 0.539 ± 0.072 |
| pcrf_aux_warmup | 0.448 ± 0.088 | 0.552 ± 0.088 |

## LUAD
### Test C-index (mean ± std)
| Method | Test C-index | Val C-index | Best Epoch | Gap | Time | Warmup |
|--------|-------------|-------------|------------|-----|------|--------|
| learned_gating_aux | 0.6288 ± 0.0537 | 0.7448 | 11 | 0.116 | 135s | 0 |
| pcrf_aux | 0.6258 ± 0.0127 | 0.7311 | 25 | 0.105 | 8s | 0 |
| pcrf_aux_warmup | 0.6226 ± 0.0332 | 0.7392 | 7 | 0.117 | 8s | 20 |

### Per-Fold Test C-index
| Method | Fold 0 | Fold 1 | Fold 2 |
|--------|-------|-------|-------|
| learned_gating_aux | 0.6444 | 0.5566 | 0.6854 |
| pcrf_aux | 0.6382 | 0.6083 | 0.6308 |
| pcrf_aux_warmup | 0.6365 | 0.5768 | 0.6546 |

### Paired Comparisons
**PCRF+Aux vs Gating+Aux (reliability effect):**
- Δ C-index = -0.0030 (95% CI: [-0.1352, +0.1292])
- t = -0.098, p = 0.9311, Significant: No

**PCRF+Aux+Warmup vs Gating+Aux (curriculum reliability effect):**
- Δ C-index = -0.0062 (95% CI: [-0.0696, +0.0573])
- t = -0.417, p = 0.7170, Significant: No

**Warmup vs No-Warmup (curriculum effect):**
- Δ C-index = -0.0032 (95% CI: [-0.0719, +0.0656])
- t = -0.198, p = 0.8616, Significant: No


### Reliability Weights (test set)
| Method | Img Weight | Gene Weight |
|--------|-----------|------------|
| learned_gating_aux | 0.426 ± 0.036 | 0.574 ± 0.036 |
| pcrf_aux | 0.425 ± 0.032 | 0.575 ± 0.032 |
| pcrf_aux_warmup | 0.433 ± 0.013 | 0.567 ± 0.013 |


## Cross-Cancer Summary
| Cancer | Gating+Aux | PCRF+Aux | PCRF+Aux+Warmup | Best | Δ_reliability | p |
|--------|-----------|---------|----------------|------|---------------|---|
| BRCA | 0.6389 | 0.6804 | 0.6745 | pcrf_aux | +0.0415 | 0.1769 |
| KIRC | 0.6960 | 0.7029 | 0.6992 | pcrf_aux | +0.0070 | 0.6254 |
| LUAD | 0.6288 | 0.6258 | 0.6226 | learned_gating_aux | -0.0030 | 0.9311 |


## Decision
Reliability supervision improves over architecture-matched baseline on **2/3** cancers.

**RECOMMENDATION: Expand to 5-fold experiments on all 5 cancers.**
