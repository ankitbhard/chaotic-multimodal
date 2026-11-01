# RA-SAMO v3: Censoring-Aware Objective Selection — Final Report

**Benchmark**: 4 methods x 5 cancers x 5 folds x 50 epochs = 100 runs
**Branch**: `experiment/censor-aware-ra-samo-v3`
**Date**: 2026-10-31
**Hardware**: Tesla T4 (15 GB), EC2 g4dn.xlarge

---

## 1. Methods

| Variant | Signal sources | Unique mechanism |
|---------|---------------|-----------------|
| v3.0 | Training loss progress only | Discrete 3-mode selection (cox/aux/rank) with exact-zero lambdas |
| v3.1 | + event rate, pair density | Censoring-aware features shift mode scores |
| v3.2 | + gradient alignment & reliability | Cosine similarity between primary and aux/rank gradients |
| v3.3 | + inner-validation rollouts | 2-epoch candidate rollouts on 20% inner-val split every 10 epochs |

All variants start in **cox mode** (all secondary weights = 0) and switch only after warmup (2 epochs) with hysteresis (min_dwell=4, switch_margin=0.035).

---

## 2. C-index Results (test, 5-fold mean +/- std)

### v3 Methods

| Method | BRCA | KIRC | LUAD | LUSC | STAD | Mean |
|--------|------|------|------|------|------|------|
| v3.0 | 0.645+/-0.066 | 0.719+/-0.062 | 0.635+/-0.041 | 0.545+/-0.071 | 0.586+/-0.038 | 0.626 |
| v3.1 | 0.612+/-0.056 | 0.719+/-0.062 | 0.635+/-0.041 | 0.544+/-0.066 | 0.590+/-0.033 | 0.620 |
| v3.2 | 0.640+/-0.067 | 0.729+/-0.061 | 0.629+/-0.058 | 0.562+/-0.052 | 0.583+/-0.023 | 0.629 |
| v3.3 | 0.640+/-0.067 | 0.729+/-0.061 | 0.629+/-0.058 | 0.562+/-0.052 | 0.583+/-0.023 | 0.629 |

### Comparison with Previous Benchmark (samo_all_cancers_50ep)

| Method | BRCA | KIRC | LUAD | LUSC | STAD | Mean |
|--------|------|------|------|------|------|------|
| **adaptive_aux** | **0.678** | 0.731 | 0.626 | 0.539 | **0.595** | **0.634** |
| **rank** | 0.621 | 0.731 | **0.646** | **0.568** | 0.591 | **0.631** |
| **baseline (cox)** | 0.629 | **0.740** | 0.641 | 0.559 | 0.583 | **0.630** |
| **ra_samo_v3_2** | 0.640 | 0.729 | 0.629 | 0.562 | 0.583 | 0.629 |
| aux_rank | 0.650 | 0.731 | 0.637 | 0.551 | 0.568 | 0.628 |
| **ra_samo_v3_0** | 0.645 | 0.719 | 0.635 | 0.545 | 0.586 | 0.626 |
| aux | 0.662 | 0.728 | 0.618 | 0.540 | 0.581 | 0.626 |
| ra_samo_v3_1 | 0.612 | 0.719 | 0.635 | 0.544 | 0.590 | 0.620 |
| samo | 0.632 | 0.731 | 0.631 | 0.536 | 0.569 | 0.620 |

**v3.2 ranks 4th overall** (0.629), behind adaptive_aux (0.634), static rank (0.631), and baseline (0.630). All differences are within noise (< 0.005).

---

## 3. Mode Selection: Rank Everywhere

The controller selects **rank mode in 95-96% of all epochs** across every cancer, every fold, and every variant. No run ever enters aux mode.

| Cancer | cox epochs | aux epochs | rank epochs | Final mode (all folds) |
|--------|-----------|------------|-------------|----------------------|
| BRCA | 11-14 | 0 | 236-239 | rank x5 |
| KIRC | 10-11 | 0 | 239-240 | rank x5 |
| LUAD | 10 | 0 | 240 | rank x5 |
| LUSC | 11-14 | 0 | 236-239 | rank x5 |
| STAD | 12-13 | 0 | 237-238 | rank x5 |

The cox-only epochs correspond to warmup + the first post-warmup epoch before the first switch.

---

## 4. Why Rank Always Wins: Root Cause Analysis

### 4.1 Loss Progress Signal

The controller computes `aux_progress = mean(img_loss_progress, gene_loss_progress) - fusion_loss_progress` and `rank_progress = rank_loss_progress - fusion_loss_progress`. By epoch 25, these scores are:

| Cancer | aux_score | rank_score |
|--------|-----------|------------|
| BRCA | -0.491 | +0.131 |
| KIRC | -0.377 | +0.223 |
| LUAD | similar | similar |
| LUSC | similar | similar |
| STAD | -0.403 | +0.241 |

**The aux branch Cox losses improve much slower than the fusion loss** (they're per-modality and harder), driving aux_score deeply negative. The ranking loss improves faster (it's a simpler pairwise objective), driving rank_score positive. This pattern is **universal** across all 5 cancers.

### 4.2 Gradient Alignment (v3.2)

Gradient cosine similarity between the primary Cox loss and each secondary loss:

| Cancer | cos(cox, aux) | cos(cox, rank) |
|--------|--------------|----------------|
| BRCA | 0.180 +/- 0.058 | **0.981 +/- 0.018** |
| KIRC | 0.168 +/- 0.074 | **0.947 +/- 0.025** |
| LUAD | 0.109 +/- 0.062 | **0.947 +/- 0.027** |
| LUSC | 0.106 +/- 0.053 | **0.933 +/- 0.031** |
| STAD | 0.121 +/- 0.055 | **0.938 +/- 0.029** |

The ranking loss gradient is **nearly perfectly aligned** (cos > 0.93) with the primary Cox gradient across all cancers. This means the ranking loss is essentially a regularized version of the same objective. The aux branch gradients are weakly aligned (cos ~0.11-0.18), confirming they pull the model in different directions.

The v3.2 alignment signal (0.15 * cos) reinforces the loss-progress signal rather than counteracting it, so mode selection doesn't change.

### 4.3 Validation Rollouts (v3.3)

v3.3 produces **identical results to v3.2** across all 100 runs. The 2-epoch rollouts (every 10 epochs) don't generate a strong enough differential signal to overcome the dominant loss-progress + alignment scores. The rollout horizon is too short to reveal mode differences.

### 4.4 Censoring Features (v3.1)

The censoring adjustments are small (+/-0.15 * deviation from center):
- Event rate adjustment: 0.15 * (0.3 - event_rate) — shifts aux score by -0.01 to +0.02
- Pair density adjustment: 0.15 * (density - 0.15) — shifts rank score by -0.02 to +0.01

These adjustments are **dwarfed** by the loss-progress signal (aux ~ -0.5, rank ~ +0.15).

---

## 5. Censoring Statistics

| Cancer | Event Rate | Pair Density | Interpretation |
|--------|-----------|-------------|----------------|
| BRCA | 0.136 | 0.049 | Heavily censored, very few comparable pairs |
| KIRC | 0.346 | 0.205 | Moderate censoring |
| LUAD | 0.356 | 0.198 | Moderate censoring |
| LUSC | 0.426 | 0.225 | Least censored |
| STAD | 0.394 | 0.230 | Moderate censoring |

BRCA has 86.4% censoring — only 13.6% of patients have observed events. This makes survival prediction hardest and explains BRCA's high variance across folds.

---

## 6. Computational Cost

| Method | Mean wall time per run | Overhead vs v3.0 |
|--------|----------------------|-------------------|
| v3.0 | 76s (1.3 min) | — |
| v3.1 | 76s (1.3 min) | +0% |
| v3.2 | 77s (1.3 min) | +1% |
| v3.3 | 79s (1.3 min) | +4% |

Overhead is negligible. Gradient alignment (v3.2) runs every 5 epochs; rollouts (v3.3) every 10 epochs. Total benchmark: ~2.2 hours for 100 runs.

---

## 7. Conclusions

1. **The discrete 3-mode controller works correctly** — it switches from cox to rank after warmup based on loss-progress signals, and the hysteresis/dwell mechanisms prevent oscillation.

2. **Rank mode dominates universally** because:
   - Ranking loss improves faster than fusion Cox loss (positive rank_score)
   - Aux branch losses improve slower than fusion loss (negative aux_score)
   - Gradient alignment confirms: rank gradients are ~95% aligned with Cox, aux only ~14%

3. **The controller never discovers the per-cancer optimal objectives** found in the static benchmark:
   - BRCA/STAD should benefit from aux (adaptive_aux wins there)
   - KIRC should stay in cox (baseline wins there)
   - The loss-progress signal doesn't capture C-index performance differences

4. **v3.1/v3.2/v3.3 enhancements don't change mode selection** — the loss-progress signal is 10-50x stronger than censoring features, gradient alignment bonuses, or validation rollout differentials.

5. **v3 overall performance** (0.626-0.629 mean C-index) is close to static rank (0.631) and slightly below baseline (0.630) and adaptive_aux (0.634). The controller effectively reduces to a static rank selector.

---

## 8. Implications for Future Work

The fundamental issue is that **loss-progress is not a reliable proxy for C-index performance**. A loss that decreases faster doesn't necessarily produce better concordance. To build an effective adaptive controller, the signal must come from:

1. **Direct C-index evaluation** on a held-out validation set (not just loss values)
2. **Per-cancer calibration** — the optimal objective depends on the censoring structure and modality balance, not just which loss drops fastest
3. **Longer rollout horizons** — 2-epoch rollouts are too short; need 5-10 epoch windows to see C-index differences emerge

The static methods (adaptive_aux, rank, baseline) remain the strongest approaches. The best strategy is cancer-specific objective selection guided by validation C-index, not automatic loss-based switching.
