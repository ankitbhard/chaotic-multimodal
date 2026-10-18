# CVPR 2027 Pilot Plan — Perturbation-Calibrated Reliability-Guided Fusion (PCRF)

**Date:** 2026-10-18
**Branch:** `experiments/cvpr-reliability-pilot`
**Deadline:** 2026-10-27 (Go/No-Go)

## Research Question

Does **perturbation-calibrated modality reliability** improve histopathology +
gene-expression survival ranking over ordinary attention and training-only
auxiliary Cox heads, particularly under realistic feature corruption?
A learned attention gate alone is NOT sufficient novelty.

## Scope / Guardrails

- Isolated branch: `experiments/cvpr-reliability-pilot`; do not modify legacy experiment files or results.
- First three cohorts: BRCA, KIRC, LUAD. Full follow-up: LUSC, STAD.
- Use existing UNI/UNI2-h feature files and RNA-seq inputs; record embedding backbone per cohort.
- No test-set-based tuning, no use of test labels for reliability estimates, no preprocessing fitted outside train fold.
- Do not assume 20,000 genes are available: log gene names and actual feature counts per fold.
- Do not claim executed runs unless JSON and manifest files exist.

## Pilot Cohorts

| Cancer | Patients (approx) | Image Backbone | Image Dim |
|--------|--------------------|----------------|-----------|
| BRCA   | ~1050              | UNI            | 1024      |
| KIRC   | ~519               | UNI2-h         | 1536      |
| LUAD   | ~530               | UNI2-h         | 1536      |

## Experimental Design

- **Folds:** 3-fold stratified CV (by vital_status)
- **Epochs:** 100
- **Optimizer:** AdamW (lr=1e-3, wd=1e-4) for ALL methods
- **LR Schedule:** Epoch-level cosine annealing
- **Checkpoint:** Best validation C-index
- **Seeds:** 42 (data split), 42+fold (model init)

## Methods (6 total)

1. **adamw** — Standard attention fusion + AdamW baseline
2. **aux_cox** — Attention fusion + auxiliary unimodal Cox heads (0.1 weight)
3. **mod_dropout** — Attention fusion + modality dropout (p=0.15)
4. **learned_gating** — PCRF architecture, no reliability loss (capacity control)
5. **pcrf** — PCRF with reliability supervision, no aux Cox in main loss
6. **pcrf_aux** — PCRF with reliability supervision + aux Cox

## Key Comparisons

| Comparison | Tests |
|------------|-------|
| 1 vs 2 | Does auxiliary supervision help? |
| 1 vs 3 | Does modality dropout improve robustness? |
| 3 vs 4 | Learned gating vs dropout for robustness |
| 4 vs 5 | Does reliability supervision help (controlling for capacity)? |
| 2 vs 6 | Does PCRF+reliability help on top of aux? |
| 5 vs 6 | Does aux help on top of PCRF? |

## Robustness Evaluation

- Clean (no perturbation)
- Gaussian noise on image embeddings (sigma = 0.1, 0.3, 0.5, 1.0)
- Gaussian noise on gene features (sigma = 0.1, 0.3, 0.5, 1.0)
- Missing image modality (zero embedding)
- Missing gene modality (zero features)

## Success Criteria

Proceed to full-scale (5 folds x 5 cancers) if:
1. PCRF improves over learned gating on >=2/3 pilot cancers, OR
2. PCRF shows reproducible robustness advantage under perturbation/missing modality
3. No data leakage or evaluation flaws detected

## Known Implementation Risks

- Prior BRCA audit observed only ~1,000 genes although config requested 20,000.
- Auxiliary supervision changes training objective; do not label it a pure optimizer.
- Image/gene gradient equalization harmed BRCA; do not reintroduce it by default.
- Verify test folds are untouched by selection.
- Existing code uses a two-logit head but Cox loss only reads logit index 1.
- Zero-event batches produce zero loss and zero gradients.

## File Structure

```
experiments/cvpr_reliability_pilot/
├── models.py           # PCRF model architecture
├── reliability.py      # Perturbation calibration module
├── losses.py           # Cox loss, reliability loss
├── train.py            # Training script (all 6 methods)
├── evaluate.py         # Evaluation + robustness testing
├── test_pilot.py       # Unit tests (23 tests)
├── run_baselines.sh    # Phase 2 baseline runs
├── run_pilot.sh        # Full pilot runs
├── analyze_results.py  # Analysis and visualization
├── METHODOLOGY.md      # Mathematical formulation
└── PILOT_REPORT.md     # Final report
```

## Status

Implementation complete. 23/23 unit tests passing. Ready for EC2 deployment and execution.
