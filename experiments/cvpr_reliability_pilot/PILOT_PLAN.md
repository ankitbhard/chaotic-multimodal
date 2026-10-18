# CVPR reliability-aware multimodal survival pilot (October 2026)

## Research question
Does **perturbation-calibrated modality reliability** improve histopathology + gene-expression survival ranking over ordinary attention and training-only auxiliary Cox heads, particularly under realistic feature corruption? A learned attention gate alone is NOT sufficient novelty.

## Scope / guardrails
- Isolated branch: `experiments/cvpr-reliability-pilot`; do not modify legacy experiment files or results.
- First three cohorts: BRCA, KIRC, LUAD. Full follow-up: LUSC, STAD.
- Use existing UNI/UNI2-h feature files and RNA-seq inputs; record embedding backbone per cohort.
- Use existing `experiments/phase1_optimizer_validation/run_phase1.py` unchanged for reproducible baseline and auxiliary-head runs.
- No test-set-based tuning, no use of test labels for reliability estimates, no preprocessing fitted outside train fold.
- Do not assume 20,000 genes are available: log gene names and actual feature counts per fold.
- Do not claim executed runs unless JSON and manifest files exist.

## Pilot 1: establish reference results
Run AdamW and `disentangle_aux_only` with matched folds, initialization, 100 epochs and cosine LR. If prior 200-epoch results are comparable, retain them as an additional reference, not a replacement. Start with fold 0 for smoke test, then folds 0/1/2. Use the provided `run_baselines.sh`.

## Pilot 2: reliability method (implementation pending)
Architecture:
1. Keep existing image encoder, gene encoder and joint survival head.
2. Keep training-only image and gene auxiliary Cox heads.
3. Create two small trainable reliability heads q_I(z_I), q_G(z_G). Map image/gene embeddings to a common dimension before fusion.
4. Define *training-only* reliability targets from small, biologically plausible, train-fold-calibrated feature perturbations. For each modality m, compute d_m = E[(r_m(z_m+delta)-r_m(z_m))^2] using a frozen or stop-gradient unimodal teacher; do not backprop through the targets.
5. Calibrate targets using train-only statistics; assess whether stability correlates with predictive utility. Stability is NOT equivalent to correctness.
6. Gate fusion with softmax(q_I,q_G), but compare with an equal-capacity ordinary learned gate.
7. Proposed loss: L_joint_Cox + 0.1(L_image_Cox+L_gene_Cox) + gamma * L_reliability. Choose gamma on validation only.
8. Evaluate image-only, gene-only, both, image-corrupted, gene-corrupted, and missing-modality (if trained for missingness). A missing input must not silently be replaced by a zero vector without explicit masking.
9. Avoid batchwise Cox objective changes between methods. Document zero-event batch handling and ties.

## Required ablations
A. Existing attention + AdamW
B. Existing attention + auxiliary Cox heads
C. Equal-capacity learned reliability/gating without perturbation targets
D. Perturbation-guided reliability, without auxiliary Cox heads
E. Full perturbation-guided reliability + auxiliary Cox heads
F. Simple modality dropout or corruption-trained attention control (matched corruption budget)

## Evaluation
- Main metric: Harrell C-index per patient-level fold, paired differences, uncertainty intervals; also report integrated Brier score if censoring-aware evaluation is correctly implemented.
- 3-fold quick pilot is a screen, NOT publication-grade evidence. Scale to 5 folds x 5 cancers only if the method shows a credible signal.
- Record val/test performance, number of patients/events, feature counts, runtime, seed, git SHA, backbone, corruption severity, reliability/attention distributions.
- Distinguish generalization to *different cancer cohorts* from generalization across image encoders.
- Compare clean and corruption robustness at matched training budgets. Include failure cases.

## Go/no-go by Oct 27
Proceed only if (1) method has a meaningful conceptual distinction from related work, (2) gains over aux-only and ordinary gate appear on >=2 of 3 pilot cohorts or there is a strong reproducible robustness benefit, and (3) no data leakage. Otherwise pivot before full benchmark.

## Known implementation risks
- Prior BRCA audit observed only ~1,000 genes although config requested 20,000.
- Auxiliary supervision changes training objective; do not label it a pure optimizer.
- Image/gene gradient equalization harmed BRCA; do not reintroduce it by default.
- Verify test folds are untouched by selection.
- Existing code may use a two-logit head but Cox loss only reads logit index 1.

## Status
Pilot branch and baseline launcher configured. Reliability method NOT yet implemented or executed; this plan is a protocol, not experimental evidence.
