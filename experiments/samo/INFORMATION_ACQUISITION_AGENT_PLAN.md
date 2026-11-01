# Information Acquisition Dynamics — Agent Execution Plan

Branch: `experiment/information-acquisition-diagnostics`
Source: v3 branch (commit 864d5dd). **Diagnostic study, not a new optimizer or claimed result.**

## Question
Do image/gene encoder learning dynamics (gradient-sensitivity proxy, gradient norm, unimodal C-index) explain why adaptive auxiliary Cox improves BRCA, Cox-only leads KIRC, and ranking leads LUAD?

## Scope
Three cohorts (brca, kirc, luad), three methods (baseline, adaptive_aux, rank), three folds (0–2), 50 epochs = 27 runs. Match existing 250-run settings (seed 42, preprocessing, architecture, split, LR, weight decay, checkpoint selection). **Prefer instrumented reruns**; existing logs can provide preliminary C-index and loss trajectories but not missing Fisher measurements.

## Measurement protocol
1. Build a **fixed, deterministic training-only probe subset** per fold, stratified by event indicator where possible; record patient IDs/hashes and event count. Never sample validation or test patients for Fisher or gradient diagnostics.
2. Every epoch, or at epochs 1–15 then 20,25,30,40,50, calculate modality encoder gradient diagnostics on probe data with model in eval mode and gradients enabled. Preserve/restore RNG state, .grad fields, train/eval mode and optimizer state; no optimizer steps on probe.
3. For survival Cox likelihood, compute empirical per-observation score-gradient squared norm only if its statistical meaning is documented: Cox partial likelihood couples individuals via risk sets. A batch loss gradient squared norm is **not automatically the empirical Fisher trace**. Report it separately as a **squared Cox gradient norm proxy**. If implementing an empirical Fisher trace, derive risk-set-aware per-event score contributions and sum squared per-event parameter-score norms, stating tie handling, batch vs full-risk-set approximation, and censoring assumptions. Compare image and gene traces using parameter-count normalization and layerwise breakdown.
4. Calculate fused Cox gradient and each auxiliary branch's Cox gradient on the same probe batches, using autograd.grad and retain_graph only when necessary; compute norms, cosines and gradient variance across multiple probe batches. Ensure aux heads are available for diagnostic-only evaluation on baseline/rank **without changing baseline/rank model capacity or training**; alternatively log unimodal linear probes fitted on separate train-only data, and explain their limitations.
5. At each epoch, evaluate fused model C-index on validation. For auxiliary models, evaluate image and gene auxiliary-head C-index; for baseline/rank use separately trained frozen-feature linear Cox probes with train-only fitting, or use consistent predeclared ablations. Do not compare trained auxiliary-head scores to untrained heads. C-index should use consistent censoring and tie handling.
6. Save `diagnostics.csv` with epoch, method, fold, cancer, gradient proxy/trace definitions, layerwise and normalized statistics, fused/unimodal validation CI, observed events, risk-pair counts, walltime, memory; save `metadata.json` (probe IDs, config, commit, seeds, implementation definitions). Plot learning windows and correlations with subsequent changes in validation CI; don't claim causality from correlations.
7. Compare trajectories across methods with paired fold plots and variability. Analyze whether early gene/image imbalance predicts subsequent underperformance, and whether auxiliary learning improves weaker modality.
8. Produce `results/information_acquisition_diagnostics/final_report.md` with methodology, equations, plots, findings, limitations and concrete go/no-go recommendation for an InfoReg-inspired intervention.

## Required tests and readiness
- Validate gradient-probe invariance: instrumented run matches uninstrumented baseline within deterministic tolerance, and optimizer/model state unchanged by probes.
- Test all-censored batches, no comparable pairs, tied event times, missing aux heads, empty probes, finite gradients, and deterministic probe sampling.
- Run 2-epoch smoke tests BRCA/KIRC for all 3 methods; check output columns and GPU memory before full runs.
- Confirm no train/val/test leakage and identical data partitions.
- Do not label squared batch Cox gradient norm as Fisher trace without implementing the correct estimator.
- Do not compare raw traces across different-size encoders without normalized diagnostics.

## Run and completion
Only after tests, run 27 instrumented training runs on EC2; checkpoint and resume safely. Save artifacts on persistent EBS; commit code, plots, small CSVs, final report to branch. Record source SHA. Verify EC2 shutdown behavior is STOP (not TERMINATE); stop after successful persistence, report actual completion status.

## Scientific interpretation
Existing five-fold test winners are exploratory; do not tune diagnostic thresholds using test results. This pilot is exploratory and cannot prove the superiority of a future optimizer. If no reproducible signal emerges, report negative findings rather than inventing a regulation mechanism.
