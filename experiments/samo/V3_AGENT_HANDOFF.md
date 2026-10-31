# RA-SAMO v3: EC2 agent handoff (READ BEFORE RUNNING)

Branch: experiment/censor-aware-ra-samo-v3. Goal: four variants x five cancers x five folds x 50 epochs = 100 final runs, after validation and smoke tests.

## Status: SCAFFOLD, NOT A COMPLETED FOUR-VARIANT IMPLEMENTATION
- Implemented: `censor_aware_controller.py` with three exact-zero modes (cox/aux/rank), train-only survival information, four variant identifiers, and initial heuristics.
- Wired into `run.py`: four method names, train-only event/pair features, epoch-wise objective gates, epoch CSV logging.
- Missing critical features: v3.2 gradient-alignment and gradient-variance inputs; v3.3 independent matched short-rollout inner-validation mode scoring. They currently fall back to v3.1-like behavior. **DO NOT report v3.2/v3.3 as distinct experiments until completed.**
- The controller is a proposed heuristic, not an established censoring formula. No cancer-name switches; do not fit against prior outer test scores.

## Required engineering work
1. Inspect all code, configs, data and current branch. Run `python -m unittest experiments.samo.test_censor_aware_controller` and all pre-existing SAMO tests. Check `loss_parts` for correct v2 gene auxiliary bug fix and v3 losses.
2. Implement v3.2: estimate gradient cosine and reliability on *training batches only*, periodically (e.g. every 5 epochs) to control overhead. Reuse `representation_gradients.py`; avoid altering gradients used for AdamW, except explicitly documented. Feed `alignment={"aux":...,"rank":...}` into controller. Ensure v3.2 differs meaningfully from v3.1; record features and time/memory.
3. Implement v3.3: from SAME checkpoint, clone model/optimizer/RNG state, perform equal-budget short candidate rollouts for cox/aux/rank using TRAINING batches only, score on an **inner validation split derived only from the outer-training partition**, and restore all original state afterward. Do not use outer validation for controller tuning if it is also used for checkpoint selection without explicitly documenting adaptive reuse. Never use outer test for decisions. Feed scores into controller; implement confidence margin and min dwell. Record extra GPU compute.
4. Correct v3.0/v3.1 training features: ensure the loss-progress signal is meaningful even when an objective is gated off, and confirm censoring/pair density calculated only from outer training. Avoid drawing survival labels from validation/test.
5. Implement stronger tests: exact-zero Cox equivalence, correct aux/rank gating, censored/tied pairs, no leakage, switch hysteresis, gradient feature propagation, independent rollouts and restoration, identical initial weights, deterministic seeds, resume/skip.
6. Test 2-epoch smoke on BRCA and KIRC for all four; confirm methods produce different decisions when features differ. Fail loudly if features for v3.2/v3.3 are missing rather than silently degrade.
7. Run 100 full runs only after readiness gate. Compare with matched baseline, adaptive_aux, rank, SAMO, RA-SAMO v1/v2 using identical splits, seed, preprocessing, model, budget. If previous runs are compatible, reuse; otherwise rerun references.
8. Report per-cancer and pooled means, fold-level paired deltas, confidence intervals, selected-mode fractions, switch count, event and pair statistics, gradient signals, compute cost, and limitations. Do not pick a final winner using the same outer test folds and claim unbiased performance. Use inner validation to choose the candidate, then report outer test performance; for definitive claims use a separate locked evaluation.
9. Save results as `results/ra_samo_v3_50ep/{cancer}/fold_{fold}/{method}/{epochs.csv,summary.json}`, final_report.md, plots, logs. Commit source, reports, compact summaries to this branch. Don't commit huge model checkpoints. Optionally back up to approved S3 if configured.
10. For unattended EC2: verify EBS persistence and instance-initiated shutdown behavior STOP, not TERMINATE; check passwordless sudo. Use tmux/nohup/systemd, logs, an EXIT trap that records failures, and shutdown only when safe. EC2 stopping must never risk losing unpersisted results. Agent must launch, monitor, fix, resume and confirm EC2 stopped.

## Scientific guardrails
The observed cohort targets (BRCA/STAD aux, KIRC Cox, LUAD/LUSC rank) came from prior **test** results and must be diagnostic only, never hard-coded training labels or validation targets. The objective is generalization, not retrospective matching. The reported 100 runs exclude v3.3 candidate-rollout compute.
