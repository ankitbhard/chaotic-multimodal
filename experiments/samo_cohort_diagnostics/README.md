# SAMO cohort diagnostics

Non-destructive exploratory analysis of cohort properties and fixed-objective validation performance. Branch: `experiment/samo-cohort-diagnostics`.

## Run on the existing experiment machine

From the repository root, after fetching this branch and making the same TCGA data/configs available as in the original SAMO experiment:

```bash
python -m experiments.samo_cohort_diagnostics.analyze \
  --cancers brca kirc stad --folds 3 --seed 42 \
  --samo-results results/samo_survival \
  --out results/samo_cohort_diagnostics
```

Outputs: `cohort_objective_features.csv` (one row per cohort/fold) and `report.json`.

Metrics are computed separately on training, validation, and test survival labels: sample count, events, censoring fraction, number/fraction of censoring-aware strictly comparable ordered pairs, and median follow-up/event duration. No model training or GPU is needed, but existing feature files and gene CSVs must be accessible because the standard data loader is reused.

Objective selection uses **only validation C-index**, with the method ordering used to break ties. The resulting associations are exploratory and in-sample: nine cohort-fold points are too few for strong inference, and the same validation observations used to pick the objective must not be used as an unbiased estimate of the selection rule's performance.

**Critical limitation:** Existing SAMO `summary.json` test C-indices are from each method's best checkpoint over the full training trajectory. They are NOT valid test C-indices for a model chosen at an earlier `--selection-epoch`; this script deliberately does not combine them. A confirmatory experiment needs an independent controller/selection split, fixed checkpoint selection, and one untouched held-out test evaluation.

### Follow-on experiment (not yet implemented)

- Record modality-only image and gene C-indices on validation, using trained auxiliary heads, without inspecting test outcomes.
- Test associations between cohort characteristics and *validation objective gains* across folds. Account for correlated folds from the same cohort.
- Pre-register a simple rule using training-only censoring/comparable-pair statistics and validation modality signal; compare against always-Cox and best globally fixed objective using leave-one-cancer-out evaluation on five cohorts.
- For unbiased test reporting, select both objective and checkpoint using validation-only data, then evaluate the selected checkpoint once on the outer test fold. Five cohorts are still too few for a strong generalization claim.
