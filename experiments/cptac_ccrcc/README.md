# CPTAC-CCRCC independent-cohort feasibility audit

This is **phase 1** of external validation, not an external validation result. It audits whether independent CPTAC clear-cell renal cell carcinoma cases have patient-matched whole-slide pathology, RNA expression, and time-to-event clinical outcomes. It does not download protected data, train models, or inspect external test performance.

## Source data

- TCIA CPTAC-CCRCC pathology collection: https://www.cancerimagingarchive.net/collection/cptac-ccrcc/
- NCI CPTAC/GDC genomic and clinical resources: https://gdc.cancer.gov/about-gdc/contributed-genomic-data-cancer-research/clinical-proteomic-tumor-analysis-consortium-cptac

Download or export **three patient-mapped manifests** from official sources, each CSV or TSV:
- `slides.csv`: `patient_id,slide_id` (one row per WSI; multiple slides per patient allowed).
- `genes.csv`: `patient_id,sample_id` (one row per RNA sample; confirm tumor RNA and expression modality).
- `clinical.csv`: `patient_id,os_months,vital_status` (duration > 0, event=1 death, event=0 censored).

The source identifiers may differ across TCIA, GDC and clinical tables. **Reconcile IDs using official mapping tables before running the audit.** Never infer patient identity from partial filename matches. Use consistently defined overall survival origin and duration units. Do not commit patient-level manifests or identifiable clinical data to GitHub.

## Run

```bash
python -m experiments.cptac_ccrcc.audit \
  --slides /path/to/slides.csv \
  --genes /path/to/genes.csv \
  --clinical /path/to/clinical.csv \
  --time-col os_months --event-col vital_status \
  --out results/cptac_ccrcc_audit
```

The script writes `eligibility_report.json` and `eligible_patients.csv`, reports overlap and event counts, and exits with status 2 if fewer than 100 matched patients or 30 events (exploratory thresholds only). IDs are treated as exact, case-normalized identifiers; no automatic cross-source ID matching is attempted.

## External validation plan after feasibility

1. Lock the TCGA-KIRC training protocol, selected objective, feature normalization and checkpoint **without consulting CPTAC outcomes**.
2. Generate CPTAC pathology embeddings using the same frozen encoder and aggregation pipeline as TCGA-KIRC (currently UNI2-h, 1536 dimensions). Verify slides are tumor slides, patient aggregation, and tissue preparation compatibility.
3. Align gene identifiers to the **training-derived TCGA gene panel** in the exact order. Apply **TCGA-fitted** preprocessing/scaler only; do not refit on external cohort. Handle missing genes with a documented predeclared policy.
4. Apply the frozen TCGA-trained model to eligible CPTAC cases. Compute patient-level Harrell C-index and bootstrap confidence intervals; report the number of events, censoring, calibration caveats and dataset shift.
5. Optionally compare frozen TCGA baseline and frozen TCGA-SAMO on the same eligible CPTAC patients. Do not tune on external outcomes. If external fine-tuning is desired, treat it as a **separate** train/validation/test experiment with disjoint patient splits.

**Not yet implemented:** data acquisition/mapping adapters, UNI2-h WSI extraction, frozen-model loading and external evaluation. Those require inspection of the available source manifests and checkpoint formats. Avoid presenting this phase-1 audit as completed external validation.
