# CVPR Reliability Pilot — PCRF

Perturbation-Calibrated Reliability-Guided Fusion for multimodal cancer survival prediction.

## Quick Start

```bash
# Run unit tests
python experiments/cvpr_reliability_pilot/test_pilot.py

# Smoke test (5 epochs, fold 0, BRCA)
python -m experiments.cvpr_reliability_pilot.train \
    --config configs/brca.yaml --methods adamw pcrf_aux --smoke

# Full pilot (3 folds × 100 epochs × 3 cancers)
bash experiments/cvpr_reliability_pilot/run_pilot.sh

# Robustness evaluation
python -m experiments.cvpr_reliability_pilot.evaluate \
    --config configs/brca.yaml \
    --results_dir results/cvpr_reliability_pilot/brca \
    --methods all --folds 3

# Statistical analysis
python -m experiments.cvpr_reliability_pilot.analyze_results \
    --results_base results/cvpr_reliability_pilot \
    --cancers brca kirc luad --folds 3
```

## Methods

| ID | Description | Architecture |
|----|-------------|--------------|
| `adamw` | Standard attention fusion baseline | JointModel |
| `aux_cox` | + Auxiliary unimodal Cox heads | JointModel + AuxCoxHead |
| `mod_dropout` | + Training-time modality dropout (p=0.15) | JointModelWithDropout |
| `learned_gating` | PCRF architecture, no reliability loss | PCRFModel (capacity control) |
| `pcrf` | + Perturbation-calibrated reliability | PCRFModel |
| `pcrf_aux` | + Reliability + aux Cox supervision | PCRFModel (full method) |

## Results

See `PILOT_REPORT.md` for complete results. Key finding: pcrf_aux significantly
improves on 2/3 pilot cancers (BRCA p=0.002, LUAD p=0.016) but the contribution
of reliability supervision vs. auxiliary supervision + capacity remains unclear.

## Files

| File | Description |
|------|-------------|
| `models.py` | PCRFModel, JointModelWithAux, JointModelWithDropout |
| `losses.py` | Cox loss, reliability loss, combined loss |
| `reliability.py` | Perturbation calibrator, diagnostics, collapse detection |
| `train.py` | Training script for all 6 methods |
| `evaluate.py` | Robustness evaluation suite |
| `test_pilot.py` | 23 unit tests |
| `analyze_results.py` | Statistical analysis and visualization |
| `METHODOLOGY.md` | Mathematical formulation |
| `PILOT_REPORT.md` | Full results and go/no-go decision |
