# Modality-specific sharpness pilot (isolated)

This folder is a **new exploratory experiment**. No existing training code, data, configurations, or results have been changed.

## Four comparisons

- `sam`: existing vanilla SAM (all parameters).
- `msam`: existing M-SAM (original Shapley/OGM behavior, including known classification-based contribution issue).
- `block_fixed`: SAM perturbations on image and gene encoders with independent fixed radii. Fusion/head weights receive second-pass gradients but are not perturbed.
- `block_adaptive`: same blockwise SAM, with per-block radii adapted inversely to estimated loss sensitivity. This is a **heuristic prototype**, not a proven optimizer.

**Fairness note:** `sam` perturbs all parameters; block variants perturb only two encoders. This experiment compares both targeting and radius adaptation. The cleanest primary contrast is `block_fixed` versus `block_adaptive`. Additional fusion-targeting controls should follow.

## Data from existing code

The pilot imports `build_survival_dataloaders`, `cox_loss`, `evaluate`, and `build_model` from the existing repository. It accepts the same `configs/brca.yaml`, etc. Those configs reference **local paths**, not S3 URIs. The data must first be synchronized from S3 onto EC2. No S3 buckets or object prefixes were invented.

Example EC2 sync (replace placeholders with your actual S3 paths):

```bash
aws s3 sync s3://YOUR-BUCKET/YOUR-UNI-PREFIX/ /home/ubuntu/tangle_data/brca/brca/
aws s3 sync s3://YOUR-BUCKET/YOUR-GENE-PREFIX/ /home/ubuntu/tcga_brca_data/
```

The loader looks for `.pt`/`.h5` UNI embeddings (including nested `uni_features/tcga_features`), and gene CSVs with `sample_id`, `OS_months`, and `vital_status`. Check exact paths before running.

## Setup on EC2

From repository root:

```bash
git fetch origin
git switch experiments/modality-sharpness-pilot
python -m pip install torch numpy pandas scikit-learn pyyaml h5py
python -m pytest experiments/modality_sharpness_pilot/test_block_sam.py -q
python experiments/modality_sharpness_pilot/run_pilot.py --config configs/brca.yaml --output /home/ubuntu/pilot_results/brca --dry-run
```

Run a small pilot:

```bash
python experiments/modality_sharpness_pilot/run_pilot.py \
  --config configs/brca.yaml \
  --output /home/ubuntu/pilot_results/brca_fold0 \
  --fold 0 --n-folds 5 --epochs 10 \
  --optimizers sam msam block_fixed block_adaptive \
  --rho 0.05 --rho-image 0.05 --rho-gene 0.05 \
  --probe-every 20
```

Or override local paths explicitly:

```bash
python experiments/modality_sharpness_pilot/run_pilot.py \
  --config configs/brca.yaml --uni-dir /data/brca/uni \
  --gene-csv /data/brca/train.csv --gene-csv /data/brca/val.csv \
  --output /data/pilot_out --fold 0 --epochs 10
```

For all folds, run the command with `--fold 0`, `1`, `2`, `3`, and `4`, using distinct `--output` directories. Results are JSON files; existing results are never read or overwritten by this pilot.

## Scientific cautions

1. Original M-SAM uses classification accuracy for modality Shapley despite training Cox survival loss; preserve it as an **existing-code baseline**, not a corrected method.
2. Gene variance selection and standardization are fit on training fold only, following the existing loader.
3. Current loader uses mini-batch Cox risk sets and an unstratified inner validation split; these are inherited limitations.
4. The pilot fixes initial model weights across optimizers, but stochastic trajectories can differ because each optimizer consumes randomness differently.
5. The adaptive probe measures the **loss increase from a one-block perturbation**, not Hessian eigenvalues. Probe evaluation is extra compute.
6. Compare wall-clock and forward passes as well as C-index. Ten epochs is only a smoke/pilot test, not publication evidence.
7. No real S3 sync, EC2 execution, or full training has been performed by this change.

## Proposed next tests

Run BRCA fold 0, inspect radius histories and event counts, then extend to KIRC and LUAD. Add a fusion/head perturbation control and multiple seeds before drawing conclusions.
