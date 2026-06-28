# Adaptive Gradient Modulation for Multimodal Cancer Prognosis

<img src="paper/overleaf_sync/figures/fig_pipeline.png" width="1000px" align="center" />

**Adaptive Gradient Modulation (AGM)** for multimodal survival prediction from paired whole-slide histopathology images and bulk RNA-seq gene expression data.

[**[Paper]**](#citation) | [**[Results]**](#results)

## Overview

Multimodal cancer prognosis models that integrate histopathology and genomics often suffer from **modality imbalance** during training — image-derived gradients dominate optimization, limiting genomic representation learning. We investigate gradient modulation for multimodal cancer prognosis and propose **Adaptive Gradient Modulation (AGM)**, which dynamically regulates modality-specific gradients based on optimization state.

Unlike static approaches (e.g., OGM-GE) that apply fixed modulation throughout training, AGM intervenes only when:
1. **Imbalance gate**: modality contribution gap exceeds threshold (Δ > θ = 0.10)
2. **Trend gate**: the weaker modality is not already self-correcting

<img src="paper/overleaf_sync/figures/fig_imbalance_problem.png" width="800px" align="center" />

## Key Results

Evaluated on **2,952 matched patients** across five TCGA cancer cohorts (BRCA, LUAD, KIRC, STAD, LUSC) using Cox partial likelihood loss and Harrell's C-index.

| Method | BRCA | LUAD | KIRC | STAD | LUSC | Avg |
|--------|------|------|------|------|------|-----|
| PORPOISE (CV) | 0.638 | 0.589 | 0.729 | 0.595 | 0.584 | 0.627 |
| SGD+Mom (baseline) | 0.742 | 0.620 | 0.756 | 0.612 | 0.565 | 0.659 |
| **SGD+Mom+AGM (ours)** | **0.761** | **0.609** | **0.756** | **0.660** | 0.567 | **0.671** |

- AGM achieves the highest average C-index (0.671 vs. PORPOISE 0.627, +0.044 absolute)
- On LUAD, AGM recovers the degradation caused by static OGM-GE (+0.155 over SGD+Mom-GE)
- A 12-optimizer ablation reveals substantial cancer-specific optimization behavior

## Installation

### Prerequisites
- Linux (tested on Ubuntu 22.04)
- NVIDIA GPU with ≥16 GB VRAM (tested on NVIDIA T4)
- Python 3.10+
- PyTorch 2.0+

### Dependencies
```bash
pip install torch torchvision numpy pandas scikit-learn h5py pyyaml lifelines
```

## Data Preparation

### 1. Gene Expression (RNA-seq)
Download RNA-seq data from [UCSC Xena](https://xenabrowser.net/datapages/):
- **BRCA**: FPKM normalized (log1p)
- **LUAD, KIRC, STAD, LUSC**: HiSeqV2_PANCAN (log2 normalized)

Each gene CSV should contain columns for patient barcode, gene expression values, overall survival time (`OS.time`), and vital status (`OS`).

### 2. Whole-Slide Image Embeddings
Download pre-extracted patch embeddings:
- **BRCA**: [UNI embeddings](https://huggingface.co/MahmoodLab/UNI) (1,024-dim) from [TANGLE](https://github.com/mahmoodlab/TANGLE)
- **LUAD, KIRC, STAD, LUSC**: [UNI2-h embeddings](https://huggingface.co/MahmoodLab/UNI2-h-features) (1,536-dim)

### 3. Directory Structure
```
DATA_ROOT/
├── brca/
│   ├── gene_expression.csv
│   └── embeddings/          # .pt files (UNI 1024-dim)
├── luad/
│   ├── gene_expression.csv
│   └── embeddings/          # .h5 files (UNI2-h 1536-dim)
├── kirc/
│   ├── gene_expression.csv
│   └── embeddings/
├── stad/
│   ├── gene_expression.csv
│   └── embeddings/
└── lusc/
    ├── gene_expression.csv
    └── embeddings/
```

### 4. Configuration
Update the YAML config files in `configs/` with your data paths:
```yaml
# configs/brca.yaml
uni_dir: /path/to/brca/embeddings
gene_csv_paths:
  - /path/to/brca/gene_expression.csv
image_dim: 1024        # 1024 for UNI, 1536 for UNI2-h
n_top_genes: 20000
epochs: 200
batch_size: 32
```

## Training

### Single Optimizer
```bash
python train_survival.py --config configs/brca.yaml --optimizers sgd_mom_adaptive --epochs 200
```

### All 12 Optimizers (Single Cancer)
```bash
python train_survival.py --config configs/brca.yaml --optimizers all --epochs 200
```

### Full Ablation (All Cancers × All Optimizers)
```bash
bash run_all.sh
```

### Available Optimizers

| Group | Optimizer | Description |
|-------|-----------|-------------|
| **Baselines** | `sgd` | Vanilla SGD |
| | `sgd_mom` | SGD + Momentum (0.9) |
| | `adam` | Adam |
| | `adadelta` | Adadelta |
| | `cosine` | SGD + Cosine Annealing |
| **Always-on OGM-GE** | `sgd_ge` | SGD + OGM-GE |
| | `sgd_mom_ge` | SGD+Mom + OGM-GE |
| | `adam_ge` | Adam + OGM-GE |
| | `adadelta_ge` | Adadelta + OGM-GE |
| | `cosine_ge` | Cosine + OGM-GE |
| **AGM (ours)** | `adam_adaptive` | Adam + AGM (θ=0.10) |
| | `sgd_mom_adaptive` | SGD+Mom + AGM (θ=0.10) |

### Multi-Seed Evaluation
```bash
bash run_cindex_exp_a.sh    # 16 optimizers × 5 seeds × 5 cancers
```

### 5-Fold Cross-Validation
```bash
bash run_cindex_exp_b.sh    # 16 optimizers × 5 folds × 5 cancers
```

## Model Architecture

```
Input
├── Whole-slide image embedding (1024/1536-dim)
│   └── SlideEmbeddingEncoder: Linear → BN → ReLU → Dropout → [512-dim]
├── RNA-seq gene expression (20,000 genes)
│   └── GeneEncoder: MLP (20000 → 512 → 256 → 128)
│
├── Cross-Modal Attention Fusion
│   └── Softmax(α) per patient → weighted sum → [256-dim]
│
└── Survival Head
    └── MLP (256 → 256 → 1) → scalar risk score
    └── Cox partial likelihood loss
    └── Evaluation: Harrell's C-index
```

## Project Structure

```
chaotic-multimodal/
├── models/
│   ├── joint_model.py          # JointModel: image + gene → survival
│   ├── encoders.py             # SlideEmbeddingEncoder, GeneEncoder
│   └── fusion.py               # CrossModalAttention, ConcatFusion
├── optimizers/
│   ├── chaotic_optimizer.py    # MultimodalChaoticOptimizer
│   ├── ogm_wrapper.py          # OGMWrapper (OGM-GE without chaotic LR)
│   └── chaotic_lr_scheduler.py # Logistic-map LR scheduler
├── data/
│   └── dataset.py              # TCGAMultimodalDataset, build_dataloaders
├── configs/
│   ├── brca.yaml               # BRCA config (UNI 1024-dim)
│   ├── luad.yaml               # LUAD config (UNI2-h 1536-dim)
│   ├── kirc.yaml               # KIRC config
│   ├── stad.yaml               # STAD config
│   └── lusc.yaml               # LUSC config
├── train.py                    # Main training script (build_optimizer)
├── train_survival.py           # Cox PH survival training
├── run_all.sh                  # Full ablation script
├── run_cindex_exp_a.sh         # Multi-seed experiment
└── run_cindex_exp_b.sh         # 5-fold CV experiment
```

## Results

Output JSON files contain per-epoch metrics and test results:

```json
{
  "test_cindex": 0.761,
  "best_val_cindex": 0.742,
  "best_epoch": 128,
  "train_loss": [...],
  "val_cindex": [...],
  "risk_scores": [...]
}
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{bhardwaj2026agm,
  title={Adaptive Gradient Modulation for Multimodal Cancer Prognosis Prediction},
  author={Bhardwaj, Ankit},
  year={2026}
}
```

## Acknowledgments

- [PORPOISE](https://github.com/mahmoodlab/PORPOISE) — Multimodal survival prediction framework
- [TANGLE](https://github.com/mahmoodlab/TANGLE) — Multimodal foundation model and UNI embeddings
- [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h-features) — Pre-extracted histopathology features
- [UCSC Xena](https://xenabrowser.net/) — TCGA gene expression data
- [OGM-GE](https://github.com/GeWu-Lab/OGM-GE_CVPR2022) — On-the-fly Gradient Modulation

## License

This project is licensed under the GPLv3 License — see [LICENSE](LICENSE) for details.
