# Phase 1 Experiment Plan

## Design Principles

1. All methods use the same LR schedule (cosine annealing, epoch-level)
2. Adaptive methods modulate the *scheduled* LR, not a fixed reference
3. Disentanglement is presented as a training-objective variant, not a pure optimizer
4. Paired comparisons on identical folds with identical initialization

## Methods

### Primary (5-fold, 200 epochs)
| ID | Name | Base | Aux heads | Loss | LR schedule |
|----|------|------|-----------|------|-------------|
| P1 | adamw | AdamW | No | Cox | Cosine |
| P2 | adaptive_loss_ema | AdamW | No | Cox | Cosine × loss-EMA |
| P3 | disentangle_balance | AdamW | Yes | Cox + aux | Cosine |

### Reference (5-fold, 200 epochs)
| ID | Name | Base | Aux heads | Loss | LR schedule |
|----|------|------|-----------|------|-------------|
| R1 | sam | SAM(AdamW) | No | Cox | Cosine |
| R2 | msam | M-SAM | No | Cox | Cosine |

## Hyperparameters

### Fixed across all methods
- LR: 0.001
- Weight decay: 0.0001
- Batch size: 32
- Epochs: 200
- Cosine annealing: epoch-level, T_max=200, eta_min=0
- Model: JointModel (attention, dropout=0.3)
- Gene features: top-K by variance (K determined by data)
- Image: UNI 1024-dim embeddings

### Method-specific (documented, not tuned on test)
- Loss-EMA: decay=0.9, min_scale=0.5, max_scale=1.5
- Disentanglement: aux_weight=0.1, mode="disentangle_balance"
- SAM: rho=0.05
- M-SAM: rho=0.05

## Data
- Cancer: BRCA (1052 patients, 146 events)
- 5-fold stratified CV (stratify by vital_status)
- Split seed: 42 (same for all methods)
- Model init seed: 42 + fold_index

## Checkpoint Selection
- Save model state when val C-index improves
- Final test evaluation uses best-val-C-index checkpoint
- No test data used for any selection

## Pilot Reproduction (Stage B)
- 30 epochs, fold 0 only
- Methods: P1, P2, P3
- Purpose: verify training pipeline before committing to 200-epoch runs

## Ablation Plan (Stage D, after primary benchmark)
### D1: Disentanglement components
- AdamW (baseline)
- AdamW + auxiliary Cox only (no grad balancing)
- AdamW + grad balancing only (no aux heads)
- AdamW + auxiliary Cox + grad balancing (full disentangle_balance)

### D2: Adaptive LR components
- AdamW + cosine (baseline)
- AdamW + flat LR (no cosine, no adaptive)
- AdamW + loss-EMA only (no cosine)
- AdamW + cosine + loss-EMA (adaptive on top of cosine)

## Run Matrix
See RUN_MATRIX.csv for complete fold × method × seed assignments.
