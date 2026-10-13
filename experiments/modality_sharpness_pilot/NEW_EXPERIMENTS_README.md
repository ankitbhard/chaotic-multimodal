# New optimizer and WSI patch-ROI experiments (staged)

## Status / scientific scope

Implemented and wired into `run_pilot.py`:
- `new_sgd`: SGD+momentum reference
- `new_adamw`: AdamW reference
- `new_adaptive_step`: bounded loss-EMA step-size heuristic; **not Polyak-SAM**
- `new_gradient_diagnostic`: SGD with shared fusion-space modality gradient cosine logging; **does not perform gradient conflict correction**

The module also contains an `encoder_auxiliary` placeholder but it is **not enabled** in the CLI because no validated unimodal Cox auxiliary objectives have been implemented. The gradient-conflict experiment is **diagnostic only**. Do not describe either as a reproduction of RGM/DGL.

Independent patch-ROI preprocessing pilot:
- `roi_patch_pilot.py` consumes per-slide HDF5 `features` [N,D] with optional `coords` [N,2], selects diverse central patches without survival labels, and exports selected indices, coordinates and a pooled .npy vector.
- It is **not** an implementation of the prostate IRM/Gabor graph method, does not operate on raw WSI pixels, and is **not integrated into the survival loader**.
- Current `train_survival.py` averages patch features in `_load_embedding`; it cannot model spatial ROIs from that aggregate alone. Ensure patient/slide IDs and coordinate conventions are correct before integration. Do not substitute zero vectors for missing WSIs in a purported ROI comparison.

## Preflight

```bash
cd ~/chaotic-multimodal
git fetch origin
git switch experiments/modality-sharpness-pilot
git pull --ff-only origin experiments/modality-sharpness-pilot
python -m unittest experiments.modality_sharpness_pilot.test_new_experiments -v
```

Find a working cohort config and any EC2 data path overrides from your earlier successful pilot. Do not assume the config path exists.

## Pilot: optimizer baselines + diagnostic

```bash
python experiments/modality_sharpness_pilot/run_pilot.py \
  --config YOUR_WORKING_BRCA_CONFIG.yaml \
  --output results/new_optimizer_brca_fold0 \
  --fold 0 --epochs 30 \
  --optimizers new_sgd new_adamw new_adaptive_step new_gradient_diagnostic
```

Use the same data overrides, seed, and fold for each method. Results include per-epoch C-index and per-step optimizer history. Diagnostics cost logging and may affect runtime; use separate runs for clean timing. No GPU training has been executed as part of these commits.

## ROI feature preflight (requires patch-level HDF5)

```bash
python experiments/modality_sharpness_pilot/roi_patch_pilot.py \
  --input YOUR_PATCH_H5_DIRECTORY \
  --output results/roi_feature_pilot --top-k 64
```

Check feature dimensions, slide/patient linkage, patch coordinates, and output manifest. **Do not use this as a survival benchmark until a split-safe loader and matching validation are implemented.**

## Further research engineering required before large-scale launch

1. **Conflict correction:** derive independent modality-specific objectives or gradient contributions to common fusion parameters, then implement projected gradients with tests for conflicts, no-conflict behavior and gradient flow. Existing cosine alone is not evidence of objective conflict.
2. **Encoder/fusion disentanglement:** implement explicit unimodal Cox heads, train-only auxiliary losses, and carefully separated gradient pathways. Include objective-weight ablations and inference compatibility.
3. **Adaptive step-size:** establish a justified step-size rule and learning-rate scheduler interaction, compare against tuned AdamW/SGD, and avoid calling an EMA heuristic Polyak-SAM.
4. **WSI-to-ROI:** locate original WSIs or patch embeddings and coordinates; implement tissue masking/tiling where needed, ROI scoring and slide aggregation, and integrate through a new split-safe dataloader. Test mean pooling, ROI-only pooling, and full WSI graph as separate ablations.
5. Run fold-0 smoke tests, then 5 folds only if results justify cost. Do not claim statistical significance from one fold.

Do not launch a full overnight sweep on unverified code or missing ROI data.
