# RA-SAMO Architecture Audit & Attention MIL Feasibility Report

**Date:** 2026-11-03
**Branch:** `experiment/ra-samo-ham10000-classification`
**Scope:** Read-only audit — no code modified

---

## Phase 1: Verified Architecture

### 1.1 Where is MeanPool implemented?

**MeanPool is performed during data loading, NOT during model training.**

Two independent implementations exist, both doing the same thing:

| File | Function | Lines | What it does |
|------|----------|-------|-------------|
| `data/dataset.py` | `_load_embedding()` | 54–79 | `feat.mean(0)` on `[N,D]` → `[D]` |
| `train_survival.py` | `_load_embedding()` | 96–115 | `t.mean(0)` on `[N,D]` → `[D]`; `.reshape(-1, t.shape[-1]).mean(0)` for h5 |

The SAMO experiment uses `train_survival.py:build_survival_dataloaders()` (line 157), which calls `train_survival.py:_load_embedding()` (line 96). This mean-pools at load time and caches the result per patient.

**Verified:** The model never sees patch-level features. By the time data reaches the model, each slide is a single `[1024]` vector.

### 1.2 What does the model receive?

**One mean-pooled slide embedding per patient: `[B, 1024]`.**

- `SurvivalDataset.__getitem__()` (`train_survival.py:128–142`) returns `{"image": [1024], "gene": [n_genes]}`
- The DataLoader stacks these into `{"image": [B, 1024], "gene": [B, n_genes]}`
- No patch dimension exists at model input time

### 1.3 Are UNI/UNI2-h parameters frozen?

**UNI/UNI2-h is not in the model at all.** The `.pt` / `.h5` files contain pre-extracted features. Only the downstream `SlideEmbeddingEncoder` (a linear projection) is trained.

### 1.4 Is SlideEmbeddingEncoder trainable?

**Yes, under both AdamW and RA-SAMO.** It is part of `model.image_encoder` and all its parameters have `requires_grad=True`.

- `SlideEmbeddingEncoder` (`models/encoders.py:12–33`): `Linear(1024, 512) + BN + ReLU + Dropout`
- `FEATURE_DIM = 512`

### 1.5 What does the current "attention" do?

**Cross-modal attention between modalities, NOT between WSI patches.**

`CrossModalAttention` (`models/fusion.py:17–50`) computes per-patient softmax weights over two modalities:
```
[B, 512] (image) + [B, 128] (gene) → project to [B, 2, 256] → attention → [B, 256]
```
Output: `α_image, α_gene` per patient — which modality to trust more.

### 1.6 RA-SAMO gradient flow

`prepare_branch_gradients()` (`experiments/samo/representation_gradients.py:49–91`) operates on:

```python
params = getattr(model, "image_encoder").parameters()  # line 64
params = getattr(model, "gene_encoder").parameters()    # line 64
```

**Gradient recipients by loss component:**

| Loss | Image encoder | Gene encoder | Fusion | Shared/Head | Aux heads |
|------|:---:|:---:|:---:|:---:|:---:|
| `fusion_loss` (Cox on fused risk) | via backprop | via backprop | via backprop | via backprop | — |
| `image_loss` (aux Cox on image) | **replaced by RA-SAMO** | — | — | — | via backprop |
| `gene_loss` (aux Cox on gene) | — | **replaced by RA-SAMO** | — | — | via backprop |
| `ranking_loss` (pairwise ranking) | **replaced by RA-SAMO** | **replaced by RA-SAMO** | — | — | — |

The key mechanism: RA-SAMO computes gradients from aux/ranking losses on encoder parameters, projects away conflicting components (if `project=True`), then replaces `param.grad` after `loss.backward()`. Fusion, shared, and head parameters get normal backprop gradients.

### 1.7 Full architecture diagram (verified from code)

```
Input: {"image": [B, 1024], "gene": [B, n_genes]}
                    │                      │
         SlideEmbeddingEncoder      GeneEncoder
         Linear(1024→512)+BN+ReLU   3×Linear MLP
                    │                      │
                [B, 512]             [B, 128]
                    │                      │
                    └───── CrossModalAttention ─────┘
                           project → stack → attention
                                    │
                              [B, 256] fused
                                    │
                           Shared (Linear+ReLU+Drop)
                                    │
                          head_survival (Linear→2)
                                    │
                              [B, 2] logits

  Auxiliary (JointModelWithAux only):
  img_feat [B,512] → AuxCoxHead → [B] risk
  gene_feat [B,128] → AuxCoxHead → [B] risk
```

### 1.8 Model used in SAMO experiments

`experiments/samo/run.py:198`: Methods other than `baseline`/`rank` use `JointModelWithAux` from `experiments/cvpr_reliability_pilot/models.py:242`. This has:
- `.image_encoder` = `SlideEmbeddingEncoder`
- `.gene_encoder` = `GeneEncoder`
- `.fusion` = `CrossModalAttention`
- `.aux_img` = `AuxCoxHead(512)`
- `.aux_gene` = `AuxCoxHead(128)`

---

## Phase 2: Patch-Level Embedding Availability

### 2.1 What do the `.pt` / `.h5` files contain?

**Evidence from code — patch-level embeddings `[N_patches, D]`:**

Both `_load_embedding()` implementations explicitly handle multi-patch tensors before mean-pooling:

```python
# train_survival.py:100-102
if t.dim() > 1:
    t = t.mean(0)    # [N, 1024] → [1024]

# data/dataset.py:72-75
if feat.dim() == 3:
    feat = feat.squeeze(0).mean(0)   # [1, N, D] → [D]
elif feat.dim() == 2:
    feat = feat.mean(0)              # [N, D] → [D]
```

The self-test in `data/dataset.py:539` creates synthetic data with shape `[50, 1024]` per `.pt` file — confirming that patch-level embeddings with ~50 patches per slide are expected.

**CANNOT VERIFY** actual tensor shapes on EC2 data without instance access. No local BRCA `.pt` files exist at `/Users/ankitbhardwaj/Documents/chaotic-multimodal-v2/tangle_data/brca/`.

### 2.2 Data location and layout

| Item | Value | Source |
|------|-------|--------|
| BRCA path (EC2) | `~/tangle_data/brca/brca/` | `configs/brca.yaml:8` |
| File count | 1,124 `.pt` files | MEMORY.md |
| Feature dim | 1024 (UNI) | `configs/brca.yaml:16` |
| KIRC path (EC2) | `~/tangle_data/kirc` | `configs/kirc.yaml` (presumed) |
| KIRC files | 519 `.h5` files | MEMORY.md |
| KIRC feature dim | 1536 (UNI2-h) | MEMORY.md |

### 2.3 Patient-slide mapping

`_discover_uni_dir()` + barcode extraction: `fname[:12]` maps filenames to 12-character TCGA patient barcodes. **One embedding file per patient** (deduplication in `build_survival_dataloaders` via `drop_duplicates(subset="patient_barcode")`).

### 2.4 What CANNOT be verified without EC2 access

1. **Actual tensor shapes** — are they truly `[N_patches, 1024]` or already `[1024]`?
2. **Patch count distribution** — how many patches per slide? (affects memory planning)
3. **Whether patch coordinates are stored** (unlikely given `.pt` format)
4. **Whether multiple WSIs exist per patient** (code assumes 1:1)

**Recommended verification commands (run on EC2):**
```bash
python3 -c "
import torch, os
d = os.path.expanduser('~/tangle_data/brca/brca')
files = [f for f in os.listdir(d) if f.endswith('.pt')][:10]
for f in files:
    t = torch.load(os.path.join(d,f), map_location='cpu')
    print(f'{f}: shape={t.shape}, dtype={t.dtype}')
"
```

### 2.5 Split leakage

**No leakage detected.** `build_survival_dataloaders()` splits at the patient level using 12-character TCGA barcodes (`fname[:12]`), then does StratifiedKFold on vital_status. Train/val/test are patient-disjoint.

---

## Phase 3: Best RA-SAMO Variant

### 3.1 Available results

**Most comprehensive run:** `results/samo_all_cancers_50ep/` — 5 cancers × 5 folds × 10 methods = 250 runs.

### 3.2 BRCA Test C-Index (5-fold, 50 epochs)

| Method | Mean Test CI | Std | Best Val CI |
|--------|:-----------:|:---:|:-----------:|
| **adaptive_aux** | **0.6780** | **0.0570** | **0.7073** |
| ra_samo_v1 | 0.6664 | 0.0762 | 0.6987 |
| aux | 0.6624 | 0.0489 | 0.6934 |
| **ra_samo** | **0.6580** | **0.0501** | **0.7122** |
| ra_samo_no_proj | 0.6548 | 0.0679 | 0.6942 |
| aux_rank | 0.6502 | 0.0758 | 0.6949 |
| ra_samo_v3_0 | 0.6450 | 0.0655 | 0.6971 |
| samo | 0.6317 | 0.0517 | 0.6885 |
| baseline (AdamW) | 0.6288 | 0.0754 | 0.6918 |
| ra_samo_v2 | 0.6288 | 0.0551 | 0.7024 |
| rank | 0.6207 | 0.0584 | 0.6864 |

### 3.3 KIRC Test C-Index (5-fold, 50 epochs)

| Method | Mean Test CI | Std | Best Val CI |
|--------|:-----------:|:---:|:-----------:|
| **baseline (AdamW)** | **0.7395** | **0.0567** | **0.7837** |
| adaptive_aux | 0.7313 | 0.0599 | 0.7761 |
| rank | 0.7309 | 0.0437 | 0.7736 |
| aux_rank | 0.7308 | 0.0596 | 0.7753 |
| samo | 0.7306 | 0.0520 | 0.7788 |
| ra_samo_no_proj | 0.7289 | 0.0576 | 0.7756 |
| aux | 0.7284 | 0.0612 | 0.7801 |
| ra_samo_v1 | 0.7273 | 0.0483 | 0.7719 |
| ra_samo | 0.7237 | 0.0568 | 0.7725 |
| ra_samo_v2 | 0.7195 | 0.0550 | 0.7754 |

### 3.4 Best candidate assessment

**BRCA:**
- `adaptive_aux` is the top performer (0.678) but is NOT an RA-SAMO variant — it uses dynamic weights without gradient projection.
- `ra_samo` (with projection) achieves 0.658 (+0.029 over baseline) with the highest validation CI (0.712), suggesting good model selection.
- `ra_samo_v1` (with progress scaling) achieves 0.666 but with high variance (0.076).

**KIRC:**
- Baseline AdamW is the best performer (0.740). All RA-SAMO variants slightly underperform.
- This suggests RA-SAMO's auxiliary objectives may not help when the baseline already leverages both modalities effectively.

**Recommended variant for the 2×2 experiment: `ra_samo`**
- Rationale: It is the core RA-SAMO variant (gradient projection enabled). On BRCA it shows meaningful improvement (+0.029) with relatively low variance (0.050). The highest val CI (0.712) suggests robust checkpoint selection. On KIRC it slightly underperforms, making the interaction test more informative — we can see if Attention MIL helps specifically where RA-SAMO already struggles.

### 3.5 Flags and caveats

1. **High variance across folds** — all methods show std 0.05–0.08, making differences of 0.01–0.03 statistically uncertain without paired tests.
2. **KIRC: RA-SAMO ≤ baseline** — gradient projection may not help when modalities are already well-balanced.
3. **No v3 variant improves over basic ra_samo** — additional complexity (censor-aware, alignment-based) did not help.
4. **v3_2 and v3_3 produce identical results** on both cancers — suggestive of a bug or no rollout divergence.
5. **modality_imbalance_v1 results are worse** (BRCA ra_samo=0.595, 3-fold only) — possibly different hyperparameters or earlier code version.
6. **Training time not recorded** in summary.json `wall_seconds` field — need to check.

---

## Phase 4: Attention MIL Feasibility

### 4.1 Proposed architecture change

```
Current:
  .pt [N, 1024] → mean_pool → [1024] → SlideEmbeddingEncoder → [512]

Proposed:
  .pt [N, 1024] → GatedAttentionAggregator → [1024] → SlideEmbeddingEncoder → [512]
```

The GatedAttentionAggregator (from CLAM, Lu et al. 2021) computes:
```
a_k = softmax( V^T (tanh(W1 h_k) ⊙ sigmoid(W2 h_k)) )
z = Σ a_k h_k
```

### 4.2 Does this preserve the downstream architecture?

**Yes, completely.** The aggregator outputs `[B, 1024]` — the same shape that the current mean-pool produces. Everything downstream (`SlideEmbeddingEncoder → CrossModalAttention → shared → head`) remains identical.

### 4.3 Variable patch counts

**Challenge:** Standard batching requires fixed tensor dimensions. With variable N_patches per slide, options are:

1. **Custom collate + loop:** Concatenate all patches into `[total_patches, 1024]` + `patch_counts [B]`. Process each slide's patches sequentially in the image encoder.
2. **Padding + masking:** Pad to max_patches, use attention mask to ignore padding.
3. **Pre-aggregation capping:** Subsample each slide to fixed max_patches (e.g., 4096).

**Recommended:** Option 1 (custom collate) with option 3 (capping at 4096). This is what the plan specifies and is standard in MIL literature.

### 4.4 RA-SAMO compatibility

**Compatible without code changes to `representation_gradients.py`.**

`prepare_branch_gradients()` line 64:
```python
params = [p for p in getattr(model, "image_encoder").parameters()
          if p.requires_grad]
```

If we define a new `CLAMImageEncoder` class containing both the `GatedAttentionAggregator` and the original `SlideEmbeddingEncoder` projection, and assign it to `model.image_encoder`, then `model.image_encoder.parameters()` returns all CLAM attention weights + projection parameters. Gradient projection applies to these naturally.

**Key requirement:** The new model must have `model.image_encoder` and `model.gene_encoder` attributes.

### 4.5 What must change for `[B, N, D]` input

Currently the model expects `inputs["image"]` to be `[B, D]`. With Attention MIL:

1. **Dataset:** Return `[N_i, 1024]` per sample instead of `[1024]`
2. **Collate function:** Concatenate variable-length patches + return `patch_counts`
3. **Image encoder forward:** Accept `(all_patches [total, 1024], patch_counts [B])` and return `[B, 512]`
4. **Joint model forward:** Route the new input format to the new image encoder

**Files requiring modification:** None of the existing shared files. All changes are in new files under `experiments/clam_brca/`.

### 4.6 GPU memory estimate (NVIDIA T4, 16GB)

| Component | Size |
|-----------|------|
| Patches (batch=8, max=4096) | 8 × 4096 × 1024 × 4B = 128 MB |
| Attention scores | 8 × 4096 × 4B = ~0.1 MB |
| Gradients (attention + projection) | ~130 MB |
| Model + optimizer state | ~50 MB |
| Gene encoder + fusion | ~40 MB |
| **Total estimate** | **~350 MB** |

Well within T4 16GB. Even with gradient accumulation (4 steps), peak memory stays under 500 MB because only one micro-batch is active at a time.

### 4.7 Attention MIL parameter overhead

| Component | Parameters |
|-----------|-----------|
| W1: Linear(1024, 256) | 262,400 |
| W2: Linear(1024, 256) | 262,400 |
| V: Linear(256, 1) | 257 |
| **Total new params** | **~525K** |

Current `SlideEmbeddingEncoder`: 525,312 params. Attention MIL roughly doubles the image encoder parameter count. This should be reported in the experiment but is not a concern for a T4.

---

## Phase 5: Controlled 2×2 Experiment Design

### 5.1 Experiment matrix

| Experiment | Aggregation | Training | Aux Heads | Grad Projection |
|:---:|:---:|:---:|:---:|:---:|
| E1 | MeanPool | AdamW | No | No |
| E2 | MeanPool | RA-SAMO | Yes | Yes |
| E2a | MeanPool | AdamW + aux losses | Yes | No |
| E3 | Attention MIL | AdamW | No | No |
| E4 | Attention MIL | RA-SAMO | Yes | Yes |
| E4a | Attention MIL | AdamW + aux losses | Yes | No |

**E2a and E4a are critical controls** — they isolate the effect of gradient projection from the effect of auxiliary objectives. Without them, any improvement from E2 over E1 could be attributed to the extra loss signal rather than RA-SAMO's projection mechanism.

### 5.2 Shared protocol

| Setting | Value | Source |
|---------|-------|--------|
| Cancer | BRCA | Primary test |
| Patients | ~1,096 matched | `configs/brca.yaml` |
| Folds | 5 (StratifiedKFold on vital_status) | Match `samo_all_cancers_50ep` |
| Seed | 42 | Match existing |
| Epochs | 50 | Match existing |
| LR | 0.001 | Match existing |
| Weight decay | 1e-4 | Match existing |
| LR schedule | Cosine annealing | Match existing |
| Optimizer | AdamW | Match existing |
| Survival loss | `cox_efron_free` (Breslow) | Match existing |
| Checkpoint selection | Best validation C-index | Match existing |
| Batch size | 32 (MeanPool) / 8×4 accum (CLAM) | Match effective batch |
| Gene preprocessing | top-20K by variance, StandardScaler | Match existing |
| Fusion | CrossModalAttention | Match existing |
| Dropout | 0.3 | Match existing |
| max_patches (CLAM only) | 4096 | New |
| aux_weight | 0.1 | Match existing |
| rank_weight | 0.1 | Match existing |

### 5.3 What must be identical across all 6 runs

1. **Patient splits** — same StratifiedKFold with same seed
2. **Gene preprocessing** — same top genes, same scaler fit on same train set
3. **Survival labels** — same `OS_months` + `vital_status` from same CSV
4. **Random initialization** — same seed for model init (then copy matching weights for aux models)
5. **Evaluation** — same `concordance_index()` implementation, same `cox_efron_free()` for val loss

### 5.4 What differs

1. **Image input:** `[B, 1024]` (MeanPool) vs `[total_patches, 1024] + patch_counts` (CLAM)
2. **Image encoder:** `SlideEmbeddingEncoder` (525K params) vs `CLAMImageEncoder` (~1.05M params)
3. **Auxiliary heads:** Present in E2/E2a/E4/E4a, absent in E1/E3
4. **Gradient projection:** Active only in E2/E4

### 5.5 Trainable parameter counts (estimated)

| Component | E1 | E2/E2a | E3 | E4/E4a |
|-----------|:---:|:---:|:---:|:---:|
| Image encoder | 525K | 525K | 1,050K | 1,050K |
| Gene encoder | 10.4M | 10.4M | 10.4M | 10.4M |
| Fusion | 165K | 165K | 165K | 165K |
| Shared + head | 66K | 66K | 66K | 66K |
| Aux heads | — | 34K | — | 34K |
| **Total** | **~11.2M** | **~11.2M** | **~11.7M** | **~11.7M** |

### 5.6 Fairness concerns

1. **Parameter count asymmetry:** CLAM adds ~525K parameters to the image encoder. This could improve performance purely from increased capacity, independent of attention. **Mitigation:** Compare E3 (CLAM+AdamW) vs E1 (MeanPool+AdamW) to measure the capacity effect alone.

2. **Batch size difference:** CLAM requires smaller batch size (8 vs 32) due to variable-length patches. Gradient accumulation (4 steps) gives effective batch 32, but gradient noise characteristics differ. **Mitigation:** Accept this as inherent to the MIL approach; document it.

3. **Auxiliary objective asymmetry:** E2/E4 have extra loss terms that E1/E3 lack. This confounds "RA-SAMO effect" with "extra supervision effect." **Mitigation:** E2a/E4a controls isolate this.

---

## Phase 6: Evaluation Plan

### 6.1 Primary metric

**Held-out test C-index**, reporting mean ± std across 5 folds, computed with `concordance_index()` from `train_survival.py:51`.

### 6.2 Secondary metrics

| Metric | Purpose |
|--------|---------|
| Best validation C-index | Checkpoint selection quality |
| Best epoch | Convergence speed |
| Wall-clock time | Computational overhead |
| Training loss curves | Convergence behavior |
| Image/gene gradient norms | RA-SAMO diagnostic |
| Modality attention weights (cross-modal) | Balance indicator |
| Patch attention entropy (CLAM only) | Selectivity indicator |
| GPU peak memory | Practical feasibility |

### 6.3 Interaction analysis

```
MIL_Gain_AdamW   = CI(E3) - CI(E1)       # Does CLAM help without RA-SAMO?
MIL_Gain_RA_SAMO = CI(E4) - CI(E2)       # Does CLAM help with RA-SAMO?
Interaction      = MIL_Gain_RA_SAMO - MIL_Gain_AdamW

RASAMO_Gain_MeanPool = CI(E2) - CI(E1)   # Does RA-SAMO help with MeanPool?
RASAMO_Gain_CLAM     = CI(E4) - CI(E3)   # Does RA-SAMO help with CLAM?
```

**Matched fold analysis:** Compute per-fold differences to reduce between-fold variance. Report paired differences with 95% CI (bootstrap or t-distribution with df=4).

**Statistical power caveat:** With 5 folds and expected effect sizes of 0.02–0.03 C-index against noise of 0.05–0.08 std, we have **low statistical power**. A positive observed interaction does NOT constitute statistical evidence. The experiment is exploratory — it can identify promising directions but cannot confirm significance.

### 6.4 KIRC extension

If BRCA results are promising (interaction > 0 on ≥4/5 folds), repeat for KIRC. Note KIRC uses `.h5` files with 1536-dim UNI2-h features — requires adjusting `CLAMImageEncoder` input dim.

---

## Summary of Findings

### Verified facts (from code)

1. MeanPool happens at data load time (`_load_embedding()`) — the model receives `[B, D]`, never patch-level data.
2. `SlideEmbeddingEncoder` is a single linear projection (1024→512), fully trainable.
3. "Attention" in the current model is cross-modal (image vs gene weighting), NOT patch-level.
4. RA-SAMO replaces encoder gradients via `prepare_branch_gradients()` → `apply_branch_gradients()`. It requires `model.image_encoder` and `model.gene_encoder` attributes.
5. All existing SAMO results use `JointModelWithAux` from `experiments/cvpr_reliability_pilot/models.py`.

### Reasonable inferences

1. `.pt` files contain `[N_patches, 1024]` tensors (code handles dim≥2 explicitly, synthetic test uses shape `[50, 1024]`).
2. CLAM attention parameters would receive RA-SAMO gradient projection without code changes, provided `model.image_encoder` wraps them.
3. Memory footprint is feasible on T4 (estimated ~350 MB peak).

### Unknowns requiring EC2 verification

1. Actual tensor shapes in BRCA `.pt` files.
2. Patch count distribution across 1,124 slides.
3. Whether any slides have 0 patches or only 1 patch.
4. Whether patch coordinates are available (not needed for this experiment, but relevant for spatial attention extensions).

---

## Required Code Changes

All new files — **no modifications to existing shared code:**

| File | Purpose | Lines (est.) |
|------|---------|:---:|
| `experiments/clam_brca/__init__.py` | Package init | 1 |
| `experiments/clam_brca/models.py` | GatedAttentionAggregator, CLAMImageEncoder, JointModelCLAM, JointModelCLAMWithAux | ~150 |
| `experiments/clam_brca/dataset.py` | CLAMSurvivalDataset, clam_collate_fn, build_clam_survival_dataloaders | ~200 |
| `experiments/clam_brca/train.py` | Training loop for 6 methods (E1–E4a) | ~300 |
| `experiments/clam_brca/test_clam.py` | Unit tests (shapes, attention, gradient flow, RA-SAMO compat) | ~250 |
| `experiments/clam_brca/analyze.py` | Results aggregation, comparison tables, figures | ~150 |
| `experiments/clam_brca/run_ec2.sh` | EC2 launch script with auto-shutdown | ~50 |
| `configs/brca_clam.yaml` | Experiment config | ~25 |

### Reused without modification

| Module | Import |
|--------|--------|
| `experiments/samo/objectives.py` | `cox_efron_free`, `ranking_loss`, `LossWeightController` |
| `experiments/samo/representation_gradients.py` | `prepare_branch_gradients`, `apply_branch_gradients` |
| `models/encoders.py` | `GeneEncoder` |
| `models/fusion.py` | `CrossModalAttention`, `FUSED_DIM`, `build_fusion` |
| `experiments/cvpr_reliability_pilot/models.py` | `AuxCoxHead` |
| `train_survival.py` | `concordance_index`, `build_survival_dataloaders` |

---

## Compute & Memory

| Resource | MeanPool (E1/E2) | CLAM (E3/E4) |
|----------|:---:|:---:|
| Batch size | 32 | 8 (×4 accum) |
| Peak GPU memory | ~200 MB | ~350 MB |
| Training time/epoch (est.) | ~7s | ~15s |
| Total per run (50 ep) | ~6 min | ~12 min |
| Full experiment (6×5 folds) | ~3 hours | ~6 hours |
| **Grand total** | **~9 hours** |

---

## Recommendation

### **CONDITIONAL GO**

**Prerequisites before proceeding:**

1. **Verify patch-level embeddings on EC2** — Run the shape-check command from §2.4. If files are already mean-pooled `[1024]`, this experiment is **NO-GO** (no patch information to aggregate).

2. **Confirm patch count distribution** — If median patches/slide < 10, attention MIL provides minimal benefit over mean-pool. If max patches > 10,000, capping at 4096 discards significant information.

3. **Accept exploratory scope** — With 5 folds and high variance (std ~0.06), this experiment can detect trends but cannot prove statistical significance. Results are hypothesis-generating.

**If prerequisites are met**, implementation order:
1. Create models, dataset, tests under `experiments/clam_brca/`
2. Run unit tests locally (CPU, synthetic data)
3. Start EC2, verify data
4. Smoke test: 2 epochs, E1 + E3, fold 0
5. Full run: 6 methods × 5 folds = 30 runs
6. Download, analyze, report
7. If interaction > 0, extend to KIRC
