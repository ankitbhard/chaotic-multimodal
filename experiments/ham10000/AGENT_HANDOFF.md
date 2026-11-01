# HAM10000 Multimodal Classification — Execution Handoff

Branch: `experiment/ra-samo-ham10000-classification`
Source: `experiment/information-acquisition-diagnostics` (completed 27-run diagnostics).
Status: **Experiment specification only. Implementation, dataset download, smoke tests, GPU training, and benchmark results are NOT yet complete.**

## Objective
Evaluate whether existing RA-SAMO generalizes from image + RNA survival prediction to **dermoscopic image + tabular metadata** seven-class lesion classification. This is an exploratory CVPR 2027 workshop benchmark; do not promise a positive result.

## Dataset and leakage prevention
- HAM10000: 10,015 dermoscopy images; labels: akiec, bcc, bkl, df, mel, nv, vasc.
- Image branch: dermoscopic RGB image; tabular branch: age, sex, anatomical site only. Do not include dx, dx_type, image_id, lesion_id, or any diagnosis-related or post-diagnosis information as model inputs.
- Group by `lesion_id` when splitting to prevent duplicate-lesion leakage. If patient IDs exist and are reliable, additionally group by patient; otherwise disclose that patient-disjointness cannot be guaranteed.
- Produce a **fixed, reproducible train/validation/test split** (suggest 70/15/15) stratified at lesion-group level where feasible. Persist split manifest and per-class distributions; do not use test data for model or hyperparameter selection.
- Fit categorical vocabularies, age imputation, age scaling on training only. Encode missing metadata explicitly. Use augmentation on training images only. Verify dataset usage and licensing conditions.

## Model and methods
- Pretrained ResNet-18 image encoder (224x224), metadata MLP, late concatenation fusion MLP, seven-class logits. Record pretrained weights and preprocessing versions.
- Image-only AdamW control; metadata-only AdamW control.
- Multimodal AdamW; SAM; existing SAMO; existing RA-SAMO (use **specific previously benchmarked version**, record exact algorithm and hyperparameters); adaptive_aux.
- Main loss: multiclass CE. Aux loss: image-only and metadata-only CE, with same label mapping. If existing SAMO/RA-SAMO includes Cox-specific terms, adapt to CE explicitly and document every modification; do not silently claim algorithm identity.
- Equalize backbone, initialization, batch order, augmentation, LR schedule, training steps, and optimizer budget where feasible. SAM's two forward/backward passes must be reported in compute comparisons. Use same head capacity for fair comparisons and disclose extra aux parameters.

## Evaluation
- Primary: **macro-F1**; secondary: balanced accuracy, macro one-vs-rest AUROC (when class support permits), per-class precision/recall, accuracy, confusion matrix.
- Report paired results on **3 fixed seeds** for smoke/pilot; expand to 5 if justified. Report mean, standard deviation, individual seeds, runtime, peak GPU memory, convergence, and trainable parameter count.
- Evaluate held-out test **once per predeclared finalized method/configuration** after selecting using validation only. Do not choose a method by repeatedly inspecting test scores.
- Add modality ablations (image only, metadata only, shuffled metadata as a diagnostic). Quantify whether metadata contributes real signal.

## Implementation tasks
1. Inspect existing optimizer code in `experiments/samo/` and preserve the original gradient update rules. Create `experiments/ham10000/` with reproducible config, dataset/split scripts, model, training CLI, metrics, tests, and a launcher.
2. Provide a dataset download/preparation script with official source instructions; never commit images, credentials, or large pretrained checkpoints.
3. Build unit tests for group split leakage, class encoding, missing metadata, all methods' optimizer steps, gradient finite checks, checkpoint reload, and metrics.
4. Run a 1–2 epoch smoke test on a small sample for all methods; inspect logs, memory and correctness.
5. Then run pilot with 3 seeds, 30 epochs per method on EC2 T4, checkpoint/resume, record full config and code commit.
6. Create `results/ham10000_classification/final_report.md` and plots. Include negative findings and limitations, particularly that metadata is low-dimensional vs RNA-seq.

## Execution safety
- Verify storage and estimated compute budget before launching.
- Use persistent storage for data and outputs, commit scripts and lightweight summaries only.
- Do not start costly runs until tests pass.
- Stop (not terminate) EC2 only after outputs are safely saved and pushed.
- Report exact branch, commit, run count, best validation-selected methods, held-out test metrics, and EC2 state.
