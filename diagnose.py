"""
Gradient Imbalance Diagnostic

Runs two quick checks before committing to a full 200-epoch training run:

  1. Gene heterogeneity  — inter-patient variance in RNA-seq (from CSV, instant)
  2. Gradient imbalance  — measures ||grad(image_encoder)|| vs ||grad(gene_encoder)||
     during N mini-batches of joint training (~2 min vs 15+ min for unimodal training)

Decision rule (calibrated on TCGA cancers):

  N_patients < 400    →  SGD        (small dataset, stability first)
  ratio > 5.0         →  Chaotic    (strong gradient imbalance — OGM-GE will rebalance)
  ratio < 2.0         →  Adadelta   (balanced gradients, adaptive rates sufficient)
  2.0 ≤ ratio ≤ 3.0  →  Adaptive   (borderline, let epoch-level selection decide)

Why gradient ratio works:
  OGM-GE suppresses whichever encoder has large gradients at each step.
  If gradients are already balanced (low ratio), OGM-GE modulates almost nothing
  and the chaotic LR adds noise with no benefit — Adadelta wins.
  If one encoder dominates (high ratio), OGM-GE actively rebalances — Chaotic wins.

Usage:
    python diagnose.py --config configs/gbm.yaml
    python diagnose.py --config configs/luad.yaml --n_batches 100
    python diagnose.py --config configs/brca.yaml --skip_train
"""

import argparse
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from models.joint_model import build_model
from data.dataset       import build_dataloaders


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device, dtype=torch.long)
    return torch.tensor(x, dtype=torch.long, device=device)


def _encoder_grad_norm(model, key: str) -> float:
    """L2 norm of gradients across all parameters whose name contains `key`."""
    sq = sum(
        p.grad.norm().item() ** 2
        for n, p in model.named_parameters()
        if key in n and p.grad is not None
    )
    return sq ** 0.5


# ── Check 1 — Gene heterogeneity (no model needed) ────────────────────────────

def compute_gene_heterogeneity(gene_csv_paths: list, n_top_genes: int) -> dict:
    """
    Measure inter-patient variance in RNA-seq profiles.
    High variance → patients are molecularly diverse → gene encoder carries
    unique information that the image may not capture.
    """
    frames = [pd.read_csv(p) for p in gene_csv_paths if os.path.exists(p)]
    df     = pd.concat(frames, ignore_index=True)

    meta   = {"sample_id", "label", "subtype_label", "patient_barcode",
               "survival_label", "patient_id"}
    gcols  = [c for c in df.columns
               if c not in meta and pd.api.types.is_numeric_dtype(df[c])]

    raw    = df[gcols].fillna(0).clip(lower=0).values.astype(np.float32)
    mat    = np.log1p(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0))
    variances = mat.var(axis=0)

    k     = min(n_top_genes, len(variances))
    top_v = np.sort(variances)[::-1][:k]

    return {
        "mean_var":   float(top_v.mean()),
        "median_var": float(np.median(top_v)),
        "cv":         float(top_v.std() / (top_v.mean() + 1e-8)),
        "n_patients": len(df),
    }


# ── Check 2 — Gradient imbalance (joint training, N mini-batches) ─────────────

def measure_gradient_imbalance(model, train_loader, device,
                                n_batches: int, lr: float) -> dict:
    """
    Run N mini-batches of joint training and measure the gradient magnitude ratio
    between image_encoder and gene_encoder at each step.

    High ratio → one encoder dominates the gradient signal →
    OGM-GE will actively modulate every step → Chaotic wins.

    Low ratio → gradients already balanced → OGM-GE does little → Adadelta wins.

    Unlike unimodal accuracy at 10 epochs, this metric is not biased by convergence
    speed — it directly measures what OGM-GE is designed to fix.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    img_norms, gene_norms, ratios = [], [], []
    loader_iter = iter(train_loader)

    for step in range(n_batches):
        try:
            inputs, labels = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            inputs, labels = next(loader_iter)

        inputs = {k: v.to(device) for k, v in inputs.items()}
        surv   = _to_tensor(labels["survival"], device)

        opt.zero_grad()
        out  = model(inputs)
        loss = F.cross_entropy(out["survival"], surv, ignore_index=-1)
        loss.backward()

        img_g  = _encoder_grad_norm(model, "image_encoder")
        gene_g = _encoder_grad_norm(model, "gene_encoder")

        if img_g > 0 and gene_g > 0:
            img_norms.append(img_g)
            gene_norms.append(gene_g)
            ratios.append(max(img_g, gene_g) / min(img_g, gene_g))

        opt.step()

        if (step + 1) % 10 == 0 or step == n_batches - 1:
            r = ratios[-1] if ratios else float("nan")
            print(f"      batch {step+1:3d}/{n_batches}"
                  f"  img_grad={img_g:.4f}  gene_grad={gene_g:.4f}"
                  f"  ratio={r:.2f}x")

    mean_img   = float(np.mean(img_norms))  if img_norms  else 0.0
    mean_gene  = float(np.mean(gene_norms)) if gene_norms else 0.0
    mean_ratio = float(np.mean(ratios))     if ratios     else 1.0
    dominant   = "image" if mean_img > mean_gene else "gene"

    return {
        "mean_img_norm":  mean_img,
        "mean_gene_norm": mean_gene,
        "mean_ratio":     mean_ratio,
        "dominant":       dominant,
    }


# ── Main diagnostic ───────────────────────────────────────────────────────────

def diagnose(cfg: dict, n_batches: int, skip_train: bool = False):
    cancer = cfg["cancer"].upper()

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))

    print(f"\n{'═'*62}")
    print(f"  Gradient Imbalance Diagnostic  ·  {cancer}")
    print(f"  Device : {device}  |  Diag batches : {n_batches}")
    print(f"{'═'*62}")

    # ── Check 1: gene heterogeneity (instant, no GPU needed) ──────────────────
    print("\n[1/3] Gene expression heterogeneity (from CSV)...")
    het = compute_gene_heterogeneity(cfg["gene_csv_paths"], cfg["n_top_genes"])

    print(f"  Patients            : {het['n_patients']}")
    print(f"  Mean gene variance  : {het['mean_var']:.4f}")
    print(f"  Median gene variance: {het['median_var']:.4f}")
    print(f"  Coeff of variation  : {het['cv']:.4f}")

    if het["mean_var"] > 0.8:
        het_label = "HIGH  — patients are molecularly diverse"
    elif het["mean_var"] > 0.4:
        het_label = "MEDIUM — moderate inter-patient diversity"
    else:
        het_label = "LOW   — relatively homogeneous cohort"
    print(f"  Heterogeneity       : {het_label}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[2/3] Loading data...")
    try:
        loaders = build_dataloaders(
            uni_dir           = cfg["uni_dir"],
            gene_csv_paths    = cfg["gene_csv_paths"],
            subtype_csv_paths = cfg.get("subtype_csv_paths", []),
            image_dim         = cfg["image_dim"],
            n_top_genes       = cfg["n_top_genes"],
            val_fraction      = cfg["val_fraction"],
            test_fraction     = cfg["test_fraction"],
            batch_size        = cfg["batch_size"],
            seed              = cfg["seed"],
        )
    except FileNotFoundError as e:
        print(f"\n  WARNING: Could not load slide embeddings — {e}")
        print("  Skipping gradient check. Only gene heterogeneity reported.")
        _print_partial_recommendation(cancer, het)
        return

    n_genes = loaders["n_genes"]
    n_total = loaders["train_size"] + loaders["val_size"] + loaders["test_size"]
    print(f"  Total patients : {n_total}  "
          f"(train={loaders['train_size']}  val={loaders['val_size']}  "
          f"test={loaders['test_size']})")

    if skip_train:
        print("\n  --skip_train set. Skipping gradient imbalance measurement.")
        _print_partial_recommendation(cancer, het, n_total=n_total)
        return

    # ── Check 2: gradient imbalance ───────────────────────────────────────────
    print(f"\n[3/3] Gradient imbalance  ({n_batches} mini-batches of joint training)...")
    model = build_model(
        n_genes=n_genes, image_dim=cfg["image_dim"],
        n_subtypes=cfg["n_subtypes"], fusion_type=cfg["fusion_type"],
        dropout=cfg["dropout"], verbose=False,
    ).to(device)

    grad = measure_gradient_imbalance(
        model, loaders["train"], device, n_batches=n_batches, lr=cfg["lr"],
    )

    # ── Results + recommendation ───────────────────────────────────────────────
    ratio    = grad["mean_ratio"]
    dominant = grad["dominant"]

    print(f"\n{'─'*62}")
    print(f"  RESULTS  ·  {cancer}")
    print(f"  Mean image_encoder grad norm : {grad['mean_img_norm']:.4f}")
    print(f"  Mean gene_encoder  grad norm : {grad['mean_gene_norm']:.4f}")
    print(f"  Mean gradient ratio          : {ratio:.2f}x  →  {dominant} dominates")
    print(f"  Gene heterogeneity           : {het_label}")
    print(f"  Dataset size                 : {n_total} patients")
    print(f"{'─'*62}")

    print(f"\n  RECOMMENDATION:")
    if n_total < 400:
        rec    = "SGD"
        reason = (f"Small dataset (N={n_total} < 400) — "
                  "chaotic LR amplifies label noise; stability wins")
    elif ratio > 5.0:
        rec    = "Chaotic + OGM-GE"
        reason = (f"Strong gradient imbalance (ratio={ratio:.2f}x, {dominant} dominates) — "
                  "OGM-GE will actively rebalance encoders every step")
    elif ratio < 2.0:
        rec    = "Adadelta"
        reason = (f"Balanced gradients (ratio={ratio:.2f}x) — "
                  "OGM-GE has little to modulate; adaptive rates are sufficient")
    else:
        rec    = "Run train_adaptive.py"
        reason = (f"Borderline imbalance (ratio={ratio:.2f}x) — "
                  "let epoch-level adaptive selection decide")

    print(f"  →  {rec}")
    print(f"     {reason}")
    print(f"{'═'*62}\n")

    return {
        "cancer":         cancer,
        "mean_img_norm":  grad["mean_img_norm"],
        "mean_gene_norm": grad["mean_gene_norm"],
        "ratio":          ratio,
        "dominant":       dominant,
        "n_total":        n_total,
        "het_mean_var":   het["mean_var"],
        "recommendation": rec,
    }


def _print_partial_recommendation(cancer, het, n_total=None):
    """Print recommendation based on gene heterogeneity alone."""
    print(f"\n{'─'*62}")
    print(f"  PARTIAL RESULTS  ·  {cancer}  (no slide features available locally)")
    if n_total:
        print(f"  Dataset size         : {n_total} patients")
    print(f"  Gene heterogeneity   : {het['mean_var']:.4f} mean variance")

    if n_total and n_total < 400:
        rec = "SGD  (small dataset)"
    elif het["mean_var"] > 0.8:
        rec = "Chaotic + OGM-GE  (high genomic heterogeneity — likely imbalanced)"
    elif het["mean_var"] < 0.4:
        rec = "Adadelta  (low heterogeneity — gene signal probably weak)"
    else:
        rec = "Run full diagnostic on EC2 where slide features are available"

    print(f"  Recommendation       : {rec}")
    print(f"{'═'*62}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gradient imbalance diagnostic — choose optimizer before training"
    )
    parser.add_argument("--config",     required=True,
                        help="Path to cancer YAML config (e.g. configs/gbm.yaml)")
    parser.add_argument("--n_batches",  type=int, default=50,
                        help="Mini-batches of joint training to sample (default 50)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Only compute gene heterogeneity, skip gradient check")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    diagnose(cfg, args.n_batches, args.skip_train)
