"""Controlled 2x3 experiment: MeanPool vs CLAM x {AdamW, aux+AdamW, RA-SAMO}.

Six methods:
  E1  meanpool_adamw        : JointModel + AdamW
  E2  meanpool_ra_samo      : JointModelWithAux + adaptive weights + projection
  E3  meanpool_aux_adamw    : JointModelWithAux + adaptive weights, no projection
  E4  clam_adamw            : JointModelCLAM + AdamW
  E5  clam_ra_samo          : JointModelCLAMWithAux + adaptive weights + projection
  E6  clam_aux_adamw        : JointModelCLAMWithAux + adaptive weights, no projection

E3/E6 are matched controls isolating the projection effect from auxiliary losses.

Usage:
    python -m experiments.clam_brca.train --config configs/brca_clam.yaml
    python -m experiments.clam_brca.train --config configs/brca_clam.yaml \\
        --methods clam_adamw clam_ra_samo --folds 1 --epochs 2
"""

import argparse
import copy
import csv
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models.joint_model import build_model
from experiments.cvpr_reliability_pilot.models import JointModelWithAux
from experiments.clam_brca.models import (
    JointModelCLAM, JointModelCLAMWithAux)
from experiments.clam_brca.dataset import (
    build_split, build_clam_loaders, build_meanpool_loaders)
from experiments.samo.objectives import (
    cox_efron_free, ranking_loss, LossWeightController)
from experiments.samo.representation_gradients import (
    prepare_branch_gradients, apply_branch_gradients)
from train_survival import concordance_index


# ---------------------------------------------------------------------------
# Method configuration
# ---------------------------------------------------------------------------

METHOD_CONFIG = {
    "meanpool_adamw": {
        "clam": False, "aux": False, "project": False, "adaptive": False,
    },
    "meanpool_ra_samo": {
        "clam": False, "aux": True, "project": True, "adaptive": True,
    },
    "meanpool_aux_adamw": {
        "clam": False, "aux": True, "project": False, "adaptive": True,
    },
    "clam_adamw": {
        "clam": True, "aux": False, "project": False, "adaptive": False,
    },
    "clam_ra_samo": {
        "clam": True, "aux": True, "project": True, "adaptive": True,
    },
    "clam_aux_adamw": {
        "clam": True, "aux": True, "project": False, "adaptive": True,
    },
}

ALL_METHODS = list(METHOD_CONFIG.keys())


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _copy_matching(src_model, dst_model):
    """Copy matching parameter tensors from src to dst."""
    src = src_model.state_dict()
    dst = dst_model.state_dict()
    matched = {k: v for k, v in src.items()
               if k in dst and dst[k].shape == v.shape}
    dst_model.load_state_dict(matched, strict=False)
    return len(matched)


def build_experiment_model(method, n_genes, image_dim, hidden_dim, dropout,
                           seed, fold):
    """Build model for a given method with matched initialization."""
    cfg = METHOD_CONFIG[method]

    # Always init baseline first for reproducible shared parameters
    seed_all(seed + fold)
    baseline = build_model(n_genes, image_dim, "attention", dropout,
                           verbose=False)

    if not cfg["clam"] and not cfg["aux"]:
        return baseline

    if not cfg["clam"] and cfg["aux"]:
        model = JointModelWithAux(n_genes, image_dim, dropout)
        _copy_matching(baseline, model)
        return model

    if cfg["clam"] and not cfg["aux"]:
        model = JointModelCLAM(n_genes, image_dim, hidden_dim, dropout)
        _copy_matching(baseline, model)
        return model

    # clam + aux
    model = JointModelCLAMWithAux(n_genes, image_dim, hidden_dim, dropout)
    _copy_matching(baseline, model)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device):
    """Compute validation/test loss and C-index."""
    model.eval()
    risks, times, events, losses = [], [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            x = {k: v.to(device) for k, v in inputs.items()}
            t = labels["os_months"].float().to(device)
            e = labels["vital_status"].float().to(device)
            out = model(x)
            risk = out["survival"][:, 1]
            losses.append(float(cox_efron_free(risk, t, e)))
            risks.extend(risk.cpu().tolist())
            times.extend(t.cpu().tolist())
            events.extend(e.cpu().tolist())
    ci = concordance_index(times, risks, events)
    return float(np.mean(losses)), float(ci)


def compute_attention_entropy(model):
    """Mean attention entropy across slides in the last forward pass."""
    enc = getattr(model, "image_encoder", None)
    if enc is None or not hasattr(enc, "get_attention_weights"):
        return float("nan")
    attns = enc.get_attention_weights()
    if attns is None:
        return float("nan")
    entropies = []
    for att in attns:
        att = att.clamp(min=1e-10)
        entropies.append(float(-(att * att.log()).sum()))
    return float(np.mean(entropies))


def compute_grad_norms(model):
    """Gradient L2 norms for image and gene encoders."""
    norms = {}
    for name in ("image_encoder", "gene_encoder"):
        enc = getattr(model, name, None)
        if enc is None:
            continue
        total = 0.0
        for p in enc.parameters():
            if p.grad is not None:
                total += p.grad.norm().item() ** 2
        norms[name] = total ** 0.5
    return norms


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(args, fold, method):
    cfg = METHOD_CONFIG[method]
    out_dir = Path(args.out) / f"fold_{fold}" / method
    summary_file = out_dir / "summary.json"
    if args.skip_existing and summary_file.exists():
        data = json.loads(summary_file.read_text())
        if data.get("completed"):
            print(f"SKIP {out_dir}", flush=True)
            return
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_all(args.seed + fold)

    # Build shared split (cached across methods for same fold)
    split_key = (args.seed, fold)
    if not hasattr(args, "_split_cache"):
        args._split_cache = {}
    if split_key not in args._split_cache:
        args._split_cache[split_key] = build_split(
            uni_dir=args.uni_dir,
            gene_csv_paths=args.gene_csv_paths,
            image_dim=args.image_dim,
            n_top_genes=args.n_top_genes,
            seed=args.seed, fold=fold, n_folds=args.n_folds)
    split = args._split_cache[split_key]

    # Build dataloaders — matched batch_size and max_patches for fair comparison
    if cfg["clam"]:
        loaders = build_clam_loaders(
            split, batch_size=args.batch_size,
            max_patches=args.max_patches)
    else:
        loaders = build_meanpool_loaders(
            split, batch_size=args.batch_size,
            max_patches=args.max_patches)

    # Build model
    model = build_experiment_model(
        method, split["n_genes"], args.image_dim,
        args.clam_hidden_dim, args.dropout, args.seed, fold)
    model = model.to(args.device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
    print(f"  {method}: {trainable_count:,} trainable params", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Weight controller (RA-SAMO & aux methods)
    controller = LossWeightController(
        initial=(args.aux_weight, args.aux_weight, args.rank_weight),
        min_weight=args.min_weight, max_weight=args.max_weight,
        warmup=args.warmup)
    weights = [args.aux_weight, args.aux_weight, args.rank_weight]

    # Reset peak memory counter
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(args.device)

    history = []
    best_ci = -float("inf")
    best_state = None
    best_epoch = 0
    start = time.perf_counter()

    for epoch in range(args.epochs):
        # Cosine LR
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))
        for group in optimizer.param_groups:
            group["lr"] = lr

        # --- Train ---
        model.train()
        sums = np.zeros(4)
        n_batches = 0
        diag_sums = {k: 0.0 for k in (
            "image_cosine_aux", "image_cosine_rank",
            "image_projected_aux", "image_projected_rank",
            "gene_cosine_aux", "gene_cosine_rank",
            "gene_projected_aux", "gene_projected_rank",
        )}
        diag_n = 0
        grad_norm_sums = {"image_encoder": 0.0, "gene_encoder": 0.0}
        entropy_sum = 0.0
        entropy_n = 0

        for inputs, labels in loaders["train"]:
            x = {k: v.to(args.device) for k, v in inputs.items()}
            t = labels["os_months"].float().to(args.device)
            e = labels["vital_status"].float().to(args.device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)

            # Loss decomposition
            risk = outputs["survival"][:, 1]
            fusion_loss = cox_efron_free(risk, t, e)
            zero = fusion_loss * 0.0

            if cfg["aux"]:
                img_loss = cox_efron_free(outputs["aux_img_risk"], t, e)
                gene_loss = cox_efron_free(outputs["aux_gene_risk"], t, e)
                rank_loss_val = ranking_loss(risk, t, e, args.temperature)
            else:
                img_loss = gene_loss = rank_loss_val = zero

            loss = (fusion_loss
                    + weights[0] * img_loss
                    + weights[1] * gene_loss
                    + weights[2] * rank_loss_val)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")

            # RA-SAMO gradient preparation
            prepared = None
            if cfg["project"]:
                prepared = prepare_branch_gradients(
                    model, fusion_loss, img_loss, gene_loss,
                    rank_loss_val, weights, project=True)

            loss.backward()

            if prepared is not None:
                diagnostics = apply_branch_gradients(prepared)
                diag_n += 1
                for branch, diag in diagnostics.items():
                    diag_sums[f"{branch}_cosine_aux"] += diag.cosine_aux
                    diag_sums[f"{branch}_cosine_rank"] += diag.cosine_rank
                    diag_sums[f"{branch}_projected_aux"] += float(
                        diag.projected_aux)
                    diag_sums[f"{branch}_projected_rank"] += float(
                        diag.projected_rank)

            # Record gradient norms before step
            gnorms = compute_grad_norms(model)
            for k, v in gnorms.items():
                grad_norm_sums[k] += v

            # Attention entropy (CLAM methods)
            if cfg["clam"]:
                ent = compute_attention_entropy(model)
                if np.isfinite(ent):
                    entropy_sum += ent
                    entropy_n += 1

            optimizer.step()
            sums += np.array([float(z.detach()) for z in
                              [fusion_loss, img_loss, gene_loss,
                               rank_loss_val]])
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("Empty training epoch")

        means = (sums / n_batches).tolist()

        # Update adaptive weights
        if cfg["adaptive"]:
            weights = list(controller.update(epoch, means))

        # --- Validate ---
        val_loss, val_ci = evaluate(model, loaders["val"], args.device)

        # Record epoch
        row = {
            "epoch": epoch + 1,
            "train_fusion_loss": means[0],
            "train_img_loss": means[1],
            "train_gene_loss": means[2],
            "train_rank_loss": means[3],
            "val_loss": val_loss,
            "val_ci": val_ci,
            "lr": lr,
            "weight_img": weights[0],
            "weight_gene": weights[1],
            "weight_rank": weights[2],
            "grad_norm_image": grad_norm_sums["image_encoder"] / n_batches,
            "grad_norm_gene": grad_norm_sums["gene_encoder"] / n_batches,
            "attention_entropy": (entropy_sum / entropy_n
                                  if entropy_n > 0 else ""),
            "elapsed_s": time.perf_counter() - start,
        }
        if diag_n > 0:
            for k, v in diag_sums.items():
                row[k] = v / diag_n

        if torch.cuda.is_available():
            row["peak_gpu_mb"] = (
                torch.cuda.max_memory_allocated(args.device) / 1024 ** 2)

        history.append(row)

        # Checkpoint on best val C-index
        if np.isfinite(val_ci) and val_ci > best_ci:
            best_ci = val_ci
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            print(f"  {method} fold={fold} ep={epoch+1}/{args.epochs} "
                  f"val_ci={val_ci:.4f} (best={best_ci:.4f}@{best_epoch})",
                  flush=True)

    # --- Test ---
    if best_state is None:
        raise RuntimeError("No finite validation C-index achieved")
    model.load_state_dict(best_state)
    model = model.to(args.device)
    test_loss, test_ci = evaluate(model, loaders["test"], args.device)

    wall_s = time.perf_counter() - start
    print(f"  {method} fold={fold} TEST ci={test_ci:.4f}  "
          f"({wall_s:.0f}s)", flush=True)

    # Free loaders/datasets to reclaim RAM before next method
    del loaders, model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save best checkpoint for post-hoc analysis (attention weights, etc.)
    ckpt_path = out_dir / "best_model.pt"
    torch.save(best_state, ckpt_path)
    print(f"  Checkpoint: {ckpt_path}", flush=True)

    # Save epoch history
    with (out_dir / "epochs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    # Save summary
    summary = {
        "completed": True,
        "method": method,
        "fold": fold,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_ci": best_ci,
        "test_ci": test_ci,
        "test_loss": test_loss,
        "wall_seconds": wall_s,
        "parameter_count": param_count,
        "trainable_count": trainable_count,
        "seed": args.seed + fold,
        "batch_size": args.batch_size,
        "max_patches": args.max_patches,
        "image_dim": args.image_dim,
        "n_genes": split["n_genes"],
    }
    if torch.cuda.is_available():
        summary["peak_gpu_mb"] = float(
            torch.cuda.max_memory_allocated(args.device) / 1024 ** 2)
    summary_file.write_text(json.dumps(summary, indent=2))

    del best_state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="CLAM Attention MIL 2x3 experiment")
    p.add_argument("--config", default="configs/brca_clam.yaml")
    p.add_argument("--methods", nargs="+", default=ALL_METHODS,
                   choices=ALL_METHODS)
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override from CLI
    args.uni_dir = cfg["uni_dir"]
    args.gene_csv_paths = cfg["gene_csv_paths"]
    args.image_dim = cfg.get("image_dim", 1024)
    args.n_top_genes = cfg.get("n_top_genes", 20000)
    args.dropout = cfg.get("dropout", 0.3)
    args.lr = cfg.get("lr", 1e-3)
    args.weight_decay = cfg.get("weight_decay", 1e-4)
    args.batch_size = cfg.get("batch_size", 32)
    args.max_patches = cfg.get("max_patches", 4096)
    args.clam_hidden_dim = cfg.get("clam_hidden_dim", 256)
    args.aux_weight = cfg.get("aux_weight", 0.1)
    args.rank_weight = cfg.get("rank_weight", 0.1)
    args.min_weight = cfg.get("min_weight", 0.02)
    args.max_weight = cfg.get("max_weight", 0.3)
    args.warmup = cfg.get("warmup", 2)
    args.temperature = cfg.get("temperature", 1.0)

    # n_folds = total CV splits (always from config); folds = how many to run
    args.n_folds = cfg.get("folds", 5)
    if args.folds is None:
        args.folds = args.n_folds
    if args.epochs is None:
        args.epochs = cfg.get("epochs", 50)
    if args.seed is None:
        args.seed = cfg.get("seed", 42)
    if args.out is None:
        args.out = cfg.get("save_dir", "results/clam_brca")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {args.device}")
    print(f"Methods: {args.methods}")
    print(f"Folds: {args.folds}/{args.n_folds}, Epochs: {args.epochs}")

    for fold in range(args.folds):
        for method in args.methods:
            print(f"\n{'='*60}")
            print(f"  {method}  fold={fold}")
            print(f"{'='*60}")
            run_one(args, fold, method)


if __name__ == "__main__":
    main()
