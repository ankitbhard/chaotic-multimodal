#!/usr/bin/env python3
"""Phase 1: Rigorous optimizer validation for multimodal survival prediction.

Fixes critical audit issue C1: LR scheduling is now epoch-level cosine for ALL
methods. Adaptive optimizers modulate the *scheduled* LR, not a fixed reference.

Usage:
    # Pilot reproduction (30 epochs, fold 0)
    python -m experiments.phase1_optimizer_validation.run_phase1 \
        --config configs/brca.yaml --output results/phase1_optimizer_validation/pilot \
        --epochs 30 --folds 0

    # Full benchmark (200 epochs, all folds)
    python -m experiments.phase1_optimizer_validation.run_phase1 \
        --config configs/brca.yaml --output results/phase1_optimizer_validation/benchmark \
        --epochs 200 --folds 0 1 2 3 4

    # Single method
    python -m experiments.phase1_optimizer_validation.run_phase1 \
        --config configs/brca.yaml --output results/phase1_optimizer_validation/pilot \
        --epochs 30 --folds 0 --methods adamw
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from train_survival import build_survival_dataloaders, cox_loss, evaluate, concordance_index
from models.joint_model import build_model
from optimizers.sam_optimizer import SAMOptimizer
from optimizers.chaotic_optimizer_msam import MultimodalChaoticOptimizerMSAM


# ── Deterministic seeding ──────────────────────────────────────────────────

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Epoch-level cosine LR ─────────────────────────────────────────────────

def cosine_lr(base_lr, epoch, total_epochs, eta_min=0.0):
    """Compute cosine-annealed LR for a given epoch."""
    return eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * epoch / total_epochs)) / 2


# ── No-op scheduler (for SAM/M-SAM with externally managed LR) ───────────

class _NoOpScheduler:
    """Scheduler that does nothing. Used to override internal SAM/M-SAM schedulers
    when LR is managed externally at the epoch level."""
    def step(self):
        pass


# ── Adaptive LR controllers ───────────────────────────────────────────────

class LossEMAController:
    """Modulates a base LR by loss-EMA ratio. Applied on top of cosine schedule."""

    def __init__(self, min_scale=0.5, max_scale=1.5, ema_decay=0.9):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.ema_decay = ema_decay
        self._loss_ema = None
        self.history = []

    def get_scale(self, current_loss):
        val = float(current_loss)
        if self._loss_ema is None:
            self._loss_ema = val
        else:
            self._loss_ema = self.ema_decay * self._loss_ema + (1 - self.ema_decay) * val

        if abs(val) > 1e-12:
            ratio = self._loss_ema / val
            scale = max(self.min_scale, min(self.max_scale, ratio))
        else:
            scale = 1.0

        self.history.append({"loss": val, "loss_ema": self._loss_ema, "scale": scale})
        return scale


# ── Auxiliary heads for disentanglement ────────────────────────────────────

class AuxiliaryHeads(torch.nn.Module):
    """Lightweight unimodal Cox survival heads (training only)."""

    def __init__(self, image_feat_dim=512, gene_feat_dim=128):
        super().__init__()
        self.image_head = torch.nn.Sequential(
            torch.nn.Linear(image_feat_dim, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1))
        self.gene_head = torch.nn.Sequential(
            torch.nn.Linear(gene_feat_dim, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1))

    def forward(self, img_feat, gene_feat):
        return self.image_head(img_feat).squeeze(-1), self.gene_head(gene_feat).squeeze(-1)


# ── Training loop ──────────────────────────────────────────────────────────

def train_one_epoch_standard(model, loader, optimizer, device):
    """Standard training: zero_grad → forward → loss → backward → step."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for inputs, labels in loader:
        img = inputs["image"].to(device)
        gene = inputs["gene"].to(device)
        os_m = labels["os_months"].to(device).float()
        vs = labels["vital_status"].to(device).float()

        optimizer.zero_grad(set_to_none=True)
        out = model({"image": img, "gene": gene})
        risk = out["survival"][:, 1]
        loss = cox_loss(risk, os_m, vs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_one_epoch_disentangle(model, aux_heads, loader, optimizer, device,
                                  aux_weight=0.1, balance_grads=True):
    """Disentanglement training with auxiliary unimodal Cox heads."""
    model.train()
    aux_heads.train()
    total_loss = 0.0
    total_aux_img = 0.0
    total_aux_gene = 0.0
    img_norms = []
    gene_norms = []
    n_batches = 0

    img_params = list(model.image_encoder.parameters())
    gene_params = list(model.gene_encoder.parameters())

    for inputs, labels in loader:
        img = inputs["image"].to(device)
        gene = inputs["gene"].to(device)
        os_m = labels["os_months"].to(device).float()
        vs = labels["vital_status"].to(device).float()

        optimizer.zero_grad(set_to_none=True)

        # Forward with intermediate features
        img_feat = model.image_encoder(img)
        gene_feat = model.gene_encoder(gene)
        fused, _ = model.fusion(img_feat, gene_feat)
        shared = model.shared(fused)
        surv = model.head_survival(shared)
        risk = surv[:, 1]
        fusion_loss = cox_loss(risk, os_m, vs)

        # Auxiliary losses (no detach: aux gradients flow through encoders,
        # matching original gradient_disentanglement.py behavior)
        img_risk, gene_risk = aux_heads(img_feat, gene_feat)
        aux_img = cox_loss(img_risk, os_m, vs)
        aux_gene = cox_loss(gene_risk, os_m, vs)

        total = fusion_loss + aux_weight * (aux_img + aux_gene)
        total.backward()

        # Optional gradient balancing
        if balance_grads:
            ig = [p.grad for p in img_params if p.grad is not None]
            gg = [p.grad for p in gene_params if p.grad is not None]
            if ig and gg:
                img_norm = torch.linalg.vector_norm(
                    torch.stack([torch.linalg.vector_norm(g.detach().float()) for g in ig])
                ).item()
                gene_norm = torch.linalg.vector_norm(
                    torch.stack([torch.linalg.vector_norm(g.detach().float()) for g in gg])
                ).item()
                img_norms.append(img_norm)
                gene_norms.append(gene_norm)
                if img_norm > 1e-12 and gene_norm > 1e-12:
                    target = math.sqrt(img_norm * gene_norm)
                    with torch.no_grad():
                        for p in img_params:
                            if p.grad is not None:
                                p.grad.mul_(target / img_norm)
                        for p in gene_params:
                            if p.grad is not None:
                                p.grad.mul_(target / gene_norm)

        optimizer.step()

        total_loss += fusion_loss.item()
        total_aux_img += aux_img.item()
        total_aux_gene += aux_gene.item()
        n_batches += 1

    diag = {
        "fusion_loss": total_loss / max(n_batches, 1),
        "aux_img_loss": total_aux_img / max(n_batches, 1),
        "aux_gene_loss": total_aux_gene / max(n_batches, 1),
    }
    if img_norms:
        diag["mean_img_grad_norm"] = float(np.mean(img_norms))
        diag["mean_gene_grad_norm"] = float(np.mean(gene_norms))

    return total_loss / max(n_batches, 1), diag


def train_one_epoch_grad_balance(model, loader, optimizer, device):
    """Standard Cox loss + gradient norm balancing (no aux heads)."""
    model.train()
    total_loss = 0.0
    img_norms = []
    gene_norms = []
    n_batches = 0

    img_params = list(model.image_encoder.parameters())
    gene_params = list(model.gene_encoder.parameters())

    for inputs, labels in loader:
        img = inputs["image"].to(device)
        gene = inputs["gene"].to(device)
        os_m = labels["os_months"].to(device).float()
        vs = labels["vital_status"].to(device).float()

        optimizer.zero_grad(set_to_none=True)
        out = model({"image": img, "gene": gene})
        risk = out["survival"][:, 1]
        loss = cox_loss(risk, os_m, vs)
        loss.backward()

        # Gradient balancing: scale encoder grads to geometric mean
        ig = [p.grad for p in img_params if p.grad is not None]
        gg = [p.grad for p in gene_params if p.grad is not None]
        if ig and gg:
            img_norm = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(g.detach().float()) for g in ig])
            ).item()
            gene_norm = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(g.detach().float()) for g in gg])
            ).item()
            img_norms.append(img_norm)
            gene_norms.append(gene_norm)
            if img_norm > 1e-12 and gene_norm > 1e-12:
                target = math.sqrt(img_norm * gene_norm)
                with torch.no_grad():
                    for p in img_params:
                        if p.grad is not None:
                            p.grad.mul_(target / img_norm)
                    for p in gene_params:
                        if p.grad is not None:
                            p.grad.mul_(target / gene_norm)

        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    diag = {"fusion_loss": total_loss / max(n_batches, 1)}
    if img_norms:
        diag["mean_img_grad_norm"] = float(np.mean(img_norms))
        diag["mean_gene_grad_norm"] = float(np.mean(gene_norms))
    return total_loss / max(n_batches, 1), diag


def train_one_epoch_sam(model, loader, sam_optimizer, device):
    """SAM/M-SAM training using the existing legacy API."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for inputs, labels in loader:
        x = {k: v.to(device) for k, v in inputs.items()}
        os_m = labels["os_months"].to(device).float()
        events = labels["vital_status"].to(device).float()
        lab_dev = {"survival": labels["survival"].to(device),
                   "os_months": os_m, "vital_status": events}

        def closure():
            out = model(x)
            return cox_loss(out["survival"][:, 1], os_m, events), out

        loss, _ = sam_optimizer.step(closure, x, lab_dev)
        total_loss += float(loss.detach())
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ── Run one method ─────────────────────────────────────────────────────────

def run_method(name, model_factory, initial_state, loaders, cfg, device,
               outdir, fold, epochs, base_lr, weight_decay):
    """Run one optimizer for the configured epochs. Returns result dict."""
    seed = int(cfg.get("seed", 42)) + fold
    seed_all(seed)

    model = model_factory().to(device)
    model.load_state_dict(initial_state)

    # Build optimizer
    aux_heads = None
    ema_controller = None
    sam_opt = None
    is_sam = False

    use_cosine = True  # default: epoch-level cosine annealing
    grad_balance_only = False

    if name in ("adamw", "adamw_flat"):
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                       weight_decay=weight_decay)
        if name == "adamw_flat":
            use_cosine = False
    elif name in ("adaptive_loss_ema", "adaptive_flat"):
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                       weight_decay=weight_decay)
        ema_controller = LossEMAController(min_scale=0.5, max_scale=1.5, ema_decay=0.9)
        if name == "adaptive_flat":
            use_cosine = False
    elif name in ("disentangle_balance", "disentangle_aux_only"):
        img_dim = getattr(model.image_encoder, 'FEATURE_DIM', 512)
        gene_dim = getattr(model.gene_encoder, 'FEATURE_DIM', 128)
        aux_heads = AuxiliaryHeads(img_dim, gene_dim).to(device)
        all_params = list(model.parameters()) + list(aux_heads.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=base_lr,
                                       weight_decay=weight_decay)
    elif name == "disentangle_grad_only":
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                       weight_decay=weight_decay)
        grad_balance_only = True
    elif name in ("sam", "msam"):
        # Build SAM/M-SAM with AdamW base (matching other methods) and
        # NO internal scheduler (we apply epoch-level cosine externally)
        base_opt = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                      weight_decay=weight_decay)
        if name == "sam":
            sam_opt = SAMOptimizer(base_opt, model, rho=0.05)
        else:
            sam_opt = MultimodalChaoticOptimizerMSAM(
                base_opt, model,
                modality_names=["image", "gene"],
                base_lr=base_lr, r=3.99,
                rho=0.05, alpha=0.5, use_ge=True, compute_every=1,
            )
        sam_opt.scheduler = _NoOpScheduler()  # Override internal scheduler; LR set per epoch
        is_sam = True
        optimizer = None
    else:
        raise ValueError(f"Unknown method: {name}")

    history = []
    best_val_ci = -float("inf")
    best_state = None
    best_epoch = 0
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        # Compute scheduled LR
        if use_cosine:
            scheduled_lr = cosine_lr(base_lr, epoch - 1, epochs)
        else:
            scheduled_lr = base_lr  # flat LR

        effective_lr = scheduled_lr
        lr_scale = 1.0

        if ema_controller is not None and len(history) > 0:
            # Use previous epoch's mean train loss for EMA
            prev_loss = history[-1]["train_loss"]
            lr_scale = ema_controller.get_scale(prev_loss)
            effective_lr = scheduled_lr * lr_scale

        if is_sam:
            # Both SAMOptimizer and M-SAM delegate to .optimizer
            sam_pg = getattr(sam_opt, 'param_groups', None) or sam_opt.optimizer.param_groups
            for group in sam_pg:
                group["lr"] = effective_lr
        else:
            for group in optimizer.param_groups:
                group["lr"] = effective_lr

        # Train
        diag = {}
        if is_sam:
            train_loss = train_one_epoch_sam(model, loaders["train"], sam_opt, device)
        elif aux_heads is not None:
            balance = (name == "disentangle_balance")  # aux_only: no balancing
            train_loss, diag = train_one_epoch_disentangle(
                model, aux_heads, loaders["train"], optimizer, device,
                aux_weight=0.1, balance_grads=balance)
        elif grad_balance_only:
            train_loss, diag = train_one_epoch_grad_balance(
                model, loaders["train"], optimizer, device)
        else:
            train_loss = train_one_epoch_standard(model, loaders["train"],
                                                   optimizer, device)

        # Evaluate
        val_loss, val_ci, _, _, _ = evaluate(model, loaders["val"], device)
        epoch_time = time.perf_counter() - epoch_start

        # Record
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": float(val_loss),
            "val_cindex": float(val_ci),
            "epoch_seconds": epoch_time,
            "lr": effective_lr,
        }
        if ema_controller is not None:
            entry["lr_scale"] = lr_scale
            entry["scheduled_lr"] = scheduled_lr
        entry.update(diag)
        history.append(entry)

        # Checkpoint on best val C-index
        if np.isfinite(val_ci) and val_ci > best_val_ci:
            best_val_ci = float(val_ci)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        # NaN/Inf detection
        if not np.isfinite(train_loss):
            print(f"  WARNING: {name} fold={fold} epoch={epoch} "
                  f"train_loss={train_loss} — stopping early")
            break

        # Logging
        if epoch <= 3 or epoch % 10 == 0 or epoch == epochs:
            lr_str = f"lr={effective_lr:.6f}"
            print(f"  {name} fold={fold} ep={epoch:3d}/{epochs} "
                  f"train={train_loss:.4f} val_ci={val_ci:.4f} "
                  f"best={best_val_ci:.4f}@{best_epoch} {lr_str}",
                  flush=True)

    # Test evaluation with best model
    if best_state is None:
        raise RuntimeError(f"No valid val C-index for {name} fold={fold}")

    model.load_state_dict(best_state)
    test_loss, test_ci, test_risks, test_times, test_events = evaluate(
        model, loaders["test"], device)

    total_time = time.perf_counter() - start_time

    result = {
        "method": name,
        "fold": fold,
        "seed": int(cfg.get("seed", 42)) + fold,
        "epochs_run": len(history),
        "epochs_config": epochs,
        "best_val_cindex": best_val_ci,
        "best_epoch": best_epoch,
        "test_cindex": float(test_ci),
        "test_loss": float(test_loss),
        "total_seconds": total_time,
        "mean_epoch_seconds": total_time / max(len(history), 1),
        "history": history,
    }
    if ema_controller is not None:
        result["ema_controller_history"] = ema_controller.history

    # Save
    outpath = Path(outdir) / f"{name}_fold{fold}.json"
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)

    # Save checkpoint
    ckpt_path = Path(outdir) / f"{name}_fold{fold}_best.pt"
    torch.save(best_state, ckpt_path)

    return result


# ── Main ───────────────────────────────────────────────────────────────────

ALL_METHODS = [
    # Primary + Reference
    "adamw", "adaptive_loss_ema", "disentangle_balance", "sam", "msam",
    # D1 ablation: disentanglement components
    "disentangle_aux_only",   # aux heads, no gradient balancing
    "disentangle_grad_only",  # gradient balancing only, no aux heads
    # D2 ablation: adaptive LR components
    "adamw_flat",             # flat LR (no cosine)
    "adaptive_flat",          # flat LR + loss-EMA modulation
]


def main():
    p = argparse.ArgumentParser(description="Phase 1 optimizer validation")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--folds", nargs="+", type=int, default=[0])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--uni-dir", default=None)
    p.add_argument("--gene-csv", action="append", default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Path remapping
    if args.data_root:
        old = "/home/ubuntu/"
        if str(cfg["uni_dir"]).startswith(old):
            cfg["uni_dir"] = str(Path(args.data_root) / str(cfg["uni_dir"])[len(old):])
        cfg["gene_csv_paths"] = [
            str(Path(args.data_root) / str(v)[len(old):]) if str(v).startswith(old) else v
            for v in cfg["gene_csv_paths"]
        ]
    if args.uni_dir:
        cfg["uni_dir"] = args.uni_dir
    if args.gene_csv:
        cfg["gene_csv_paths"] = args.gene_csv

    methods = args.methods if args.methods else ALL_METHODS
    for m in methods:
        if m not in ALL_METHODS:
            p.error(f"Unknown method: {m}. Available: {ALL_METHODS}")

    base_lr = float(cfg["lr"])
    weight_decay = float(cfg.get("weight_decay", 0))

    outdir = Path(args.output).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Git info
    git_hash = "unknown"
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT)
        ).decode().strip()
    except Exception:
        pass

    print(f"Phase 1 Optimizer Validation")
    print(f"  Device: {device}")
    print(f"  Git: {git_hash[:12]}")
    print(f"  Methods: {methods}")
    print(f"  Folds: {args.folds}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Output: {outdir}")
    print()

    # Run per fold
    all_results = []

    for fold in args.folds:
        print(f"{'='*70}")
        print(f"FOLD {fold}")
        print(f"{'='*70}")

        # Build data
        base_seed = int(cfg.get("seed", 42))
        seed_all(base_seed + fold)
        loaders = build_survival_dataloaders(
            uni_dir=cfg["uni_dir"],
            gene_csv_paths=cfg["gene_csv_paths"],
            image_dim=cfg["image_dim"],
            n_top_genes=cfg.get("n_top_genes", 20000),
            batch_size=cfg.get("batch_size", 32),
            seed=base_seed,
            fold=fold,
            n_folds=args.n_folds,
        )

        # Build model prototype
        seed_all(base_seed + fold)
        prototype = build_model(
            n_genes=loaders["n_genes"],
            image_dim=loaders["image_dim"],
            fusion_type=cfg.get("fusion_type", "attention"),
            dropout=cfg.get("dropout", 0.3),
        )
        initial_state = copy.deepcopy(prototype.state_dict())
        digest = hashlib.sha256(
            b"".join(v.cpu().numpy().tobytes() for v in initial_state.values())
        ).hexdigest()
        print(f"  Initial weights: {digest[:16]}...")

        def model_factory():
            return build_model(
                n_genes=loaders["n_genes"],
                image_dim=loaders["image_dim"],
                fusion_type=cfg.get("fusion_type", "attention"),
                dropout=cfg.get("dropout", 0.3),
                verbose=False,
            )

        for method in methods:
            print(f"\n--- {method} (fold {fold}) ---")
            try:
                result = run_method(
                    method, model_factory, initial_state, loaders, cfg, device,
                    outdir, fold, args.epochs, base_lr, weight_decay)
                all_results.append(result)
                print(f"  DONE: test_ci={result['test_cindex']:.4f} "
                      f"best_val={result['best_val_cindex']:.4f}@ep{result['best_epoch']} "
                      f"({result['total_seconds']:.1f}s)")
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({"method": method, "fold": fold, "error": str(e)})

    # Save manifest
    manifest = {
        "git_hash": git_hash,
        "config": cfg,
        "args": vars(args),
        "device": str(device),
        "results": [{k: v for k, v in r.items() if k != "history"}
                     for r in all_results],
    }
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for r in sorted(all_results, key=lambda x: x.get("test_cindex", -1), reverse=True):
        if "error" in r:
            print(f"  {r['method']:25s} fold={r['fold']} ERROR: {r['error']}")
        else:
            print(f"  {r['method']:25s} fold={r['fold']} "
                  f"test_ci={r['test_cindex']:.4f} "
                  f"val_ci={r['best_val_cindex']:.4f}")

    print(f"\nResults: {outdir}")


if __name__ == "__main__":
    main()
