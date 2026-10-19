"""
CVPR Reliability Pilot — Training Script.

Unified training for all 6 methods:
  adamw, aux_cox, mod_dropout, learned_gating, pcrf, pcrf_aux

Usage:
    python -m experiments.cvpr_reliability_pilot.train \
        --config configs/brca.yaml \
        --methods adamw aux_cox \
        --folds 3 --epochs 100 \
        --save_dir results/cvpr_reliability_pilot/brca

    # Smoke test (5 epochs, fold 0 only)
    python -m experiments.cvpr_reliability_pilot.train \
        --config configs/brca.yaml \
        --methods adamw --folds 1 --epochs 5 --smoke
"""

import os, sys, json, math, time, copy, random, hashlib, argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import yaml

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from train_survival import (
    build_survival_dataloaders, concordance_index, cox_loss as existing_cox_loss,
)
from experiments.cvpr_reliability_pilot.models import (
    build_pilot_model, ALL_METHODS, BASELINE_METHODS, PHASE2_METHODS,
)
from experiments.cvpr_reliability_pilot.losses import combined_loss, cox_loss
from experiments.cvpr_reliability_pilot.reliability import PerturbationCalibrator


# ── LR schedule ──────────────────────────────────────────────────────────────

def cosine_lr(base_lr, epoch, total_epochs):
    """Epoch-level cosine annealing. Returns LR for given epoch (0-indexed)."""
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))


def set_lr(optimizer, lr):
    """Set learning rate for all parameter groups."""
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on a dataset. Returns (loss, c_index, risks, times, events)."""
    model.eval()
    all_risk, all_time, all_event = [], [], []
    total_loss = 0.0

    for inputs, labels in loader:
        img = inputs["image"].to(device)
        gene = inputs["gene"].to(device)
        os_m = labels["os_months"].float().to(device)
        vs = labels["vital_status"].float().to(device)

        outputs = model({"image": img, "gene": gene})
        risk = outputs["survival"][:, 1]
        loss = cox_loss(risk, os_m, vs)
        total_loss += loss.item()

        all_risk.extend(risk.cpu().numpy().tolist())
        all_time.extend(os_m.cpu().numpy().tolist())
        all_event.extend(vs.cpu().numpy().tolist())

    ci = concordance_index(all_time, all_risk, all_event)
    return total_loss / max(len(loader), 1), ci, all_risk, all_time, all_event


# ── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, method_cfg,
                    perturbation_sigma=0.1, n_perturbations=5,
                    reliability_weight=0.01, aux_weight=0.1):
    """
    Train one epoch for any method.

    Args:
        model: any pilot model
        loader: training DataLoader
        optimizer: AdamW
        device: cuda/cpu
        method_cfg: dict from build_pilot_model
        perturbation_sigma: sigma for reliability perturbation
        n_perturbations: MC samples for perturbation
        reliability_weight: gamma for reliability loss
        aux_weight: weight for aux Cox losses
    Returns:
        dict of average losses per component
    """
    model.train()
    accum = {"total": 0.0, "joint": 0.0, "aux_img": 0.0, "aux_gene": 0.0,
             "reliability": 0.0, "aux_img_detached": 0.0, "aux_gene_detached": 0.0}
    n_batches = 0

    for inputs, labels in loader:
        img = inputs["image"].to(device)
        gene = inputs["gene"].to(device)
        os_m = labels["os_months"].float().to(device)
        vs = labels["vital_status"].float().to(device)

        optimizer.zero_grad()

        outputs = model({"image": img, "gene": gene})

        total, loss_dict = combined_loss(
            outputs, os_m, vs,
            model=model,
            use_aux=method_cfg["use_aux_loss"],
            use_reliability=method_cfg["use_reliability_loss"],
            aux_weight=aux_weight,
            reliability_weight=reliability_weight,
            perturbation_sigma=perturbation_sigma,
            n_perturbations=n_perturbations,
        )

        total.backward()
        optimizer.step()

        for k, v in loss_dict.items():
            if k in accum:
                accum[k] += v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in accum.items()}


# ── Single method run ────────────────────────────────────────────────────────

def run_method(method, loaders, n_genes, image_dim, cfg, device,
               initial_state=None):
    """
    Train and evaluate one method on one fold.

    Args:
        method: method name string
        loaders: dict with train/val/test DataLoaders
        n_genes: number of gene features
        image_dim: image embedding dimension
        cfg: experiment config dict
        device: cuda/cpu
        initial_state: shared initial weights (for fair comparison)
    Returns:
        dict with full training history and test results
    """
    epochs = cfg["epochs"]
    base_lr = cfg.get("lr", 1e-3)
    weight_decay = cfg.get("weight_decay", 1e-4)
    reliability_weight = cfg.get("reliability_weight", 0.01)
    aux_weight = cfg.get("aux_weight", 0.1)
    perturbation_sigma = cfg.get("perturbation_sigma", 0.1)
    n_perturbations = cfg.get("n_perturbations", 5)

    # Build model
    model, method_cfg = build_pilot_model(method, n_genes, image_dim)
    model.to(device)

    # Apply shared initial weights for encoder + shared layers
    if initial_state is not None:
        _load_compatible_state(model, initial_state)

    # Optimizer: AdamW for ALL methods
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=weight_decay,
    )

    # Hash initial weights for reproducibility check
    weight_hash = _hash_params(model)
    print(f"    Initial weight hash: {weight_hash[:16]}")

    save_dir = cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    history = {
        "method": method,
        "method_cfg": method_cfg,
        "train_loss": [],
        "val_loss": [],
        "val_cindex": [],
        "lr": [],
        "epoch_losses": [],
    }
    best_val_ci = -1.0
    best_epoch = 0
    best_state = None

    t_start = time.time()

    warmup_epochs = method_cfg.get("warmup_epochs", 0)

    for epoch in range(epochs):
        # Cosine LR schedule
        lr = cosine_lr(base_lr, epoch, epochs)
        set_lr(optimizer, lr)

        # Curriculum reliability: disable reliability loss during warmup
        epoch_reliability_weight = reliability_weight
        if warmup_epochs > 0 and epoch < warmup_epochs:
            epoch_reliability_weight = 0.0

        # Train
        epoch_losses = train_one_epoch(
            model, loaders["train"], optimizer, device, method_cfg,
            perturbation_sigma=perturbation_sigma,
            n_perturbations=n_perturbations,
            reliability_weight=epoch_reliability_weight,
            aux_weight=aux_weight,
        )

        # Validate
        val_loss, val_ci, _, _, _ = evaluate(model, loaders["val"], device)

        history["train_loss"].append(epoch_losses["total"])
        history["val_loss"].append(val_loss)
        history["val_cindex"].append(val_ci)
        history["lr"].append(lr)
        history["epoch_losses"].append(epoch_losses)

        # Checkpoint on best val C-index
        if val_ci > best_val_ci:
            best_val_ci = val_ci
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            rel_info = ""
            if "reliability" in epoch_losses and epoch_losses["reliability"] > 0:
                rel_info = f"  rel={epoch_losses['reliability']:.4f}"
            warmup_tag = ""
            if warmup_epochs > 0 and epoch < warmup_epochs:
                warmup_tag = " [warmup]"
            elif warmup_epochs > 0 and epoch == warmup_epochs:
                warmup_tag = " [reliability ON]"
            print(
                f"    ep {epoch+1:3d}/{epochs}  "
                f"lr={lr:.6f}  "
                f"tr={epoch_losses['total']:.4f}  "
                f"val={val_loss:.4f}  ci={val_ci:.4f}  "
                f"(best={best_val_ci:.4f}@{best_epoch+1})"
                f"{rel_info}{warmup_tag}"
            )

    elapsed = time.time() - t_start

    # Load best checkpoint for test evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_ci, risks, times, events = evaluate(
        model, loaders["test"], device,
    )

    # Collect reliability weights on test set (for PCRF methods)
    test_reliability = None
    if hasattr(model, 'get_reliability_weights'):
        model.eval()
        rel_weights = []
        with torch.no_grad():
            for inputs, _ in loaders["test"]:
                img = inputs["image"].to(device)
                gene = inputs["gene"].to(device)
                model({"image": img, "gene": gene})
                w = model.get_reliability_weights()
                if w is not None:
                    rel_weights.append(w.cpu())
        if rel_weights:
            rel_weights = torch.cat(rel_weights, dim=0)
            test_reliability = {
                "alpha_img_mean": float(rel_weights[:, 0].mean()),
                "alpha_img_std": float(rel_weights[:, 0].std()),
                "alpha_gene_mean": float(rel_weights[:, 1].mean()),
                "alpha_gene_std": float(rel_weights[:, 1].std()),
            }

    history["test_cindex"] = test_ci
    history["test_loss"] = test_loss
    history["best_val_cindex"] = best_val_ci
    history["best_epoch"] = best_epoch + 1
    history["elapsed_seconds"] = elapsed
    history["weight_hash"] = weight_hash[:16]
    history["test_reliability"] = test_reliability
    history["warmup_epochs"] = warmup_epochs

    # Save per-patient predictions
    history["test_predictions"] = [
        {
            "patient_barcode": rec["patient_barcode"],
            "os_months": rec["os_months"],
            "vital_status": rec["vital_status"],
            "risk_score": float(risks[i]),
        }
        for i, rec in enumerate(loaders["test_recs"])
    ]

    print(f"  => {method}: test C-index = {test_ci:.4f}  "
          f"(best val {best_val_ci:.4f} @ ep {best_epoch+1})  "
          f"[{elapsed:.1f}s]")

    if test_reliability:
        print(f"     reliability weights: "
              f"img={test_reliability['alpha_img_mean']:.3f}±{test_reliability['alpha_img_std']:.3f}  "
              f"gene={test_reliability['alpha_gene_mean']:.3f}±{test_reliability['alpha_gene_std']:.3f}")

    # Save checkpoint for robustness evaluation
    ckpt_path = os.path.join(save_dir, f"{method}_best.pt")
    torch.save(best_state, ckpt_path)

    return history


# ── Helpers ──────────────────────────────────────────────────────────────────

def _hash_params(model):
    """SHA256 hash of model parameters for reproducibility verification."""
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.data.cpu().numpy().tobytes())
    return h.hexdigest()


def _load_compatible_state(model, state_dict):
    """
    Load compatible weights from a reference state dict.
    Only loads keys that exist in both model and state_dict with matching shapes.
    Used to ensure identical encoder initialization across methods.
    """
    model_state = model.state_dict()
    loaded = 0
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value.clone()
            loaded += 1
    model.load_state_dict(model_state)
    print(f"    Loaded {loaded}/{len(state_dict)} compatible weights")


def _build_initial_state(n_genes, image_dim, seed):
    """
    Build a reference model and extract its state dict for shared initialization.
    All methods start from identical encoder weights.
    """
    torch.manual_seed(seed)
    from models.joint_model import build_model
    ref_model = build_model(n_genes, image_dim, "attention", 0.3, verbose=False)
    return ref_model.state_dict()


# ── Main entry point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CVPR Reliability Pilot Training")
    parser.add_argument("--config", required=True, help="Cancer config YAML")
    parser.add_argument("--methods", nargs="+", default=["adamw"],
                        help="Methods to run")
    parser.add_argument("--folds", type=int, default=3, help="Number of CV folds")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 5 epochs, fold 0 only")
    parser.add_argument("--reliability_weight", type=float, default=0.01,
                        help="Gamma for reliability loss")
    parser.add_argument("--aux_weight", type=float, default=0.1,
                        help="Weight for auxiliary Cox losses")
    parser.add_argument("--perturbation_sigma", type=float, default=0.1,
                        help="Noise sigma for perturbation calibration")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip runs with existing results")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cancer = cfg.get("cancer", "unknown")

    if args.smoke:
        args.epochs = 5
        # StratifiedKFold needs n_splits >= 2; smoke test runs fold 0 of 3-fold
        if args.folds < 2:
            args.folds = 3
        print(f"=== SMOKE TEST: {cancer} (fold 0 only, {args.epochs} epochs) ===")
    else:
        print(f"=== CVPR Pilot: {cancer} | {args.folds} folds × {args.epochs} epochs ===")

    if args.save_dir:
        base_save = args.save_dir
    else:
        base_save = os.path.join(PROJECT_ROOT, "results", "cvpr_reliability_pilot", cancer)

    cfg["epochs"] = args.epochs
    cfg["seed"] = args.seed
    cfg["reliability_weight"] = args.reliability_weight
    cfg["aux_weight"] = args.aux_weight
    cfg["perturbation_sigma"] = args.perturbation_sigma

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
        print(f"GPU memory: {mem:.1f} GB")

    methods = args.methods
    if "all" in methods:
        methods = ALL_METHODS
    elif "phase2" in methods:
        methods = PHASE2_METHODS
    elif "baselines" in methods:
        methods = BASELINE_METHODS

    n_folds = args.folds
    all_results = {}

    # Manifest for this run
    manifest = {
        "cancer": cancer,
        "config_path": os.path.abspath(args.config),
        "methods": methods,
        "n_folds": n_folds,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
        "reliability_weight": args.reliability_weight,
        "aux_weight": args.aux_weight,
        "perturbation_sigma": args.perturbation_sigma,
    }

    run_folds = 1 if args.smoke else n_folds
    for fold in range(run_folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{n_folds}")
        print(f"{'='*60}")

        fold_seed = args.seed + fold
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)
        random.seed(fold_seed)

        # Build data loaders (same split for all methods in this fold)
        loaders = build_survival_dataloaders(
            uni_dir=cfg["uni_dir"],
            gene_csv_paths=cfg["gene_csv_paths"],
            image_dim=cfg["image_dim"],
            n_top_genes=cfg.get("n_top_genes", 20000),
            batch_size=cfg.get("batch_size", 32),
            seed=args.seed,  # data split seed is constant
            fold=fold,
            n_folds=n_folds,
        )

        # Fix: Rebuild training DataLoader with drop_last=True to avoid
        # BatchNorm failure on single-sample batches (KIRC: 289 = 9×32 + 1)
        from torch.utils.data import DataLoader
        train_ds = loaders["train"].dataset
        loaders["train"] = DataLoader(
            train_ds, batch_size=cfg.get("batch_size", 32),
            shuffle=True, num_workers=0, pin_memory=False, drop_last=True,
        )

        n_genes = loaders["n_genes"]
        image_dim = loaders["image_dim"]

        # Record dataset info
        if fold == 0:
            n_train = len(loaders["train"].dataset)
            n_val = len(loaders["val"].dataset)
            n_test = len(loaders["test"].dataset)
            manifest["n_genes_actual"] = n_genes
            manifest["n_genes_configured"] = cfg.get("n_top_genes", 20000)
            manifest["gene_count_discrepancy"] = (
                n_genes != cfg.get("n_top_genes", 20000)
            )
            manifest["image_dim"] = image_dim
            manifest["n_patients"] = {
                "train": n_train, "val": n_val, "test": n_test,
                "total": n_train + n_val + n_test,
            }
            events_train = sum(
                r["vital_status"] for r in loaders["train"].dataset.records
            )
            events_total = events_train + sum(
                r["vital_status"] for r in loaders["val"].dataset.records
            ) + sum(
                r["vital_status"] for r in loaders["test"].dataset.records
            )
            manifest["events"] = {
                "train": events_train,
                "total": events_total,
                "event_rate": events_total / (n_train + n_val + n_test),
            }
            manifest["image_backbone"] = (
                "UNI (1024)" if image_dim == 1024 else "UNI2-h (1536)"
            )
            print(f"  n_genes={n_genes} (configured={cfg.get('n_top_genes', 20000)})")
            print(f"  events: {events_total}/{n_train+n_val+n_test} "
                  f"({events_total/(n_train+n_val+n_test)*100:.1f}%)")

        # Shared initial encoder weights (identical across methods)
        torch.manual_seed(fold_seed)
        initial_state = _build_initial_state(n_genes, image_dim, fold_seed)

        for method in methods:
            fold_save = os.path.join(base_save, f"fold_{fold}")
            cfg["save_dir"] = fold_save

            out_path = os.path.join(fold_save, f"{method}.json")
            if args.skip_existing and os.path.exists(out_path):
                with open(out_path) as f:
                    existing = json.load(f)
                if "test_cindex" in existing:
                    print(f"\n  [SKIP] {method} fold {fold} — "
                          f"test_ci={existing['test_cindex']:.4f}")
                    if method not in all_results:
                        all_results[method] = {}
                    all_results[method][fold] = existing
                    continue

            print(f"\n── {method} (fold {fold}) ──")

            # Reset seed for model init
            torch.manual_seed(fold_seed)
            np.random.seed(fold_seed)
            random.seed(fold_seed)

            history = run_method(
                method, loaders, n_genes, image_dim, cfg, device,
                initial_state=initial_state,
            )

            # Save result
            os.makedirs(fold_save, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(history, f, indent=2)

            if method not in all_results:
                all_results[method] = {}
            all_results[method][fold] = history

    # ── Summary ───────────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"SUMMARY — {cancer}")
    print(f"{'='*60}")

    summary = {}
    for method in methods:
        if method in all_results:
            test_cis = [
                all_results[method][f]["test_cindex"]
                for f in range(n_folds)
                if f in all_results[method]
            ]
            val_cis = [
                all_results[method][f]["best_val_cindex"]
                for f in range(n_folds)
                if f in all_results[method]
            ]
            if test_cis:
                mean_test = np.mean(test_cis)
                std_test = np.std(test_cis)
                mean_val = np.mean(val_cis)
                print(f"  {method:18s}: test={mean_test:.4f}±{std_test:.4f}  "
                      f"val={mean_val:.4f}  "
                      f"folds={[f'{c:.4f}' for c in test_cis]}")
                summary[method] = {
                    "test_cindex_mean": mean_test,
                    "test_cindex_std": std_test,
                    "val_cindex_mean": mean_val,
                    "test_cindex_per_fold": test_cis,
                    "val_cindex_per_fold": val_cis,
                }

    # Save manifest and summary
    os.makedirs(base_save, exist_ok=True)
    manifest["summary"] = summary
    manifest_path = os.path.join(base_save, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
