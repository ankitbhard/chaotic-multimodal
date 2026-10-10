"""
train_survival_msam_chaotic.py — M-SAM + Chaotic LR pilot experiment.

Cox PH survival training with 4 optimizer configs (2x2 design):

  | Config          | Chaotic LR | M-SAM | Description            |
  |-----------------|-----------|-------|------------------------|
  | cosine          | No        | No    | Baseline               |
  | chaotic         | No        | No    | Chaotic LR only        |
  | msam            | Yes       | Yes   | Full M-SAM + Chaotic   |
  | msam_no_chaotic | No        | Yes   | M-SAM without chaos    |

All: SGD+Momentum(0.9), lr=1e-3, weight_decay=1e-4, 200 epochs, seed=42.

For non-MSAM configs, scheduler is stepped per-epoch (T_max = epochs).
For MSAM configs, scheduler is stepped per-batch inside the optimizer
(T_max = epochs * batches_per_epoch, so decay spans the full training).

Usage:
    # Single split
    python train_survival_msam_chaotic.py \\
        --config configs/brca.yaml \\
        --optimizers cosine chaotic msam msam_no_chaotic \\
        --save_dir results_msam_pilot/brca \\
        --epochs 200 --seed 42

    # 3-fold cross-validation
    python train_survival_msam_chaotic.py \\
        --config configs/brca.yaml \\
        --optimizers cosine chaotic msam msam_no_chaotic \\
        --save_dir results_msam_cv/brca \\
        --epochs 200 --seed 42 --n_folds 3
"""

import os, sys, json, random, argparse
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── reuse existing modules ──────────────────────────────────────────────────
try:
    from models.joint_model import build_model
    from optimizers.chaotic_optimizer_msam import MultimodalChaoticOptimizerMSAM
    from optimizers.chaotic_lr_scheduler import (
        ChaoticLRScheduler, CosineAnnealingScheduler,
    )
    from train_survival import (
        build_survival_dataloaders, cox_loss, concordance_index, evaluate,
    )
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(__file__))
    from models.joint_model import build_model
    from optimizers.chaotic_optimizer_msam import MultimodalChaoticOptimizerMSAM
    from optimizers.chaotic_lr_scheduler import (
        ChaoticLRScheduler, CosineAnnealingScheduler,
    )
    from train_survival import (
        build_survival_dataloaders, cox_loss, concordance_index, evaluate,
    )


OPTIMIZER_NAMES = ["cosine", "chaotic", "msam", "msam_no_chaotic"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.long, device=device)


def _to_float(x, device):
    if isinstance(x, torch.Tensor):
        return x.float().to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


# ── optimizer factory ────────────────────────────────────────────────────────

def build_msam_optimizer(name, model, cfg, n_batches):
    """
    Returns (optimizer, scheduler_or_None, opt_type).
    opt_type: 'standard' | 'msam'

    For MSAM configs, scheduler is embedded in the optimizer (stepped per-batch),
    so the returned scheduler is None.
    """
    lr     = cfg["lr"]
    wd     = cfg["weight_decay"]
    epochs = cfg["epochs"]
    total_steps = epochs * n_batches  # for per-batch schedulers

    if name == "cosine":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=wd)
        sch = CosineAnnealingScheduler(opt, T_max=epochs)
        return opt, sch, "standard"

    if name == "chaotic":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=wd)
        sch = ChaoticLRScheduler(opt, base_lr=lr, r=3.99, T_max=epochs)
        return opt, sch, "standard"

    if name == "msam":
        base = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=wd)
        msam = MultimodalChaoticOptimizerMSAM(
            base, model,
            modality_names=["image", "gene"],
            base_lr=lr, r=3.99,
            rho=0.05, alpha=0.5, use_ge=True, compute_every=1,
        )
        # Replace default scheduler with one whose T_max spans full training
        msam.scheduler = ChaoticLRScheduler(
            base, base_lr=lr, r=3.99, T_max=total_steps,
        )
        return msam, None, "msam"

    if name == "msam_no_chaotic":
        base = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=wd)
        msam = MultimodalChaoticOptimizerMSAM(
            base, model,
            modality_names=["image", "gene"],
            base_lr=lr, r=3.99,
            rho=0.05, alpha=0.5, use_ge=True, compute_every=1,
        )
        # Replace chaotic scheduler with cosine annealing
        msam.scheduler = CosineAnnealingScheduler(base, T_max=total_steps)
        return msam, None, "msam"

    raise ValueError(f"Unknown optimizer: '{name}'. Choose from {OPTIMIZER_NAMES}")


# ── training loops ───────────────────────────────────────────────────────────

def train_one_epoch_msam(model, loader, msam_opt, device):
    """Train one epoch with M-SAM optimizer using Cox loss closure."""
    model.train()
    total_loss = 0.0

    for inputs, labels in loader:
        img  = inputs["image"].to(device)
        gene = inputs["gene"].to(device)
        os_m = _to_float(labels["os_months"], device)
        vs   = _to_float(labels["vital_status"], device)

        inp = {"image": img, "gene": gene}
        labels_dev = {
            "survival":     _to_tensor(labels["survival"], device),
            "os_months":    os_m,
            "vital_status": vs,
        }

        def closure():
            out  = model(inp)
            risk = out["survival"][:, 1]
            loss = cox_loss(risk, os_m, vs)
            return loss, out

        loss, _ = msam_opt.step(closure, inp, labels_dev)
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def train_one_epoch_standard(model, loader, optimizer, device):
    """Train one epoch with standard optimizer using Cox loss."""
    model.train()
    total_loss = 0.0

    for inputs, labels in loader:
        img  = inputs["image"].to(device)
        gene = inputs["gene"].to(device)
        os_m = _to_float(labels["os_months"], device)
        vs   = _to_float(labels["vital_status"], device)

        optimizer.zero_grad()
        outputs = model({"image": img, "gene": gene})
        risk    = outputs["survival"][:, 1]
        loss    = cox_loss(risk, os_m, vs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ── main training loop ───────────────────────────────────────────────────────

def train_one_optimizer(opt_name, model_factory, loaders, cfg, device):
    model = model_factory()
    model.to(device)

    n_batches = len(loaders["train"])
    opt, scheduler, opt_type = build_msam_optimizer(
        opt_name, model, cfg, n_batches,
    )

    epochs   = cfg["epochs"]
    save_dir = cfg["save_dir"]

    history = {"train_loss": [], "val_loss": [], "val_cindex": []}
    best_val_cindex = -1.0
    best_epoch      = 0

    for epoch in range(1, epochs + 1):
        if opt_type == "msam":
            tr_loss = train_one_epoch_msam(
                model, loaders["train"], opt, device,
            )
        else:
            tr_loss = train_one_epoch_standard(
                model, loaders["train"], opt, device,
            )

        val_loss, val_ci, _, _, _ = evaluate(model, loaders["val"], device)

        # Per-epoch scheduler for standard configs
        if scheduler:
            scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_cindex"].append(val_ci)

        if val_ci > best_val_cindex:
            best_val_cindex = val_ci
            best_epoch      = epoch
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f"{opt_name}_best.pt"))

        if epoch % 20 == 0 or epoch == epochs:
            if opt_type == "msam":
                lr_now = opt.get_current_lr()
            else:
                lr_now = opt.param_groups[0]["lr"]
            print(f"    ep {epoch:3d}/{epochs}  "
                  f"tr_loss={tr_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_ci={val_ci:.4f}  "
                  f"lr={lr_now:.6f}  "
                  f"(best={best_val_cindex:.4f} @ ep{best_epoch})")

    # ── test with best checkpoint ────────────────────────────────────────────
    ckpt = os.path.join(save_dir, f"{opt_name}_best.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=True))
    test_loss, test_ci, risk_scores, os_times, events = evaluate(
        model, loaders["test"], device,
    )

    print(f"  >> {opt_name}: test C-index = {test_ci:.4f}  "
          f"(val best = {best_val_cindex:.4f} @ ep{best_epoch})")

    # Per-patient predictions
    test_preds = [
        {
            "patient_barcode": rec["patient_barcode"],
            "os_months":       rec["os_months"],
            "vital_status":    rec["vital_status"],
            "risk_score":      float(risk_scores[i]),
        }
        for i, rec in enumerate(loaders["test_recs"])
    ]

    history["test_cindex"]      = test_ci
    history["test_loss"]        = test_loss
    history["best_val_cindex"]  = best_val_cindex
    history["best_epoch"]       = best_epoch
    history["test_predictions"] = test_preds

    # Save Shapley contribution history for MSAM configs
    if opt_type == "msam":
        history["shapley_history"] = opt.get_contribution_history()

    out_path = os.path.join(save_dir, f"{opt_name}_survival.json")
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)

    # Clean up checkpoint
    if os.path.exists(ckpt):
        os.remove(ckpt)

    return history


# ── plotting ─────────────────────────────────────────────────────────────────

PALETTE = {
    "cosine": "#2166ac", "chaotic": "#d6604d",
    "msam": "#1a9850", "msam_no_chaotic": "#762a83",
}


def plot_results(results, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Val C-index curves
    ax = axes[0]
    for name, hist in results.items():
        ax.plot(hist["val_cindex"], label=name,
                color=PALETTE.get(name, "#333"),
                linewidth=2 if "msam" in name else 1.5)
    ax.set_title("Validation C-index")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("C-index")
    ax.legend()
    ax.grid(alpha=0.3)

    # Train loss curves
    ax = axes[1]
    for name, hist in results.items():
        ax.plot(hist["train_loss"], label=name,
                color=PALETTE.get(name, "#333"),
                linewidth=2 if "msam" in name else 1.5)
    ax.set_title("Training Loss (Cox)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # Test C-index bar chart
    ax = axes[2]
    names   = list(results.keys())
    test_ci = [results[n]["test_cindex"] for n in names]
    colors  = [PALETTE.get(n, "#333") for n in names]
    bars    = ax.bar(names, test_ci, color=colors, alpha=0.8)
    for bar, ci in zip(bars, test_ci):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{ci:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Test C-index")
    ax.set_ylabel("C-index")
    ax.set_xticklabels(names, rotation=15)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("BRCA — M-SAM + Chaotic LR Pilot", fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, "msam_chaotic_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


# ── entry point (single split) ───────────────────────────────────────────────

def run_single(cfg, opt_names):
    """Run all optimizers on a single train/val/test split (or one CV fold)."""
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders = build_survival_dataloaders(
        uni_dir        = cfg["uni_dir"],
        gene_csv_paths = cfg["gene_csv_paths"],
        image_dim      = cfg["image_dim"],
        n_top_genes    = cfg.get("n_top_genes", 20000),
        val_fraction   = cfg.get("val_fraction", 0.15),
        test_fraction  = cfg.get("test_fraction", 0.10),
        batch_size     = cfg.get("batch_size", 32),
        seed           = seed,
        fold           = cfg.get("fold", None),
        n_folds        = cfg.get("n_folds", 5),
    )

    n_genes   = loaders["n_genes"]
    image_dim = loaders["image_dim"]

    def model_factory():
        return build_model(
            n_genes     = n_genes,
            image_dim   = image_dim,
            fusion_type = cfg.get("fusion_type", "attention"),
            dropout     = cfg.get("dropout", 0.3),
        )

    save_dir = cfg.get("save_dir", "results_msam_pilot")
    os.makedirs(save_dir, exist_ok=True)
    cfg["save_dir"] = save_dir

    results = {}
    for opt_name in opt_names:
        out_path = os.path.join(save_dir, f"{opt_name}_survival.json")
        if os.path.exists(out_path):
            with open(out_path) as f:
                h = json.load(f)
            if "test_cindex" in h:
                print(f"  [SKIP] {opt_name} — test_cindex={h['test_cindex']:.4f}")
                results[opt_name] = h
                continue

        print(f"\n{'='*50}")
        print(f"  {opt_name}")
        print(f"{'='*50}")
        history = train_one_optimizer(
            opt_name, model_factory, loaders, cfg, device,
        )
        results[opt_name] = history

    if len(results) > 1:
        plot_results(results, save_dir)

    # Summary table
    print(f"\n{'='*65}")
    print(f"{'Optimizer':<20} {'Test C-idx':>10} {'Best Val CI':>12} "
          f"{'Best Ep':>8} {'Gap':>8}")
    print(f"{'-'*65}")
    for name, hist in results.items():
        gap = hist["best_val_cindex"] - hist["test_cindex"]
        print(f"{name:<20} {hist['test_cindex']:>10.4f} "
              f"{hist['best_val_cindex']:>12.4f} "
              f"{hist['best_epoch']:>8d} {gap:>+8.4f}")
    print(f"{'='*65}")

    return results


# ── cross-validation entry point ─────────────────────────────────────────────

def main_cv(cfg, opt_names, n_folds):
    """Run n-fold cross-validation and aggregate results."""
    base_save_dir = cfg.get("save_dir", "results_msam_cv")

    # Collect per-fold test C-indices
    all_results = {name: [] for name in opt_names}

    for fold in range(n_folds):
        print(f"\n{'#'*65}")
        print(f"  FOLD {fold}/{n_folds}")
        print(f"{'#'*65}")

        fold_cfg = dict(cfg)
        fold_cfg["fold"]    = fold
        fold_cfg["n_folds"] = n_folds
        fold_cfg["save_dir"] = os.path.join(base_save_dir, f"fold{fold}")

        fold_results = run_single(fold_cfg, opt_names)
        for name, hist in fold_results.items():
            all_results[name].append(hist["test_cindex"])

    # Aggregate summary
    print(f"\n{'#'*65}")
    print(f"  {n_folds}-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'#'*65}")
    print(f"{'Optimizer':<20} {'Mean CI':>8} {'Std':>8}  Folds")
    print(f"{'-'*65}")
    for name in opt_names:
        cis  = all_results[name]
        mean = np.mean(cis)
        std  = np.std(cis)
        fold_str = "  ".join(f"{c:.4f}" for c in cis)
        print(f"{name:<20} {mean:>8.4f} {std:>8.4f}  {fold_str}")
    print(f"{'='*65}")

    # Save aggregate JSON
    agg_path = os.path.join(base_save_dir, "cv_summary.json")
    agg = {
        name: {
            "fold_cindex": all_results[name],
            "mean_cindex": float(np.mean(all_results[name])),
            "std_cindex":  float(np.std(all_results[name])),
        }
        for name in opt_names
    }
    os.makedirs(base_save_dir, exist_ok=True)
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Saved CV summary: {agg_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="M-SAM + Chaotic LR pilot experiment (Cox survival)",
    )
    parser.add_argument("--config",     required=True)
    parser.add_argument("--optimizers", nargs="+", default=OPTIMIZER_NAMES)
    parser.add_argument("--save_dir",   default=None)
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--seed",       type=int,   default=None)
    parser.add_argument("--n_folds",    type=int,   default=None,
                        help="Run n-fold CV (default: single split)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.save_dir is not None: cfg["save_dir"] = args.save_dir
    if args.epochs   is not None: cfg["epochs"]   = args.epochs
    if args.seed     is not None: cfg["seed"]     = args.seed

    opt_names = OPTIMIZER_NAMES if "all" in args.optimizers else args.optimizers

    if args.n_folds is not None:
        main_cv(cfg, opt_names, args.n_folds)
    else:
        run_single(cfg, opt_names)
