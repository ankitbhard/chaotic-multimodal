"""
train_msam_survival.py — Unified M-SAM paper training script.

Supports 8 optimizer configs for the M-SAM paper (new runs only).
Reuses baselines from results_cindex_exp_b/ (sgd_mom, adam, cosine).

Optimizer configs (new GPU runs):
  cosine_ge      — OGM-GE only (no SAM, no chaotic)
  sam            — Vanilla SAM (all params, no modality awareness)
  msam           — Full M-SAM (cosine LR + Shapley + SAM + OGM-GE)
  msam_chaotic   — Full M-SAM with chaotic LR
  msam_no_ogm    — Ablation: Shapley + SAM, no OGM-GE
  msam_uniform   — Ablation: OGM-GE + SAM, random modality selection
  msam_shapley_gm — Ablation: Shapley + OGM-GE, no SAM perturbation

Features:
  - 5-fold stratified CV (default)
  - Skip logic: if result JSON exists with test_cindex, skip
  - Per-modality gradient norm logging
  - Shapley history for M-SAM configs
  - --rho and --alpha CLI args for sensitivity sweeps
  - Optional checkpoint saving for loss landscape visualization

Usage:
    # Main comparison (5-fold CV)
    python train_msam_survival.py \\
        --config configs/brca.yaml \\
        --optimizers cosine_ge sam msam msam_chaotic \\
        --save_dir results_msam_paper/brca \\
        --n_folds 5

    # Ablation
    python train_msam_survival.py \\
        --config configs/brca.yaml \\
        --optimizers msam_no_ogm msam_uniform \\
        --save_dir results_msam_paper/brca \\
        --n_folds 5

    # Sensitivity sweep
    python train_msam_survival.py \\
        --config configs/brca.yaml \\
        --optimizers msam \\
        --save_dir results_msam_paper/brca \\
        --n_folds 5 --fold 0 \\
        --rho 0.01
"""

import os, sys, json, random, argparse, time
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")

# ── reuse existing modules ──────────────────────────────────────────────────
try:
    from models.joint_model import build_model
    from optimizers.chaotic_optimizer_msam import MultimodalChaoticOptimizerMSAM
    from optimizers.chaotic_lr_scheduler import (
        ChaoticLRScheduler, CosineAnnealingScheduler,
    )
    from optimizers.sam_optimizer import SAMOptimizer
    from optimizers.msam_ablation import MSAMNoOGM, MSAMUniform, MSAMShapleyGMOnly
    from optimizers.ogm_wrapper import OGMWrapper
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
    from optimizers.sam_optimizer import SAMOptimizer
    from optimizers.msam_ablation import MSAMNoOGM, MSAMUniform, MSAMShapleyGMOnly
    from optimizers.ogm_wrapper import OGMWrapper
    from train_survival import (
        build_survival_dataloaders, cox_loss, concordance_index, evaluate,
    )


OPTIMIZER_NAMES = [
    "cosine_ge", "sam", "msam", "msam_chaotic",
    "msam_no_ogm", "msam_uniform", "msam_shapley_gm",
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.long, device=device)


def _to_float(x, device):
    if isinstance(x, torch.Tensor):
        return x.float().to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


def _compute_grad_norms(model):
    """Compute per-modality gradient L2 norms."""
    norms = {}
    for mod in ["image_encoder", "gene_encoder"]:
        total = 0.0
        for name, p in model.named_parameters():
            if mod in name and p.grad is not None:
                total += p.grad.norm().item() ** 2
        norms[mod] = total ** 0.5
    return norms


# ── optimizer factory ────────────────────────────────────────────────────────

def build_paper_optimizer(name, model, cfg, n_batches, rho=0.05, alpha=0.5):
    """
    Returns (optimizer, scheduler_or_None, opt_type).
    opt_type: 'standard' | 'msam' | 'sam' | 'ogm'

    For closure-based optimizers (msam/sam), scheduler is internal.
    """
    lr = cfg["lr"]
    wd = cfg["weight_decay"]
    epochs = cfg["epochs"]
    total_steps = epochs * n_batches

    if name == "cosine_ge":
        # OGM-GE with cosine LR (no SAM)
        base = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=wd)
        ogm = OGMWrapper(
            base, model,
            modality_names=["image", "gene"],
            alpha=alpha, use_ge=True,
        )
        sch = CosineAnnealingScheduler(base, T_max=epochs)
        return ogm, sch, "ogm"

    if name == "sam":
        # Vanilla SAM — perturbs ALL params
        base = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=wd)
        sam = SAMOptimizer(base, model, rho=rho)
        sam.scheduler = CosineAnnealingScheduler(base, T_max=total_steps)
        return sam, None, "sam"

    if name == "msam":
        # Full M-SAM with cosine LR
        base = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=wd)
        msam = MultimodalChaoticOptimizerMSAM(
            base, model,
            modality_names=["image", "gene"],
            base_lr=lr, r=3.99,
            rho=rho, alpha=alpha, use_ge=True, compute_every=1,
        )
        # Override scheduler with cosine (not chaotic)
        msam.scheduler = CosineAnnealingScheduler(base, T_max=total_steps)
        return msam, None, "msam"

    if name == "msam_chaotic":
        # Full M-SAM with chaotic LR
        base = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=wd)
        msam = MultimodalChaoticOptimizerMSAM(
            base, model,
            modality_names=["image", "gene"],
            base_lr=lr, r=3.99,
            rho=rho, alpha=alpha, use_ge=True, compute_every=1,
        )
        msam.scheduler = ChaoticLRScheduler(base, base_lr=lr, r=3.99,
                                            T_max=total_steps)
        return msam, None, "msam"

    if name == "msam_no_ogm":
        # Ablation: Shapley + SAM, no OGM-GE
        base = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=wd)
        msam = MSAMNoOGM(
            base, model,
            modality_names=["image", "gene"],
            base_lr=lr, r=3.99,
            rho=rho, alpha=alpha, use_ge=False, compute_every=1,
        )
        msam.scheduler = CosineAnnealingScheduler(base, T_max=total_steps)
        return msam, None, "msam"

    if name == "msam_uniform":
        # Ablation: OGM-GE + SAM, random modality
        base = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=wd)
        msam = MSAMUniform(
            base, model,
            modality_names=["image", "gene"],
            base_lr=lr, r=3.99,
            rho=rho, alpha=alpha, use_ge=True, compute_every=1,
        )
        msam.scheduler = CosineAnnealingScheduler(base, T_max=total_steps)
        return msam, None, "msam"

    if name == "msam_shapley_gm":
        # Ablation: Shapley + OGM-GE, NO SAM perturbation
        base = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=wd)
        msam = MSAMShapleyGMOnly(
            base, model,
            modality_names=["image", "gene"],
            base_lr=lr, r=3.99,
            rho=rho, alpha=alpha, use_ge=True, compute_every=1,
        )
        msam.scheduler = CosineAnnealingScheduler(base, T_max=total_steps)
        return msam, None, "msam"

    raise ValueError(f"Unknown optimizer: '{name}'. Choose from {OPTIMIZER_NAMES}")


# ── training loops ───────────────────────────────────────────────────────────

def train_one_epoch_closure(model, loader, opt, device, log_grads=False):
    """Train one epoch with closure-based optimizer (M-SAM or SAM)."""
    model.train()
    total_loss = 0.0
    grad_norms_epoch = []

    for inputs, labels in loader:
        img = inputs["image"].to(device)
        gene = inputs["gene"].to(device)
        os_m = _to_float(labels["os_months"], device)
        vs = _to_float(labels["vital_status"], device)

        inp = {"image": img, "gene": gene}
        labels_dev = {
            "survival": _to_tensor(labels["survival"], device),
            "os_months": os_m,
            "vital_status": vs,
        }

        def closure():
            out = model(inp)
            risk = out["survival"][:, 1]
            loss = cox_loss(risk, os_m, vs)
            return loss, out

        loss, _ = opt.step(closure, inp, labels_dev)
        total_loss += loss.item()

        if log_grads:
            grad_norms_epoch.append(_compute_grad_norms(model))

    avg_loss = total_loss / max(len(loader), 1)
    avg_grads = None
    if log_grads and grad_norms_epoch:
        avg_grads = {
            k: np.mean([g[k] for g in grad_norms_epoch])
            for k in grad_norms_epoch[0]
        }
    return avg_loss, avg_grads


def train_one_epoch_ogm(model, loader, ogm_opt, device, log_grads=False):
    """Train one epoch with OGM-GE optimizer (non-closure API)."""
    model.train()
    total_loss = 0.0
    grad_norms_epoch = []

    for inputs, labels in loader:
        img = inputs["image"].to(device)
        gene = inputs["gene"].to(device)
        os_m = _to_float(labels["os_months"], device)
        vs = _to_float(labels["vital_status"], device)

        inp = {"image": img, "gene": gene}
        labels_dev = {
            "survival": _to_tensor(labels["survival"], device),
            "os_months": os_m,
            "vital_status": vs,
        }

        outputs = model(inp)
        risk = outputs["survival"][:, 1]
        loss = cox_loss(risk, os_m, vs)

        ogm_opt.step(loss, inp, labels_dev)
        total_loss += loss.item()

        if log_grads:
            grad_norms_epoch.append(_compute_grad_norms(model))

    avg_loss = total_loss / max(len(loader), 1)
    avg_grads = None
    if log_grads and grad_norms_epoch:
        avg_grads = {
            k: np.mean([g[k] for g in grad_norms_epoch])
            for k in grad_norms_epoch[0]
        }
    return avg_loss, avg_grads


# ── main training function ───────────────────────────────────────────────────

def train_one_optimizer(opt_name, model_factory, loaders, cfg, device,
                        rho=0.05, alpha=0.5, save_checkpoints=False):
    """Train one optimizer config, return history dict."""
    model = model_factory()
    model.to(device)

    n_batches = len(loaders["train"])
    opt, scheduler, opt_type = build_paper_optimizer(
        opt_name, model, cfg, n_batches, rho=rho, alpha=alpha,
    )

    epochs = cfg["epochs"]
    save_dir = cfg["save_dir"]

    history = {
        "train_loss": [], "val_loss": [], "val_cindex": [],
        "grad_norms": [],
    }
    best_val_cindex = -1.0
    best_epoch = 0

    ckpt_dir = os.path.join(save_dir, "checkpoints") if save_checkpoints else None
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        log_grads = (epoch % 10 == 0 or epoch <= 5)

        if opt_type in ("msam", "sam"):
            tr_loss, grads = train_one_epoch_closure(
                model, loaders["train"], opt, device, log_grads=log_grads,
            )
        elif opt_type == "ogm":
            tr_loss, grads = train_one_epoch_ogm(
                model, loaders["train"], opt, device, log_grads=log_grads,
            )
        else:
            raise ValueError(f"Unknown opt_type: {opt_type}")

        val_loss, val_ci, _, _, _ = evaluate(model, loaders["val"], device)

        # Per-epoch scheduler for OGM configs
        if scheduler:
            scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_cindex"].append(val_ci)
        if grads:
            history["grad_norms"].append({"epoch": epoch, **grads})

        if val_ci > best_val_cindex:
            best_val_cindex = val_ci
            best_epoch = epoch
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f"{opt_name}_best.pt"))

        # Save checkpoints for loss landscape
        if save_checkpoints and epoch % 50 == 0:
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f"{opt_name}_ep{epoch}.pt"))

        if epoch % 20 == 0 or epoch == epochs:
            if hasattr(opt, 'get_current_lr'):
                lr_now = opt.get_current_lr()
            elif hasattr(opt, 'optimizer'):
                lr_now = opt.optimizer.param_groups[0]["lr"]
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
            "os_months": rec["os_months"],
            "vital_status": rec["vital_status"],
            "risk_score": float(risk_scores[i]),
        }
        for i, rec in enumerate(loaders["test_recs"])
    ]

    history["test_cindex"] = test_ci
    history["test_loss"] = test_loss
    history["best_val_cindex"] = best_val_cindex
    history["best_epoch"] = best_epoch
    history["test_predictions"] = test_preds

    # Save Shapley history for M-SAM configs
    if opt_type == "msam" and hasattr(opt, 'get_contribution_history'):
        shapley = opt.get_contribution_history()
        if shapley:
            history["shapley_history"] = shapley

    # Save rho/alpha if non-default (for sensitivity analysis)
    if rho != 0.05:
        history["rho"] = rho
    if alpha != 0.5:
        history["alpha"] = alpha

    out_path = os.path.join(save_dir, f"{opt_name}_survival.json")
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)

    # Clean up best checkpoint
    if os.path.exists(ckpt):
        os.remove(ckpt)

    return history


# ── run one fold ─────────────────────────────────────────────────────────────

def run_fold(cfg, opt_names, fold, n_folds, rho=0.05, alpha=0.5,
             save_checkpoints=False):
    """Run all optimizers on one CV fold."""
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders = build_survival_dataloaders(
        uni_dir=cfg["uni_dir"],
        gene_csv_paths=cfg["gene_csv_paths"],
        image_dim=cfg["image_dim"],
        n_top_genes=cfg.get("n_top_genes", 20000),
        val_fraction=cfg.get("val_fraction", 0.15),
        test_fraction=cfg.get("test_fraction", 0.10),
        batch_size=cfg.get("batch_size", 32),
        seed=seed,
        fold=fold,
        n_folds=n_folds,
    )

    n_genes = loaders["n_genes"]
    image_dim = loaders["image_dim"]

    def model_factory():
        return build_model(
            n_genes=n_genes,
            image_dim=image_dim,
            fusion_type=cfg.get("fusion_type", "attention"),
            dropout=cfg.get("dropout", 0.3),
        )

    save_dir = cfg.get("save_dir", "results_msam_paper")
    fold_dir = os.path.join(save_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    fold_cfg = dict(cfg)
    fold_cfg["save_dir"] = fold_dir

    results = {}
    for opt_name in opt_names:
        out_path = os.path.join(fold_dir, f"{opt_name}_survival.json")
        if os.path.exists(out_path):
            with open(out_path) as f:
                h = json.load(f)
            if "test_cindex" in h:
                print(f"  [SKIP] {opt_name} fold {fold} — "
                      f"test_cindex={h['test_cindex']:.4f}")
                results[opt_name] = h
                continue

        print(f"\n{'='*60}")
        print(f"  {opt_name} | fold {fold}/{n_folds}")
        print(f"{'='*60}")
        t0 = time.time()
        history = train_one_optimizer(
            opt_name, model_factory, loaders, fold_cfg, device,
            rho=rho, alpha=alpha, save_checkpoints=save_checkpoints,
        )
        elapsed = time.time() - t0
        print(f"  [{opt_name}] elapsed: {elapsed/60:.1f} min")
        results[opt_name] = history

    return results


# ── cross-validation ─────────────────────────────────────────────────────────

def main_cv(cfg, opt_names, n_folds, rho=0.05, alpha=0.5,
            single_fold=None, save_checkpoints=False):
    """Run n-fold CV (or a single fold if specified)."""
    base_save_dir = cfg.get("save_dir", "results_msam_paper")
    os.makedirs(base_save_dir, exist_ok=True)

    folds = [single_fold] if single_fold is not None else range(n_folds)

    all_results = {name: [] for name in opt_names}

    for fold in folds:
        print(f"\n{'#'*65}")
        print(f"  FOLD {fold}/{n_folds}")
        print(f"{'#'*65}")

        fold_cfg = dict(cfg)
        fold_cfg["save_dir"] = base_save_dir

        fold_results = run_fold(
            fold_cfg, opt_names, fold, n_folds,
            rho=rho, alpha=alpha, save_checkpoints=save_checkpoints,
        )
        for name, hist in fold_results.items():
            all_results[name].append(hist["test_cindex"])

    # Aggregate summary (only if all folds run)
    if single_fold is None:
        print(f"\n{'#'*65}")
        print(f"  {n_folds}-FOLD CROSS-VALIDATION SUMMARY")
        print(f"{'#'*65}")
        print(f"{'Optimizer':<20} {'Mean CI':>8} {'Std':>8}  Folds")
        print(f"{'-'*65}")
        for name in opt_names:
            cis = all_results[name]
            if not cis:
                continue
            mean = np.mean(cis)
            std = np.std(cis)
            fold_str = "  ".join(f"{c:.4f}" for c in cis)
            print(f"{name:<20} {mean:>8.4f} {std:>8.4f}  {fold_str}")
        print(f"{'='*65}")

        # Save aggregate JSON
        agg_path = os.path.join(base_save_dir, "cv_summary.json")
        agg = {
            name: {
                "fold_cindex": all_results[name],
                "mean_cindex": float(np.mean(all_results[name])),
                "std_cindex": float(np.std(all_results[name])),
            }
            for name in opt_names if all_results[name]
        }
        if rho != 0.05:
            agg["_rho"] = rho
        if alpha != 0.5:
            agg["_alpha"] = alpha
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"Saved CV summary: {agg_path}")


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="M-SAM paper unified training script (Cox survival)",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--optimizers", nargs="+", default=OPTIMIZER_NAMES)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None,
                        help="Run only this fold (for parallelization)")
    parser.add_argument("--rho", type=float, default=0.05,
                        help="SAM perturbation radius")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="OGM-GE suppression strength")
    parser.add_argument("--save_checkpoints", action="store_true",
                        help="Save epoch checkpoints (for loss landscape)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.save_dir is not None:
        cfg["save_dir"] = args.save_dir
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.seed is not None:
        cfg["seed"] = args.seed

    opt_names = OPTIMIZER_NAMES if "all" in args.optimizers else args.optimizers

    main_cv(
        cfg, opt_names,
        n_folds=args.n_folds,
        rho=args.rho,
        alpha=args.alpha,
        single_fold=args.fold,
        save_checkpoints=args.save_checkpoints,
    )
