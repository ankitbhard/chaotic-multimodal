"""
Unified training entrypoint for pan-cancer multimodal survival learning.

Usage:
    # Single optimizer (chaotic, default)
    python train.py --config configs/brca.yaml

    # Run all 12 optimizers
    python train.py --config configs/brca.yaml --optimizers all

    # Ablation: concat fusion instead of attention
    python train.py --config configs/brca.yaml --fusion concat

    # Override any config value on the command line
    python train.py --config configs/brca.yaml --epochs 50 --batch_size 16

Outputs (in config.save_dir):
    <opt_name>_history.json   — per-epoch metrics
    training.png              — multi-optimizer comparison plot

Optimizer groups (12 total for the paper):
    Baseline (no OGM-GE, no chaotic LR):
        sgd, sgd_mom, adam, adadelta, cosine
    Chaotic LR only (no OGM-GE):
        chaotic_no_ge
    OGM-GE only (no chaotic LR):
        sgd_ge, sgd_mom_ge, adam_ge, adadelta_ge, cosine_ge
    Full method (chaotic LR + OGM-GE):
        chaotic
    Additional variants:
        chaotic_adadelta, chaotic_sep_lr
"""

import os
import json
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import (
    f1_score, roc_auc_score,
    brier_score_loss, average_precision_score,
    confusion_matrix,
)

try:
    from models.joint_model               import build_model
    from optimizers.chaotic_optimizer     import MultimodalChaoticOptimizer
    from optimizers.chaotic_lr_scheduler  import CosineAnnealingScheduler
    from optimizers.ogm_wrapper           import OGMWrapper
    from data.dataset                     import build_dataloaders
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from models.joint_model               import build_model
    from optimizers.chaotic_optimizer     import MultimodalChaoticOptimizer
    from optimizers.chaotic_lr_scheduler  import CosineAnnealingScheduler
    from optimizers.ogm_wrapper           import OGMWrapper
    from data.dataset                     import build_dataloaders


# ── Loss ──────────────────────────────────────────────────────────────────────

class SurvivalLoss(nn.Module):
    """Single-task survival loss: w_surv * CrossEntropy(survival)."""
    def __init__(self, surv_w: float = 1.0):
        super().__init__()
        self.surv_w = surv_w

    def forward(self, outputs: dict, labels: dict, device) -> torch.Tensor:
        y = _to_tensor(labels["survival"], device)
        return self.surv_w * F.cross_entropy(outputs["survival"], y, ignore_index=-1)


def _to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device, dtype=torch.long)
    return torch.tensor(x, dtype=torch.long, device=device)


# ── Train / Eval loops ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, is_chaotic):
    model.train()
    total_loss, n = 0.0, 0
    preds_buf, probs_buf, labs_buf = [], [], []

    for inputs, labels in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        lbls   = {t: _to_tensor(v, device) for t, v in labels.items()}

        if is_chaotic:
            model.train()
            out  = model(inputs)
            loss = criterion(out, lbls, device)
            optimizer.step(loss, inputs, lbls)
        else:
            optimizer.zero_grad()
            out  = model(inputs)
            loss = criterion(out, lbls, device)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * lbls["survival"].size(0)

        with torch.no_grad():
            prob = torch.softmax(out["survival"], dim=1)[:, 1].cpu().numpy()
            p    = out["survival"].argmax(dim=1).cpu().numpy()
            l    = lbls["survival"].cpu().numpy()
            mask = l != -1
            preds_buf.extend(p[mask].tolist())
            probs_buf.extend(prob[mask].tolist())
            labs_buf.extend(l[mask].tolist())
        n += lbls["survival"].size(0)

    surv_acc = float(np.mean(np.array(preds_buf) == np.array(labs_buf))) if labs_buf else 0.0
    try:
        surv_auc = float(roc_auc_score(labs_buf, probs_buf)) if len(set(labs_buf)) > 1 else 0.5
    except Exception:
        surv_auc = 0.5
    return total_loss / n, surv_acc, surv_auc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    preds_buf, probs_buf, labs_buf = [], [], []

    for inputs, labels in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        lbls   = {t: _to_tensor(v, device) for t, v in labels.items()}

        out  = model(inputs)
        loss = criterion(out, lbls, device)
        total_loss += loss.item() * lbls["survival"].size(0)

        prob = torch.softmax(out["survival"], dim=1)[:, 1].cpu().numpy()
        p    = out["survival"].argmax(dim=1).cpu().numpy()
        l    = lbls["survival"].cpu().numpy()
        mask = l != -1
        preds_buf.extend(p[mask].tolist())
        probs_buf.extend(prob[mask].tolist())
        labs_buf.extend(l[mask].tolist())
        n += lbls["survival"].size(0)

    surv_acc = float(np.mean(np.array(preds_buf) == np.array(labs_buf))) if labs_buf else 0.0
    surv_f1  = float(f1_score(labs_buf, preds_buf, average="macro", zero_division=0)) \
               if labs_buf else 0.0
    try:
        surv_auc = float(roc_auc_score(labs_buf, probs_buf)) if len(set(labs_buf)) > 1 else 0.5
    except Exception:
        surv_auc = 0.5

    # Brier Score
    brier = float(brier_score_loss(np.array(labs_buf), np.array(probs_buf)))

    # Precision-Recall AUC
    try:
        pr_auc = float(average_precision_score(np.array(labs_buf), np.array(probs_buf))) \
                 if len(set(labs_buf)) > 1 else 0.0
    except Exception:
        pr_auc = 0.0

    # Sensitivity + Specificity
    preds_hard = (np.array(probs_buf) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        np.array(labs_buf), preds_hard, labels=[0, 1]
    ).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return total_loss / n, surv_acc, surv_f1, surv_auc, brier, pr_auc, sensitivity, specificity


# ── Optimizer factory ─────────────────────────────────────────────────────────

OPTIMIZER_NAMES = [
    # Baseline (no chaotic LR, no OGM-GE)
    "sgd", "sgd_mom", "adam", "adadelta", "cosine",
    # Chaotic LR only (no OGM-GE)
    "chaotic_no_ge",
    # OGM-GE only (no chaotic LR) — via OGMWrapper
    "sgd_ge", "sgd_mom_ge", "adam_ge", "adadelta_ge", "cosine_ge",
    # Full method: chaotic LR + OGM-GE
    "chaotic",
    # Additional variants
    "chaotic_adadelta", "chaotic_sep_lr",
]


def build_optimizer(name: str, model, cfg: dict):
    """
    Returns (optimizer, scheduler_or_None, is_chaotic).

    is_chaotic values:
        False      — standard optimizer (zero_grad/backward/step in training loop)
        True       — MultimodalChaoticOptimizer or OGMWrapper (step(loss, inputs, lbls))
    """
    lr = cfg["lr"]
    wd = cfg["weight_decay"]

    # ── Baseline optimizers ───────────────────────────────────────────────────

    if name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
        return opt, None, False

    if name == "sgd_mom":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=wd)
        return opt, None, False

    if name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        return opt, None, False

    if name == "adadelta":
        opt = torch.optim.Adadelta(model.parameters(), weight_decay=wd)
        return opt, None, False

    if name == "cosine":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=wd)
        sch = CosineAnnealingScheduler(opt, T_max=cfg["epochs"])
        return opt, sch, False

    # ── OGM-GE only (fixed LR) ────────────────────────────────────────────────

    if name == "sgd_ge":
        base = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
        wrapped = OGMWrapper(base, model, modality_names=["image", "gene"],
                             alpha=0.5, use_ge=True)
        return wrapped, None, True

    if name == "sgd_mom_ge":
        base = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=wd)
        wrapped = OGMWrapper(base, model, modality_names=["image", "gene"],
                             alpha=0.5, use_ge=True)
        return wrapped, None, True

    if name == "adam_ge":
        base = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        wrapped = OGMWrapper(base, model, modality_names=["image", "gene"],
                             alpha=0.5, use_ge=True)
        return wrapped, None, True

    if name == "adadelta_ge":
        base = torch.optim.Adadelta(model.parameters(), weight_decay=wd)
        wrapped = OGMWrapper(base, model, modality_names=["image", "gene"],
                             alpha=0.5, use_ge=True)
        return wrapped, None, True

    if name == "cosine_ge":
        base = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=wd)
        sch  = CosineAnnealingScheduler(base, T_max=cfg["epochs"])
        wrapped = OGMWrapper(base, model, modality_names=["image", "gene"],
                             alpha=0.5, use_ge=True)
        return wrapped, sch, True

    # ── Chaotic LR only (no OGM-GE) ──────────────────────────────────────────

    if name == "chaotic_no_ge":
        base = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=wd)
        chaotic = MultimodalChaoticOptimizer(
            base, model,
            modality_names=["image", "gene"],
            base_lr=lr,
            r=3.99, alpha=0.5, use_ge=False,
            compute_every=1,
        )
        return chaotic, None, True

    # ── Full method: chaotic LR + OGM-GE ─────────────────────────────────────

    if name == "chaotic":
        base = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=wd)
        chaotic = MultimodalChaoticOptimizer(
            base, model,
            modality_names=["image", "gene"],
            base_lr=lr,
            r=3.99, alpha=0.5, use_ge=True,
            compute_every=1,
        )
        return chaotic, None, True

    # ── Additional variants ───────────────────────────────────────────────────

    if name == "chaotic_adadelta":
        base = torch.optim.Adadelta(model.parameters(), lr=1.0,
                                     rho=0.9, eps=1e-6, weight_decay=wd)
        chaotic = MultimodalChaoticOptimizer(
            base, model,
            modality_names=["image", "gene"],
            base_lr=1.0,
            r=3.99, T_max=cfg["epochs"],
            alpha=0.5, use_ge=True,
            compute_every=1,
        )
        return chaotic, None, True

    if name == "chaotic_sep_lr":
        # Per-modality LRs: image encoder (thin projection) at 10× lower LR.
        image_lr_scale = cfg.get("image_lr_scale", 0.1)
        param_groups = [
            {"params": model.image_encoder.parameters(), "lr": lr * image_lr_scale},
            {"params": model.gene_encoder.parameters(),  "lr": lr},
            {"params": list(model.fusion.parameters()) +
                       list(model.shared.parameters()) +
                       list(model.head_survival.parameters()), "lr": lr},
        ]
        base = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=wd)
        chaotic = MultimodalChaoticOptimizer(
            base, model,
            modality_names=["image", "gene"],
            base_lr=lr,
            r=3.99, T_max=cfg["epochs"],
            alpha=0.5, use_ge=True,
            compute_every=1,
        )
        return chaotic, None, True

    raise ValueError(f"Unknown optimizer: '{name}'. "
                     f"Choose from {OPTIMIZER_NAMES}")


# ── Plotting ──────────────────────────────────────────────────────────────────

PALETTE = ["#2166ac", "#d6604d", "#4dac26", "#b2182b", "#762a83", "#f4a582"]


def plot_results(results: dict, cancer: str, save_dir: str):
    opt_keys  = list(results.keys())
    color_map = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(opt_keys)}
    n = len(opt_keys)

    fig, axes = plt.subplots(1, 4, figsize=(22, 4))

    # Val loss over epochs
    ax = axes[0]
    for opt, hist in results.items():
        ax.plot(hist["val_loss"], label=opt, color=color_map[opt])
    ax.set_title("Val Loss")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Val survival accuracy over epochs
    ax = axes[1]
    for opt, hist in results.items():
        ax.plot(hist["survival_acc"], label=opt, color=color_map[opt])
    ax.set_title("Val Survival Accuracy")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Val AUROC over epochs
    ax = axes[2]
    for opt, hist in results.items():
        ax.plot(hist["survival_auc"], label=opt, color=color_map[opt])
    ax.set_title("Val AUROC")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Final bar chart: test acc & auc
    ax = axes[3]
    x     = np.arange(2)
    width = 0.8 / n
    for i, (opt, hist) in enumerate(results.items()):
        vals   = [hist.get("test_survival_acc", hist["survival_acc"][-1]) * 100,
                  hist.get("test_survival_auc", hist["survival_auc"][-1]) * 100]
        offset = (i - n / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=opt,
                        color=color_map[opt], alpha=0.8)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{bar.get_height():.1f}",
                    ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(["Test Acc (%)", "Test AUC (%)"])
    ax.set_title("Test Results")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"{cancer.upper()} — Multi-Optimizer Comparison", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "training.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot : {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(cfg: dict, opt_names: list):
    # Seed
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Device
    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"Device     : {device}")
    print(f"Cancer     : {cfg['cancer'].upper()}")
    print(f"Optimizers : {opt_names}")
    print(f"Fusion     : {cfg['fusion_type']}")

    # Data
    loaders = build_dataloaders(
        uni_dir           = cfg["uni_dir"],
        gene_csv_paths    = cfg["gene_csv_paths"],
        subtype_csv_paths = cfg.get("subtype_csv_paths", []),
        image_dim         = cfg["image_dim"],
        n_top_genes       = cfg["n_top_genes"],
        val_fraction      = cfg["val_fraction"],
        test_fraction     = cfg["test_fraction"],
        batch_size        = cfg["batch_size"],
        seed              = seed,
    )

    n_genes = loaders["n_genes"]
    os.makedirs(cfg["save_dir"], exist_ok=True)
    results = {}

    for opt_name in opt_names:
        print(f"\n{'─'*60}")
        print(f"  Optimizer: {opt_name.upper()}")
        print(f"{'─'*60}")

        model = build_model(
            n_genes     = n_genes,
            image_dim   = cfg["image_dim"],
            fusion_type = cfg["fusion_type"],
            dropout     = cfg["dropout"],
            verbose     = (opt_names.index(opt_name) == 0),  # print once
        ).to(device)

        criterion = SurvivalLoss(surv_w=cfg["loss_weights"]["survival"])
        opt, scheduler, is_chaotic = build_optimizer(opt_name, model, cfg)

        history = {
            # Per-epoch training metrics
            "train_loss":     [],
            "train_surv_acc": [],
            "train_surv_auc": [],
            # Per-epoch validation metrics
            "val_loss":        [],
            "survival_acc":    [],
            "survival_f1":     [],
            "survival_auc":    [],
            "val_brier":       [],
            "val_pr_auc":      [],
            "val_sensitivity": [],
            "val_specificity": [],
        }

        best_val_auc = 0.0

        for epoch in range(cfg["epochs"]):
            train_loss, train_surv_acc, train_surv_auc = train_one_epoch(
                model, loaders["train"], opt, criterion, device, is_chaotic
            )
            val_loss, surv_acc, surv_f1, surv_auc, brier, pr_auc, sens, spec = evaluate(
                model, loaders["val"], criterion, device
            )

            # Cosine-SGD / cosine_ge: step external scheduler
            if scheduler is not None and not is_chaotic:
                scheduler.step()

            history["train_loss"].append(float(train_loss))
            history["train_surv_acc"].append(float(train_surv_acc))
            history["train_surv_auc"].append(float(train_surv_auc))
            history["val_loss"].append(float(val_loss))
            history["survival_acc"].append(float(surv_acc))
            history["survival_f1"].append(float(surv_f1))
            history["survival_auc"].append(float(surv_auc))
            history["val_brier"].append(float(brier))
            history["val_pr_auc"].append(float(pr_auc))
            history["val_sensitivity"].append(float(sens))
            history["val_specificity"].append(float(spec))

            if surv_auc > best_val_auc:
                best_val_auc = surv_auc

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"  ep {epoch+1:3d}/{cfg['epochs']}  "
                      f"loss={val_loss:.4f}  "
                      f"surv={surv_acc*100:.1f}%  "
                      f"f1={surv_f1:.3f}  "
                      f"auc={surv_auc:.3f}")

        # Final test evaluation
        _, test_surv_acc, test_surv_f1, test_surv_auc, \
            test_brier, test_pr_auc, test_sens, test_spec = evaluate(
            model, loaders["test"], criterion, device
        )
        history["test_survival_acc"]  = float(test_surv_acc)
        history["test_survival_f1"]   = float(test_surv_f1)
        history["test_survival_auc"]  = float(test_surv_auc)
        history["test_brier"]         = float(test_brier)
        history["test_pr_auc"]        = float(test_pr_auc)
        history["test_sensitivity"]   = float(test_sens)
        history["test_specificity"]   = float(test_spec)
        print(f"  [TEST] surv={test_surv_acc*100:.2f}%  f1={test_surv_f1:.3f}  "
              f"auc={test_surv_auc:.3f}  pr_auc={test_pr_auc:.3f}  "
              f"brier={test_brier:.3f}  sens={test_sens:.3f}  spec={test_spec:.3f}")

        # Save history
        hist_path = os.path.join(cfg["save_dir"], f"{opt_name}_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"  Saved : {hist_path}")

        results[opt_name] = history

    # Plot
    plot_results(results, cfg["cancer"], cfg["save_dir"])
    print(f"\nDone. Results in: {cfg['save_dir']}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multimodal cancer training (v2)"
    )
    parser.add_argument("--config",  required=True,
                        help="Path to YAML config (e.g. configs/brca.yaml)")

    # Optimizers
    parser.add_argument("--optimizers", nargs="+",
                        default=["chaotic"],
                        help=f"Optimizer(s) to run. Use 'all' for all 6. "
                             f"Choices: {OPTIMIZER_NAMES}")

    # Config overrides (all optional)
    parser.add_argument("--fusion",       type=str,   default=None,
                        help="Override fusion_type: 'attention' | 'concat'")
    parser.add_argument("--epochs",       type=int,   default=None)
    parser.add_argument("--batch_size",   type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--n_top_genes",  type=int,   default=None)
    parser.add_argument("--save_dir",     type=str,   default=None)
    parser.add_argument("--seed",         type=int,   default=None)

    args = parser.parse_args()

    # Load YAML
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply overrides
    if args.fusion      is not None: cfg["fusion_type"]  = args.fusion
    if args.epochs      is not None: cfg["epochs"]       = args.epochs
    if args.batch_size  is not None: cfg["batch_size"]   = args.batch_size
    if args.lr          is not None: cfg["lr"]           = args.lr
    if args.n_top_genes is not None: cfg["n_top_genes"]  = args.n_top_genes
    if args.save_dir    is not None: cfg["save_dir"]     = args.save_dir
    if args.seed        is not None: cfg["seed"]         = args.seed

    # Expand "all"
    opt_names = OPTIMIZER_NAMES if "all" in args.optimizers else args.optimizers

    main(cfg, opt_names)
