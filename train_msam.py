"""
Training entrypoint with M-SAM (Modality-Aware SAM) optimizer.

Extends train.py with a 7th optimizer: 'msam'
    - Chaotic LR scheduling (logistic map)
    - Shapley-based modality contribution estimation
    - SAM perturbation on dominant modality encoder
    - OGM-GE gradient modulation

Results saved to: <save_dir>/msam/ (separate from main results)

Usage:
    # Run only msam optimizer
    python train_msam.py --config configs/brca.yaml --optimizers msam

    # Compare msam vs chaotic (original)
    python train_msam.py --config configs/brca.yaml --optimizers chaotic msam

    # All 7 optimizers
    python train_msam.py --config configs/brca.yaml --optimizers all

    # Test on fewer epochs first
    python train_msam.py --config configs/kirc.yaml --optimizers msam --epochs 50
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
from sklearn.metrics import f1_score

try:
    from models.joint_model import build_model
    from optimizers.chaotic_optimizer      import MultimodalChaoticOptimizer
    from optimizers.chaotic_optimizer_msam import MultimodalChaoticOptimizerMSAM
    from optimizers.chaotic_lr_scheduler   import CosineAnnealingScheduler
    from data.dataset import build_dataloaders
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from models.joint_model import build_model
    from optimizers.chaotic_optimizer      import MultimodalChaoticOptimizer
    from optimizers.chaotic_optimizer_msam import MultimodalChaoticOptimizerMSAM
    from optimizers.chaotic_lr_scheduler   import CosineAnnealingScheduler
    from data.dataset import build_dataloaders


# ── Loss ──────────────────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    def __init__(self, surv_w: float = 1.0, sub_w: float = 1.5):
        super().__init__()
        self.surv_w = surv_w
        self.sub_w  = sub_w

    def forward(self, outputs: dict, labels: dict, device) -> torch.Tensor:
        loss = torch.tensor(0.0, device=device)
        if "survival" in outputs and "survival" in labels:
            y = _to_tensor(labels["survival"], device)
            loss = loss + self.surv_w * F.cross_entropy(outputs["survival"], y,
                                                         ignore_index=-1)
        if "subtype" in outputs and "subtype" in labels:
            y = _to_tensor(labels["subtype"], device)
            loss = loss + self.sub_w * F.cross_entropy(outputs["subtype"], y,
                                                        ignore_index=-1)
        return loss


def _to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device, dtype=torch.long)
    return torch.tensor(x, dtype=torch.long, device=device)


# ── Train / Eval loops ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, opt_type):
    """
    opt_type: 'standard' | 'chaotic' | 'msam'
    """
    model.train()
    total_loss, n = 0.0, 0
    preds_buf = {"survival": [], "subtype": []}
    labs_buf  = {"survival": [], "subtype": []}

    for inputs, labels in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        lbls   = {t: _to_tensor(v, device) for t, v in labels.items()}

        if opt_type == "msam":
            # M-SAM requires closure (called twice internally)
            def closure():
                out  = model(inputs)
                loss = criterion(out, lbls, device)
                return loss, out

            loss, out = optimizer.step(closure, inputs, lbls)

        elif opt_type == "chaotic":
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
            for task in ("survival", "subtype"):
                p    = out[task].argmax(dim=1).cpu().numpy()
                l    = lbls[task].cpu().numpy()
                mask = l != -1
                preds_buf[task].extend(p[mask])
                labs_buf[task].extend(l[mask])
        n += lbls["survival"].size(0)

    accs = {t: float(np.mean(np.array(preds_buf[t]) == np.array(labs_buf[t])))
            if labs_buf[t] else 0.0
            for t in ("survival", "subtype")}
    return total_loss / n, accs


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    preds_buf = {"survival": [], "subtype": []}
    labs_buf  = {"survival": [], "subtype": []}

    for inputs, labels in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        lbls   = {t: _to_tensor(v, device) for t, v in labels.items()}

        out  = model(inputs)
        loss = criterion(out, lbls, device)
        total_loss += loss.item() * lbls["survival"].size(0)

        for task in ("survival", "subtype"):
            p    = out[task].argmax(dim=1).cpu().numpy()
            l    = lbls[task].cpu().numpy()
            mask = l != -1
            preds_buf[task].extend(p[mask])
            labs_buf[task].extend(l[mask])
        n += lbls["survival"].size(0)

    accs = {t: float(np.mean(np.array(preds_buf[t]) == np.array(labs_buf[t])))
            if labs_buf[t] else 0.0
            for t in ("survival", "subtype")}
    f1s  = {t: float(f1_score(labs_buf[t], preds_buf[t],
                               average="macro", zero_division=0))
            if labs_buf[t] else 0.0
            for t in ("survival", "subtype")}
    return total_loss / n, accs, f1s


# ── Optimizer factory ─────────────────────────────────────────────────────────

OPTIMIZER_NAMES = ["sgd", "sgd_mom", "adam", "adadelta", "cosine", "chaotic", "msam"]


def build_optimizer(name: str, model, cfg: dict):
    """
    Returns (optimizer, scheduler_or_None, opt_type).
    opt_type: 'standard' | 'chaotic' | 'msam'
    """
    lr = cfg["lr"]
    wd = cfg["weight_decay"]

    if name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
        return opt, None, "standard"

    if name == "sgd_mom":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=wd)
        return opt, None, "standard"

    if name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        return opt, None, "standard"

    if name == "adadelta":
        opt = torch.optim.Adadelta(model.parameters(), weight_decay=wd)
        return opt, None, "standard"

    if name == "cosine":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=wd)
        sch = CosineAnnealingScheduler(opt, T_max=cfg["epochs"])
        return opt, sch, "standard"

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
        return chaotic, None, "chaotic"

    if name == "msam":
        base = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=wd)
        msam = MultimodalChaoticOptimizerMSAM(
            base, model,
            modality_names=["image", "gene"],
            base_lr=lr,
            r=3.99,
            rho=0.05,
            alpha=0.5,
            use_ge=True,
            compute_every=1,
        )
        return msam, None, "msam"

    raise ValueError(f"Unknown optimizer: '{name}'. Choose from {OPTIMIZER_NAMES}")


# ── Plotting ──────────────────────────────────────────────────────────────────

PALETTE = ["#2166ac", "#d6604d", "#4dac26", "#b2182b", "#762a83", "#f4a582", "#1a9850"]


def plot_results(results: dict, cancer: str, save_dir: str):
    opt_keys  = list(results.keys())
    color_map = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(opt_keys)}
    n = len(opt_keys)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    for ax, metric, label in zip(
        axes[:2],
        ["survival_acc", "subtype_f1"],
        ["Survival Accuracy", "Subtype F1 (macro)"],
    ):
        for opt, hist in results.items():
            ax.plot(hist[metric], label=opt, color=color_map[opt],
                    linewidth=2 if opt == "msam" else 1)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    ax = axes[2]
    x     = np.arange(2)
    width = 0.8 / n
    for i, (opt, hist) in enumerate(results.items()):
        vals   = [hist["survival_acc"][-1] * 100, hist["subtype_f1"][-1] * 100]
        offset = (i - n / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=opt,
                        color=color_map[opt], alpha=0.8)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{bar.get_height():.1f}",
                    ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(["Survival Acc (%)", "Subtype F1 (%)"])
    ax.set_title("Final Results")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"{cancer.upper()} — M-SAM Comparison", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "training_msam.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot : {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(cfg: dict, opt_names: list):
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = (torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cuda") if torch.cuda.is_available() else
              torch.device("cpu"))
    print(f"Device     : {device}")
    print(f"Cancer     : {cfg['cancer'].upper()}")
    print(f"Optimizers : {opt_names}")
    print(f"Fusion     : {cfg['fusion_type']}")

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

    n_genes   = loaders["n_genes"]
    criterion = MultiTaskLoss(
        surv_w=cfg["loss_weights"]["survival"],
        sub_w =cfg["loss_weights"]["subtype"],
    )

    # Save to msam subdirectory
    save_dir = os.path.join(cfg["save_dir"], "msam")
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    for opt_name in opt_names:
        print(f"\n{'─'*60}")
        print(f"  Optimizer: {opt_name.upper()}")
        print(f"{'─'*60}")

        model = build_model(
            n_genes     = n_genes,
            image_dim   = cfg["image_dim"],
            n_subtypes  = cfg["n_subtypes"],
            fusion_type = cfg["fusion_type"],
            dropout     = cfg["dropout"],
            verbose     = (opt_names.index(opt_name) == 0),
        ).to(device)

        opt, scheduler, opt_type = build_optimizer(opt_name, model, cfg)

        history = {
            "train_loss": [], "val_loss": [],
            "survival_acc": [], "subtype_f1": [],
            "train_surv_acc": [], "train_sub_acc": [],
        }

        for epoch in range(cfg["epochs"]):
            train_loss, train_accs = train_one_epoch(
                model, loaders["train"], opt, criterion, device, opt_type
            )
            val_loss, val_accs, val_f1s = evaluate(
                model, loaders["val"], criterion, device
            )

            if scheduler is not None:
                scheduler.step()

            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss))
            history["survival_acc"].append(float(val_accs["survival"]))
            history["subtype_f1"].append(float(val_f1s["subtype"]))
            history["train_surv_acc"].append(float(train_accs["survival"]))
            history["train_sub_acc"].append(float(train_accs["subtype"]))

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"  ep {epoch+1:3d}/{cfg['epochs']}  "
                      f"loss={val_loss:.4f}  "
                      f"surv={val_accs['survival']*100:.1f}%  "
                      f"sub_f1={val_f1s['subtype']:.3f}")

        # Final test evaluation
        _, test_accs, test_f1s = evaluate(
            model, loaders["test"], criterion, device
        )
        history["test_survival_acc"] = float(test_accs["survival"])
        history["test_subtype_f1"]   = float(test_f1s["subtype"])
        print(f"  [TEST] surv={test_accs['survival']*100:.2f}%  "
              f"sub_f1={test_f1s['subtype']:.3f}")

        hist_path = os.path.join(save_dir, f"{opt_name}_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"  Saved : {hist_path}")

        results[opt_name] = history

    plot_results(results, cfg["cancer"], save_dir)
    print(f"\nDone. Results in: {save_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multimodal cancer training with M-SAM optimizer"
    )
    parser.add_argument("--config", required=True,
                        help="Path to YAML config (e.g. configs/brca.yaml)")
    parser.add_argument("--optimizers", nargs="+",
                        default=["msam"],
                        help=f"Optimizer(s) to run. Use 'all' for all 7. "
                             f"Choices: {OPTIMIZER_NAMES}")
    parser.add_argument("--fusion",       type=str,   default=None)
    parser.add_argument("--epochs",       type=int,   default=None)
    parser.add_argument("--batch_size",   type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--n_top_genes",  type=int,   default=None)
    parser.add_argument("--save_dir",     type=str,   default=None)
    parser.add_argument("--seed",         type=int,   default=None)
    parser.add_argument("--rho",          type=float, default=None,
                        help="SAM perturbation radius (default 0.05)")

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.fusion      is not None: cfg["fusion_type"] = args.fusion
    if args.epochs      is not None: cfg["epochs"]      = args.epochs
    if args.batch_size  is not None: cfg["batch_size"]  = args.batch_size
    if args.lr          is not None: cfg["lr"]          = args.lr
    if args.n_top_genes is not None: cfg["n_top_genes"] = args.n_top_genes
    if args.save_dir    is not None: cfg["save_dir"]    = args.save_dir
    if args.seed        is not None: cfg["seed"]        = args.seed

    opt_names = OPTIMIZER_NAMES if "all" in args.optimizers else args.optimizers

    main(cfg, opt_names)
