"""HAM10000 training loop: AdamW, SAM, Adaptive-Aux, SAMO, RA-SAMO + unimodal controls.

Run from repository root:
    python -m experiments.ham10000.train --data-dir data/ham10000 --methods adamw samo ra_samo
"""

import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.ham10000.dataset import build_ham10000_splits, NUM_CLASSES
from experiments.ham10000.models import (
    HAM10000Model, HAM10000ModelWithAux, ImageOnlyModel, MetaOnlyModel,
)
from experiments.ham10000.objectives import classification_loss_parts
from experiments.samo.objectives import LossWeightController
from experiments.samo.representation_gradients import (
    prepare_branch_gradients, apply_branch_gradients,
)
from optimizers.sam_optimizer import SAMOptimizer

METHODS = ("adamw", "sam", "adaptive_aux", "samo", "ra_samo")
UNIMODAL_METHODS = ("image_only_adamw", "meta_only_adamw")
ALL_METHODS = METHODS + UNIMODAL_METHODS


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, n_classes=NUM_CLASSES):
    """Evaluate model on a data loader.

    Returns dict with: loss, accuracy, macro_f1, balanced_acc, auroc,
                       per_class_f1, confusion_matrix
    """
    model.eval()
    all_preds, all_labels, all_probs, losses = [], [], [], []

    with torch.no_grad():
        for inputs, labels_dict in loader:
            x = {k: v.to(device) for k, v in inputs.items()}
            y = labels_dict["label"].to(device)
            outputs = model(x)
            logits = outputs["logits"]
            loss = F.cross_entropy(logits, y)
            losses.append(loss.item())
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            all_probs.append(probs.cpu())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = torch.cat(all_probs, dim=0).numpy()

    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    # Per-class precision, recall, F1
    per_class_f1 = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        per_class_f1.append(f1)

    macro_f1 = float(np.mean(per_class_f1))

    # Balanced accuracy
    per_class_acc = []
    for c in range(n_classes):
        n_c = cm[c, :].sum()
        if n_c > 0:
            per_class_acc.append(cm[c, c] / n_c)
    balanced_acc = float(np.mean(per_class_acc)) if per_class_acc else 0.0

    # AUROC (one-vs-rest, macro)
    auroc = _macro_auroc(all_labels, all_probs, n_classes)

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float((all_preds == all_labels).mean()),
        "macro_f1": macro_f1,
        "balanced_acc": balanced_acc,
        "auroc": auroc,
        "per_class_f1": [float(f) for f in per_class_f1],
        "confusion_matrix": cm.tolist(),
    }


def _macro_auroc(labels, probs, n_classes):
    """Compute macro AUROC (one-vs-rest) without sklearn dependency."""
    aurocs = []
    for c in range(n_classes):
        binary_labels = (labels == c).astype(float)
        scores = probs[:, c]
        if binary_labels.sum() == 0 or binary_labels.sum() == len(binary_labels):
            continue  # skip classes not present
        aurocs.append(_binary_auroc(binary_labels, scores))
    return float(np.mean(aurocs)) if aurocs else 0.0


def _binary_auroc(labels, scores):
    """Simple AUROC computation via sorting."""
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    n_pos = labels_sorted.sum()
    n_neg = len(labels_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = 0.0
    auc = 0.0
    for i in range(len(labels_sorted)):
        if labels_sorted[i] == 1:
            tp += 1
        else:
            auc += tp
    return auc / (n_pos * n_neg)


def _build_model(method, meta_dim, n_classes, device, pretrained=True):
    """Build the appropriate model for a given method."""
    if method == "image_only_adamw":
        model = ImageOnlyModel(n_classes, pretrained=pretrained)
    elif method == "meta_only_adamw":
        model = MetaOnlyModel(meta_dim, n_classes)
    elif method in ("adamw", "sam"):
        model = HAM10000Model(meta_dim, n_classes, pretrained=pretrained)
    else:
        # adaptive_aux, samo, ra_samo need aux heads
        model = HAM10000ModelWithAux(meta_dim, n_classes, pretrained=pretrained)
    return model.to(device)


def run_one(args, seed, method):
    """Train one method with one seed."""
    out = Path(args.out) / method / f"seed_{seed}"
    summary_path = out / "summary.json"
    if args.skip_existing and summary_path.exists():
        data = json.loads(summary_path.read_text())
        if data.get("completed") and data.get("epochs") == args.epochs:
            print(f"SKIP {out}", flush=True)
            return
    out.mkdir(parents=True, exist_ok=True)

    seed_all(seed)
    splits = build_ham10000_splits(args.data_dir, seed=seed,
                                   val_frac=0.15, test_frac=0.15)
    train_ds, val_ds, test_ds = splits["train"], splits["val"], splits["test"]
    meta_dim = splits["meta_dim"]
    class_weights = splits["class_weights"].to(args.device)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=args.workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
    )

    seed_all(seed)
    model = _build_model(method, meta_dim, NUM_CLASSES, args.device,
                         pretrained=args.pretrained)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    sam_opt = None
    if method == "sam":
        sam_opt = SAMOptimizer(optimizer, model, rho=args.sam_rho)

    # Loss weight controller for adaptive/samo/ra_samo
    adaptive = method in ("adaptive_aux", "samo", "ra_samo")
    controller = LossWeightController(
        initial=(args.aux_weight, args.aux_weight, 0.0),  # no ranking
        min_weight=0.02, max_weight=0.3, warmup=2,
    )
    weights = [args.aux_weight, args.aux_weight, 0.0]

    history = []
    best_f1 = -float("inf")
    best_state = None
    best_epoch = 0
    start = time.perf_counter()

    for epoch in range(args.epochs):
        # Cosine annealing LR
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        epoch_losses = np.zeros(4)
        n_batches = 0
        diag_sums = {
            "image_cosine_aux": 0.0, "image_projected_aux": 0.0,
            "gene_cosine_aux": 0.0, "gene_projected_aux": 0.0,
        }
        diag_n = 0

        for inputs, labels_dict in train_loader:
            x = {k: v.to(args.device) for k, v in inputs.items()}
            y = labels_dict["label"].to(args.device)

            if method == "sam":
                # SAM requires closure
                def closure():
                    out = model(x)
                    loss = F.cross_entropy(out["logits"], y)
                    return loss, out
                loss, outputs = sam_opt.step(closure)
                epoch_losses += np.array([loss.item(), 0.0, 0.0, 0.0])
                n_batches += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            parts = classification_loss_parts(outputs, y, method)

            effective_weights = weights
            loss = (parts[0]
                    + effective_weights[0] * parts[1]
                    + effective_weights[1] * parts[2]
                    + effective_weights[2] * parts[3])

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: {method} seed={seed}")

            # RA-SAMO: branch gradient projection
            prepared = None
            if method == "ra_samo":
                prepared = prepare_branch_gradients(
                    model, *parts, effective_weights, project=True)

            loss.backward()

            if prepared is not None:
                diagnostics = apply_branch_gradients(prepared)
                diag_n += 1
                for branch, diag in diagnostics.items():
                    diag_sums[f"{branch}_cosine_aux"] += diag.cosine_aux
                    diag_sums[f"{branch}_projected_aux"] += float(diag.projected_aux)

            optimizer.step()
            epoch_losses += np.array([float(p.detach()) for p in parts])
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("Empty training epoch")

        means = (epoch_losses / n_batches).tolist()

        # Update adaptive weights
        if adaptive:
            new_w = controller.update(epoch, means)
            weights[:2] = new_w[:2]

        # Validation
        val_metrics = evaluate(model, val_loader, args.device)
        diag_means = {k: v / diag_n for k, v in diag_sums.items()} if diag_n else {}

        row = {
            "epoch": epoch + 1,
            "train_fusion_loss": means[0],
            "train_img_aux_loss": means[1],
            "train_meta_aux_loss": means[2],
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_acc": val_metrics["balanced_acc"],
            "val_auroc": val_metrics["auroc"],
            "weight_img": weights[0],
            "weight_meta": weights[1],
            "lr": lr,
            "elapsed_seconds": time.perf_counter() - start,
            **diag_means,
        }
        history.append(row)

        # Checkpoint on best val macro-F1
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        print(f"{method} seed={seed} epoch={epoch+1}/{args.epochs} "
              f"val_f1={val_metrics['macro_f1']:.4f} "
              f"val_acc={val_metrics['balanced_acc']:.4f}", flush=True)

    # Test with best checkpoint
    if best_state is None:
        raise RuntimeError("No valid checkpoint found")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, args.device)

    # Save results
    if history:
        with (out / "epochs.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    np.save(out / "confusion_matrix.npy", np.array(test_metrics["confusion_matrix"]))

    summary = {
        "completed": True,
        "method": method,
        "seed": seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_balanced_acc": test_metrics["balanced_acc"],
        "test_auroc": test_metrics["auroc"],
        "test_per_class_f1": test_metrics["per_class_f1"],
        "wall_seconds": time.perf_counter() - start,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "pretrained": args.pretrained,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  -> test_f1={test_metrics['macro_f1']:.4f} "
          f"test_auroc={test_metrics['auroc']:.4f} "
          f"(best epoch {best_epoch})", flush=True)


def main():
    p = argparse.ArgumentParser(description="HAM10000 multimodal classification benchmark")
    p.add_argument("--data-dir", required=True, help="Path to HAM10000 data")
    p.add_argument("--methods", nargs="+", choices=ALL_METHODS,
                   default=list(ALL_METHODS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--aux-weight", type=float, default=0.1)
    p.add_argument("--sam-rho", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--out", default="results/ham10000_classification")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {args.device}")
    print(f"Methods: {args.methods}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")

    for method in args.methods:
        for seed in args.seeds:
            run_one(args, seed, method)


if __name__ == "__main__":
    main()
