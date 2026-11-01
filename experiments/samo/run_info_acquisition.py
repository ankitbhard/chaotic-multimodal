"""Information Acquisition Diagnostics — instrumented training runner.

Runs 27 training experiments (3 cancers x 3 methods x 3 folds x 50 epochs)
with per-epoch gradient sensitivity, Fisher trace, alignment, and C-index
diagnostics. All diagnostics are noninvasive: computed on a fixed training-only
probe subset without modifying model or optimizer state.

Usage:
    python -m experiments.samo.run_info_acquisition \\
        --cancers brca kirc luad \\
        --methods baseline adaptive_aux rank \\
        --folds 3 --epochs 50 --out results/information_acquisition_diagnostics
"""
import argparse, csv, json, math, time, hashlib, subprocess
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from train_survival import build_survival_dataloaders, concordance_index
from models.joint_model import build_model
from experiments.cvpr_reliability_pilot.models import JointModelWithAux
from experiments.optimizer_dynamics_overnight.run import seed_all
from experiments.samo.objectives import cox_efron_free, ranking_loss, LossWeightController
from experiments.samo.information_diagnostics import (
    build_probe_subset,
    compute_epoch_diagnostics,
    diagnostics_to_row,
    fit_unimodal_probes,
)

METHODS = ("baseline", "adaptive_aux", "rank")


def _get_git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def run_one(args, cancer, fold, method):
    """Run one instrumented training experiment."""
    out = Path(args.out) / cancer / f"fold_{fold}" / method
    summary_path = out / "summary.json"
    if args.skip_existing and summary_path.exists():
        data = json.loads(summary_path.read_text())
        if data.get("completed") and data.get("epochs") == args.epochs:
            print(f"SKIP {out}", flush=True)
            return
    out.mkdir(parents=True, exist_ok=True)

    seed_all(args.seed + fold)
    cfg = yaml.safe_load(Path("configs", cancer + ".yaml").read_text())
    data = build_survival_dataloaders(
        uni_dir=cfg["uni_dir"],
        gene_csv_paths=cfg["gene_csv_paths"],
        image_dim=cfg["image_dim"],
        n_top_genes=cfg.get("n_top_genes", 20000),
        batch_size=cfg.get("batch_size", 32),
        seed=args.seed, fold=fold, n_folds=args.folds,
    )
    generator = torch.Generator()
    train_loader = DataLoader(
        data["train"].dataset,
        batch_size=cfg.get("batch_size", 32),
        shuffle=True, drop_last=True, num_workers=0,
        generator=generator,
    )
    val_loader = data["val"]
    test_loader = data["test"]

    # Build probe subset from training data only
    probe_indices, probe_meta = build_probe_subset(
        data["train"].dataset, n_probe=min(64, len(data["train"].dataset)),
        seed=args.seed + fold,
    )
    probe_dataset = Subset(data["train"].dataset, probe_indices)
    probe_loader = DataLoader(
        probe_dataset, batch_size=min(32, len(probe_dataset)),
        shuffle=False, num_workers=0,
    )

    # Build model — match exact initialization from run.py
    seed_all(args.seed + fold)
    baseline_model = build_model(
        data["n_genes"], data["image_dim"], "attention", 0.3, verbose=False,
    )
    if method in ("baseline", "rank"):
        model = baseline_model
    else:
        model = JointModelWithAux(data["n_genes"], data["image_dim"], 0.3)
        src = baseline_model.state_dict()
        target = model.state_dict()
        matched = {k: v for k, v in src.items()
                   if k in target and target[k].shape == v.shape}
        model.load_state_dict(matched, strict=False)
        if not matched:
            raise RuntimeError("No baseline tensors matched auxiliary model")
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # Adaptive controller for adaptive_aux
    controller = LossWeightController(
        initial=(args.aux_weight, args.aux_weight, args.rank_weight),
        min_weight=args.min_weight, max_weight=args.max_weight,
        warmup=args.warmup,
    )
    weights = [args.aux_weight, args.aux_weight, args.rank_weight]

    # Epochs to collect diagnostics
    diag_epochs = set(range(1, 16))  # epochs 1-15
    diag_epochs.update([20, 25, 30, 40, 50])

    history = []
    diagnostics_rows = []
    best = -float("inf")
    best_state = None
    best_epoch = 0
    unimodal_probes = None
    start = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_1 = epoch + 1
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        generator.manual_seed(args.seed + fold * 10000 + epoch)

        # Training loop — identical to run.py
        sums = np.zeros(4)
        n = 0
        for inputs, labels in train_loader:
            x = {k: v.to(args.device) for k, v in inputs.items()}
            t = labels["os_months"].float().to(args.device)
            e = labels["vital_status"].float().to(args.device)
            optimizer.zero_grad(set_to_none=True)

            outputs = model(x)
            risk = outputs["survival"][:, 1]
            fusion = cox_efron_free(risk, t, e)
            zero = fusion * 0.0

            # Loss parts matching run.py logic
            if method == "adaptive_aux":
                img_loss = cox_efron_free(outputs["aux_img_risk"], t, e)
                gene_loss = cox_efron_free(outputs["aux_gene_risk"], t, e)
                rank_loss_val = zero
            elif method == "rank":
                img_loss = zero
                gene_loss = zero
                rank_loss_val = ranking_loss(risk, t, e, args.temperature)
            else:  # baseline
                img_loss = zero
                gene_loss = zero
                rank_loss_val = zero

            loss = (fusion
                    + weights[0] * img_loss
                    + weights[1] * gene_loss
                    + weights[2] * rank_loss_val)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss {cancer}/{fold}/{method}")
            loss.backward()
            optimizer.step()
            sums += np.array([float(z.detach()) for z in
                              [fusion, img_loss, gene_loss, rank_loss_val]])
            n += 1

        if n == 0:
            raise RuntimeError("Empty training epoch")
        means = (sums / n).tolist()

        # Update adaptive weights
        if method == "adaptive_aux":
            new = controller.update(epoch, means)
            weights[:2] = new[:2]

        # Validation
        val_loss, val_ci = _evaluate(model, val_loader, args.device)

        # --- Epoch diagnostics ---
        diag_row = None
        if epoch_1 in diag_epochs:
            diag_start = time.perf_counter()

            # Fit unimodal probes for baseline/rank (re-fit every diagnostic epoch)
            if method in ("baseline", "rank"):
                unimodal_probes = fit_unimodal_probes(
                    model, train_loader, args.device, n_epochs=10, lr=0.01,
                )

            diag = compute_epoch_diagnostics(
                model=model,
                probe_loader=probe_loader,
                val_loader=val_loader,
                device=args.device,
                method=method,
                epoch=epoch_1,
                cancer=cancer,
                fold=fold,
                temperature=args.temperature,
                unimodal_probes=unimodal_probes,
            )
            diag.train_fusion_loss = means[0]
            diag.train_img_loss = means[1]
            diag.train_gene_loss = means[2]
            diag.train_rank_loss = means[3]
            diag.wall_seconds = time.perf_counter() - diag_start

            diag_row = diagnostics_to_row(diag)
            diag_row["val_loss"] = val_loss
            diag_row["lr"] = lr
            diag_row["weight_img"] = weights[0]
            diag_row["weight_gene"] = weights[1]
            diag_row["weight_rank"] = weights[2]
            diagnostics_rows.append(diag_row)

        # Training history
        history.append({
            "epoch": epoch_1,
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
        })

        # Checkpoint
        if np.isfinite(val_ci) and val_ci > best:
            best = val_ci
            best_epoch = epoch_1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        print(f"{cancer} fold={fold} {method} epoch={epoch_1}/{args.epochs} "
              f"val={val_ci:.4f}"
              + (f" [diag]" if diag_row else ""),
              flush=True)

    if best_state is None:
        raise RuntimeError("No finite validation C-index")

    # Test on best checkpoint
    model.load_state_dict(best_state)
    test_loss, test_ci = _evaluate(model, test_loader, args.device)

    # Save diagnostics CSV
    if diagnostics_rows:
        diag_path = out / "diagnostics.csv"
        with diag_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(diagnostics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(diagnostics_rows)

    # Save epoch history
    with (out / "epochs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # Save metadata
    metadata = {
        "completed": True,
        "dataset": cancer,
        "fold": fold,
        "method": method,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_ci": best,
        "test_ci": test_ci,
        "test_loss": test_loss,
        "wall_seconds": time.perf_counter() - start,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed + fold,
        "git_sha": _get_git_sha(),
        "probe": probe_meta,
        "diagnostic_epochs": sorted(diag_epochs),
        "definitions": {
            "squared_cox_grad_norm": (
                "L2-norm-squared of mini-batch Cox gradient w.r.t. encoder parameters. "
                "NOT the empirical Fisher trace (mini-batch Cox couples individuals via risk sets)."
            ),
            "fisher_trace": (
                "Sum of squared per-event score-gradient norms / n_events. "
                "Each event score = d/d(theta)[risk_i - log sum_{j in R_i} exp(risk_j)]. "
                "Breslow tied-event handling. Mini-batch risk set approximation (biased)."
            ),
            "fisher_trace_normalized": "fisher_trace / n_encoder_parameters.",
            "grad_sensitivity": "mean |grad_Cox| / n_encoder_parameters.",
            "unimodal_ci_baseline_rank": (
                "For baseline/rank methods without aux heads, unimodal C-index is computed "
                "via frozen-feature linear Cox probes fitted on training data. These are "
                "NOT equivalent to trained auxiliary heads and have limited capacity."
            ),
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Save summary
    summary_path.write_text(json.dumps({
        "completed": True,
        "dataset": cancer,
        "fold": fold,
        "method": method,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_ci": best,
        "test_ci": test_ci,
        "test_loss": test_loss,
        "wall_seconds": time.perf_counter() - start,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed + fold,
    }, indent=2))


def _evaluate(model, loader, device):
    """Evaluate model on a dataset. Returns (loss, c_index)."""
    model.eval()
    risks, times, events, losses = [], [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            output = model({k: v.to(device) for k, v in inputs.items()})
            risk = output["survival"][:, 1]
            t = labels["os_months"].float().to(device)
            e = labels["vital_status"].float().to(device)
            losses.append(float(cox_efron_free(risk, t, e)))
            risks.extend(risk.cpu().tolist())
            times.extend(t.cpu().tolist())
            events.extend(e.cpu().tolist())
    return float(np.mean(losses)), float(concordance_index(times, risks, events))


def main():
    p = argparse.ArgumentParser(
        description="Information Acquisition Diagnostics runner",
    )
    p.add_argument("--cancers", nargs="+", default=["brca", "kirc", "luad"])
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--aux-weight", type=float, default=0.1)
    p.add_argument("--rank-weight", type=float, default=0.1)
    p.add_argument("--min-weight", type=float, default=0.02)
    p.add_argument("--max-weight", type=float, default=0.3)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--out", default="results/information_acquisition_diagnostics")
    p.add_argument("--skip-existing", action="store_true")
    a = p.parse_args()
    a.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {a.device}")
    print(f"Cancers: {a.cancers}")
    print(f"Methods: {a.methods}")
    print(f"Folds: {a.folds}, Epochs: {a.epochs}")
    print(f"Output: {a.out}")
    print(f"Total runs: {len(a.cancers) * len(a.methods) * a.folds}")

    for cancer in a.cancers:
        for fold in range(a.folds):
            for method in a.methods:
                run_one(a, cancer, fold, method)

    print("All runs complete.", flush=True)


if __name__ == "__main__":
    main()
