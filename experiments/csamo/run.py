"""C-SAMO: validation-guided objective selection with held-out checkpoint validation.

Run from repository root: python -m experiments.csamo.run ...
"""
import argparse
import copy
import csv
import json
import math
import time
import resource
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from train_survival import build_survival_dataloaders, concordance_index
from models.joint_model import build_model
from experiments.cvpr_reliability_pilot.models import JointModelWithAux
from experiments.optimizer_dynamics_overnight.run import seed_all
from experiments.samo.objectives import cox_efron_free, ranking_loss

CANDIDATES = ("cox", "aux", "rank", "aux_rank")
WEIGHTS = {"cox": (0., 0.), "aux": (0.1, 0.), "rank": (0., 0.1),
           "aux_rank": (0.1, 0.1)}

def memory_stats(device):
    """Process peak RSS and CUDA allocator memory; values are MiB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mib = rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    result = {"process_peak_rss_mib": round(rss_mib, 2)}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        result.update(
            gpu_peak_allocated_mib=round(torch.cuda.max_memory_allocated(device) / 2**20, 2),
            gpu_peak_reserved_mib=round(torch.cuda.max_memory_reserved(device) / 2**20, 2),
            gpu_current_allocated_mib=round(torch.cuda.memory_allocated(device) / 2**20, 2))
    else:
        result.update(gpu_peak_allocated_mib=None, gpu_peak_reserved_mib=None,
                      gpu_current_allocated_mib=None)
    return result

def objective(model, inputs, labels, mode, temperature):
    outputs = model(inputs)
    risk = outputs["survival"][:, 1]
    t = labels["os_months"].float().to(risk.device)
    e = labels["vital_status"].float().to(risk.device)
    fusion = cox_efron_free(risk, t, e)
    aux_w, rank_w = WEIGHTS[mode]
    loss = fusion
    if aux_w:
        loss = loss + aux_w * (
            cox_efron_free(outputs["aux_img_risk"], t, e)
            + cox_efron_free(outputs["aux_gene_risk"], t, e))
    if rank_w:
        loss = loss + rank_w * ranking_loss(risk, t, e, temperature)
    return loss

def evaluate_ci(model, loader, device):
    model.eval()
    risks, times, events = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            out = model({k: v.to(device) for k, v in inputs.items()})
            risks.extend(out["survival"][:, 1].detach().cpu().tolist())
            times.extend(labels["os_months"].tolist())
            events.extend(labels["vital_status"].tolist())
    if len(risks) < 2:
        return float("nan")
    return float(concordance_index(times, risks, events))

def validation_loaders(loader, fraction, seed):
    dataset = loader.dataset
    n = len(dataset)
    if n < 12:
        raise ValueError("Validation partition too small for controller/checkpoint split")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).tolist()
    cut = max(2, min(n - 2, round(n * fraction)))
    kwargs = dict(batch_size=loader.batch_size, shuffle=False, num_workers=0)
    return (DataLoader(Subset(dataset, perm[:cut]), **kwargs),
            DataLoader(Subset(dataset, perm[cut:]), **kwargs))

def choose_mode(model, optimizer, batch, control_loader, device, temperature,
                current, threshold):
    """Evaluate equal one-step AdamW counterfactuals; do not modify live state."""
    x, y = batch
    x = {k: v.to(device) for k, v in x.items()}
    scores = {}
    # Deep-copy optimizer together with model parameters so Adam moments remain matched.
    for mode in CANDIDATES:
        trial_model, trial_optimizer = copy.deepcopy((model, optimizer))
        trial_model.train()
        trial_optimizer.zero_grad(set_to_none=True)
        loss = objective(trial_model, x, y, mode, temperature)
        if not torch.isfinite(loss):
            scores[mode] = float("nan")
            continue
        loss.backward()
        trial_optimizer.step()
        scores[mode] = evaluate_ci(trial_model, control_loader, device)
        del trial_model, trial_optimizer
    valid = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not valid:
        return current, scores
    best = max(valid, key=valid.get)
    # Avoid switches on ties and small noisy C-index differences.
    if current in valid and valid[best] < valid[current] + threshold:
        return current, scores
    return best, scores

def run_one(args, cancer, fold):
    out = Path(args.out) / cancer / f"fold_{fold}" / "csamo"
    summary_path = out / "summary.json"
    if args.skip_existing and summary_path.exists():
        prior = json.loads(summary_path.read_text())
        if prior.get("completed") and prior.get("epochs") == args.epochs:
            print("SKIP", out, flush=True)
            return
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed + fold)
    cfg = yaml.safe_load(Path("configs", cancer + ".yaml").read_text())
    data = build_survival_dataloaders(
        uni_dir=cfg["uni_dir"], gene_csv_paths=cfg["gene_csv_paths"],
        image_dim=cfg["image_dim"], n_top_genes=cfg.get("n_top_genes", 20000),
        batch_size=cfg.get("batch_size", 32), seed=args.seed,
        fold=fold, n_folds=args.folds)
    gen = torch.Generator()
    train = DataLoader(data["train"].dataset, batch_size=cfg.get("batch_size", 32),
                       shuffle=True, drop_last=True, num_workers=0, generator=gen)
    controller_val, checkpoint_val = validation_loaders(
        data["val"], args.controller_fraction, args.seed + 10000 + fold)
    test = data["test"]
    seed_all(args.seed + fold)
    base = build_model(data["n_genes"], data["image_dim"], "attention", 0.3, verbose=False)
    model = JointModelWithAux(data["n_genes"], data["image_dim"], 0.3)
    source, target = base.state_dict(), model.state_dict()
    matched = {k: v for k, v in source.items()
               if k in target and target[k].shape == v.shape}
    if not matched:
        raise RuntimeError("No baseline tensors matched auxiliary model")
    model.load_state_dict(matched, strict=False)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    mode = "aux_rank"
    best, best_epoch, best_state = -float("inf"), 0, None
    history, decisions = [], []
    started = time.perf_counter()
    if args.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        if args.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(args.device)
        decision_seconds = 0.0
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))
        for group in optimizer.param_groups:
            group["lr"] = lr
        gen.manual_seed(args.seed + fold * 10000 + epoch)
        # Use one fixed training batch for all candidate lookaheads.
        if epoch >= args.warmup and (epoch - args.warmup) % args.interval == 0:
            decision_started = time.perf_counter()
            probe = next(iter(train))
            new_mode, scores = choose_mode(
                model, optimizer, probe, controller_val, args.device,
                args.temperature, mode, args.min_gain)
            if args.device.type == "cuda":
                torch.cuda.synchronize(args.device)
            decision_seconds = time.perf_counter() - decision_started
            decisions.append(dict(epoch=epoch + 1, previous=mode,
                                  selected=new_mode, decision_seconds=decision_seconds,
                                  **scores))
            mode = new_mode
            # Re-seed generator: probing must not perturb the main epoch shuffle.
            gen.manual_seed(args.seed + fold * 10000 + epoch)
        model.train()
        losses = []
        for inputs, labels in train:
            x = {k: v.to(args.device) for k, v in inputs.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, x, labels, mode, args.temperature)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss: {cancer} fold {fold} epoch {epoch}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if not losses:
            raise RuntimeError("Empty training epoch")
        val_ci = evaluate_ci(model, checkpoint_val, args.device)
        if args.device.type == "cuda":
            torch.cuda.synchronize(args.device)
        epoch_seconds = time.perf_counter() - epoch_started
        mem = memory_stats(args.device)
        history.append(dict(epoch=epoch + 1, mode=mode,
                            train_loss=float(np.mean(losses)), checkpoint_val_ci=val_ci,
                            lr=lr, epoch_seconds=epoch_seconds,
                            decision_seconds=decision_seconds,
                            elapsed_seconds=time.perf_counter() - started, **mem))
        if np.isfinite(val_ci) and val_ci > best:
            best, best_epoch = val_ci, epoch + 1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        print(f"{cancer} fold={fold} epoch={epoch+1}/{args.epochs} "
              f"mode={mode} checkpoint_val_ci={val_ci:.4f} "
              f"epoch_s={epoch_seconds:.1f} peak_gpu_mib={mem['gpu_peak_allocated_mib']}", flush=True)
    if best_state is None:
        raise RuntimeError("No finite checkpoint validation C-index")
    model.load_state_dict(best_state)
    test_ci = evaluate_ci(model, test, args.device)
    for filename, rows in (("epochs.csv", history), ("decisions.csv", decisions)):
        if rows:
            with (out / filename).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    final_memory = memory_stats(args.device)
    summary_path.write_text(json.dumps(dict(
        completed=True, dataset=cancer, fold=fold, method="csamo",
        epochs=args.epochs, best_epoch=best_epoch, best_val_ci=best,
        test_ci=test_ci, wall_seconds=time.perf_counter()-started,
        seed=args.seed+fold, lr=args.lr, weight_decay=args.weight_decay,
        parameter_count=sum(p.numel() for p in model.parameters()),
        controller_fraction=args.controller_fraction,
        interval=args.interval, warmup=args.warmup,
        min_gain=args.min_gain, decisions=decisions,
        mean_epoch_seconds=float(np.mean([row["epoch_seconds"] for row in history])),
        max_epoch_seconds=float(max(row["epoch_seconds"] for row in history)),
        total_decision_seconds=float(sum(row["decision_seconds"] for row in history)),
        **final_memory), indent=2))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cancers", nargs="+", default=["brca", "kirc", "stad"])
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--interval", type=int, default=5)
    p.add_argument("--min-gain", type=float, default=0.01)
    p.add_argument("--controller-fraction", type=float, default=0.5)
    p.add_argument("--out", default="results/csamo_survival")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()
    if args.interval < 1 or args.warmup < 0 or not 0 < args.controller_fraction < 1:
        p.error("invalid controller schedule or fraction")
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for cancer in args.cancers:
        for fold in range(args.folds):
            run_one(args, cancer, fold)

if __name__ == "__main__":
    main()
