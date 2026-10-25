#!/usr/bin/env python3
"""SGC-AdamW experiment runner for KIRC/STAD survival prediction.

Uses the same data loaders, model, splits, metrics, seeds, scheduler, and
selection logic as experiments/optimizer_dynamics_overnight/run.py.  Replaces
the monkey-patching approach with an explicit optimizer factory.

Optimizer variants:
  adamw_b999             torch.optim.AdamW  beta2=0.999
  adamw_b98              torch.optim.AdamW  beta2=0.98
  adamw_b95              torch.optim.AdamW  beta2=0.95
  dynamics_aware         DynamicsAwareAdamW beta2_min=0.95, sens=3, ema=0.9
  sgc_drift              SGCAdamW drift-only
  sgc_drift_alignment    SGCAdamW drift+alignment
  sgc_drift_align_curv   SGCAdamW drift+alignment+curvature
"""
import argparse, csv, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.sgc_adamw.sgc_optimizer import SGCAdamW
from experiments.dynamics_aware_ec2.dynamics_optimizer import DynamicsAwareAdamW

ALL_OPTIMIZERS = [
    "adamw_b999", "adamw_b98", "adamw_b95",
    "dynamics_aware",
    "sgc_drift", "sgc_drift_alignment", "sgc_drift_align_curv",
]


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(name, params, lr, wd):
    """Factory: return (optimizer, needs_curvature_probe: bool)."""
    if name == "adamw_b999":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=wd), False
    elif name == "adamw_b98":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.98),
                                 eps=1e-8, weight_decay=wd), False
    elif name == "adamw_b95":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95),
                                 eps=1e-8, weight_decay=wd), False
    elif name == "dynamics_aware":
        return DynamicsAwareAdamW(params, lr=lr, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=wd,
                                  beta2_min=0.95, sensitivity=3.0,
                                  change_ema=0.9), False
    elif name == "sgc_drift":
        return SGCAdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                        weight_decay=wd, mode="drift",
                        beta2_min=0.95, drift_sensitivity=3.0,
                        drift_ema_decay=0.9), False
    elif name == "sgc_drift_alignment":
        return SGCAdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                        weight_decay=wd, mode="drift_alignment",
                        beta2_min=0.95, drift_sensitivity=3.0,
                        drift_ema_decay=0.9, alignment_sensitivity=1.0,
                        alignment_ema_decay=0.9), False
    elif name == "sgc_drift_align_curv":
        return SGCAdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                        weight_decay=wd, mode="drift_alignment_curvature",
                        beta2_min=0.95, drift_sensitivity=3.0,
                        drift_ema_decay=0.9, alignment_sensitivity=1.0,
                        alignment_ema_decay=0.9, curvature_sensitivity=1.0,
                        curvature_ema_decay=0.9), True
    else:
        raise ValueError(f"Unknown optimizer: {name}")


# ── Diagnostics ──────────────────────────────────────────────────────────────

def collect_diagnostics(model, opt, opt_name, before, raw_grads, lr,
                        prev_update):
    """Per-step diagnostics with correct variable-beta2 v_hat."""
    import torch.nn.functional as F
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad or name not in before:
            continue
        group = name.split(".")[0]
        d = groups.setdefault(group, {"g": [], "v_hat": [], "scale": [],
                                      "u": [], "w": [], "beta2": []})
        state = opt.state.get(p, {})
        if not state or "exp_avg" not in state:
            continue
        t = state.get("step", 1)
        t = float(t.item()) if torch.is_tensor(t) else float(t)
        b1 = opt.defaults.get("betas", (0.9, 0.999))[0]
        m = state["exp_avg"].detach() / (1 - b1 ** t)
        # Variable-beta2 bias correction
        bp = state.get("beta2_product", None)
        if bp is not None and bp != 1.0:
            v_hat = state["exp_avg_sq"].detach() / (1.0 - bp)
        else:
            b2 = opt.defaults.get("betas", (0.9, 0.999))[1]
            v_hat = state["exp_avg_sq"].detach() / (1 - b2 ** t)
        eps = opt.defaults.get("eps", 1e-8)
        d["g"].append(raw_grads[name].reshape(-1).float().cpu())
        d["v_hat"].append(v_hat.reshape(-1).float().cpu())
        d["scale"].append((lr / (v_hat.sqrt() + eps)).reshape(-1).float().cpu())
        d["u"].append((p.detach() - before[name]).reshape(-1).float().cpu())
        d["w"].append(before[name].reshape(-1).float().cpu())
        d["beta2"].append(state.get("last_beta2",
                                    opt.defaults.get("betas", (0.9, 0.999))[1]))

    rows = []
    for group, d in groups.items():
        if not d["g"]:
            continue
        z = {k: torch.cat(v) if isinstance(v[0], torch.Tensor) else v
             for k, v in d.items()}
        g, u, w, scale = z["g"], z["u"], z["w"], z["scale"]
        norm = lambda x: float(torch.linalg.vector_norm(x))
        cos = lambda a, b: (float(F.cosine_similarity(a, b, dim=0))
                            if norm(a) > 0 and norm(b) > 0 else float("nan"))
        beta2_mean = sum(z["beta2"]) / len(z["beta2"])
        row = {
            "group": group,
            "grad_norm": norm(g),
            "grad_weight_ratio": norm(g) / (norm(w) + 1e-12),
            "v_hat_mean": float(z["v_hat"].mean()),
            "effective_lr_p10": float(torch.quantile(scale, .1)),
            "effective_lr_p50": float(torch.quantile(scale, .5)),
            "effective_lr_p90": float(torch.quantile(scale, .9)),
            "update_norm": norm(u),
            "update_weight_ratio": norm(u) / (norm(w) + 1e-12),
            "grad_update_cos": cos(g, u),
            "update_prev_cos": (cos(u, prev_update[group])
                                if group in prev_update
                                and prev_update[group].numel() == u.numel()
                                else float("nan")),
            "beta2_mean": beta2_mean,
        }
        prev_update[group] = u
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Collect all fieldnames across all rows (some rows may have extra keys)
    seen = {}
    for row in rows:
        for k in row:
            if k not in seen:
                seen[k] = True
    fieldnames = list(seen)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ── Training loop ────────────────────────────────────────────────────────────

def run_one(args, cancer, fold, opt_name, make_loaders, make_model,
            loss_fn, score_fn):
    out = Path(args.out) / cancer / f"fold_{fold}" / opt_name
    if args.skip_existing and (out / "summary.json").exists():
        print(f"SKIP {out}", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed + fold)
    train_loader, val_loader, test_loader = make_loaders()
    model = make_model().to(args.device)
    opt, needs_curv = make_optimizer(opt_name, model.parameters(),
                                     args.lr, args.weight_decay)
    curv_interval = args.curvature_interval

    step_rows = []
    epoch_rows = []
    prev_update = {}
    best = -float("inf")
    best_state = None
    best_epoch = None
    global_step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))
        for pg in opt.param_groups:
            pg["lr"] = lr

        model.train()
        total_loss = 0
        n_batches = 0

        if hasattr(train_loader.sampler, "generator") and \
                train_loader.sampler.generator is not None:
            train_loader.sampler.generator.manual_seed(
                args.seed + fold * 10000 + epoch)

        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model, batch, args.device)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at {out} epoch {epoch}")
            loss.backward()

            capture = (global_step % args.log_every == 0)
            if capture:
                before = {name: p.detach().clone()
                          for name, p in model.named_parameters()
                          if p.requires_grad and p.grad is not None}
                grads = {name: p.grad.detach().clone()
                         for name, p in model.named_parameters()
                         if name in before}

            # Curvature probe (before step, after backward)
            curv_info = None
            if needs_curv and global_step > 0 and \
                    global_step % curv_interval == 0:
                def _closure():
                    return loss_fn(model, batch, args.device)
                curv_info = opt.curvature_probe(_closure)

            opt.step()

            if capture:
                for row in collect_diagnostics(model, opt, opt_name,
                                               before, grads, lr,
                                               prev_update):
                    row.update(epoch=epoch + 1, step=global_step,
                               optimizer=opt_name, dataset=cancer,
                               fold=fold)
                    if curv_info:
                        row["curvature_probe_s"] = curv_info.get(
                            "elapsed_s", 0)
                    step_rows.append(row)

            total_loss += float(loss.detach())
            n_batches += 1
            global_step += 1

        # Epoch end
        val_loss, val_score = score_fn(model, val_loader, args.device)
        epoch_rows.append({
            "epoch": epoch + 1,
            "train_loss": total_loss / max(n_batches, 1),
            "val_loss": val_loss,
            "val_score": val_score,
            "lr": lr,
        })
        if np.isfinite(val_score) and val_score > best:
            best = val_score
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        print(f"{cancer} fold={fold} {opt_name} "
              f"epoch={epoch+1}/{args.epochs} val={val_score:.4f}",
              flush=True)

    if best_state is None:
        raise RuntimeError("No valid validation score")
    model.load_state_dict(best_state)
    test_loss, test_score = score_fn(model, test_loader, args.device)
    elapsed = time.time() - t0

    write_csv(out / "steps.csv", step_rows)
    write_csv(out / "epochs.csv", epoch_rows)

    # Telemetry snapshot
    telemetry = {}
    if hasattr(opt, "get_telemetry"):
        telemetry = opt.get_telemetry()

    summary = {
        "dataset": cancer,
        "fold": fold,
        "optimizer": opt_name,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_score": best,
        "test_score": test_score,
        "test_loss": test_loss,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "elapsed_s": elapsed,
        "global_steps": global_step,
        "telemetry": telemetry,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"DONE {cancer} fold={fold} {opt_name} "
          f"test={test_score:.4f} best_ep={best_epoch} "
          f"time={elapsed:.0f}s", flush=True)


# ── Survival task wiring ─────────────────────────────────────────────────────

def run_survival(args):
    import yaml
    from train_survival import build_survival_dataloaders, concordance_index
    from models.joint_model import build_model
    from experiments.cvpr_reliability_pilot.losses import cox_loss

    for cancer in args.cancers:
        cfg = yaml.safe_load(Path("configs", cancer + ".yaml").read_text())
        for fold in range(args.folds):
            def make_loaders(cfg=cfg, fold=fold):
                data = build_survival_dataloaders(
                    uni_dir=cfg["uni_dir"],
                    gene_csv_paths=cfg["gene_csv_paths"],
                    image_dim=cfg["image_dim"],
                    n_top_genes=cfg.get("n_top_genes", 20000),
                    batch_size=cfg.get("batch_size", 32),
                    seed=args.seed, fold=fold, n_folds=args.folds)
                args.n_genes = data["n_genes"]
                args.image_dim = data["image_dim"]
                ds = data["train"].dataset
                gen = torch.Generator()
                tr = DataLoader(ds, batch_size=cfg.get("batch_size", 32),
                                shuffle=True, drop_last=True,
                                num_workers=0, generator=gen)
                return tr, data["val"], data["test"]

            def make_model():
                return build_model(args.n_genes, args.image_dim,
                                   "attention", 0.3, verbose=False)

            def loss_fn(m, batch, device):
                inputs, labels = batch
                outputs = m({k: v.to(device) for k, v in inputs.items()})
                return cox_loss(outputs["survival"][:, 1],
                                labels["os_months"].float().to(device),
                                labels["vital_status"].float().to(device))

            @torch.no_grad()
            def score_fn(m, loader, device):
                m.eval()
                risks, times, events, losses = [], [], [], []
                for inputs, labels in loader:
                    outputs = m({k: v.to(device)
                                 for k, v in inputs.items()})
                    risk = outputs["survival"][:, 1]
                    t = labels["os_months"].float().to(device)
                    e = labels["vital_status"].float().to(device)
                    losses.append(float(cox_loss(risk, t, e)))
                    risks.extend(risk.cpu().tolist())
                    times.extend(t.cpu().tolist())
                    events.extend(e.cpu().tolist())
                return (float(np.mean(losses)),
                        float(concordance_index(times, risks, events)))

            for opt_name in args.optimizers:
                run_one(args, cancer, fold, opt_name,
                        make_loaders, make_model, loss_fn, score_fn)


def main():
    p = argparse.ArgumentParser(description="SGC-AdamW KIRC/STAD experiments")
    p.add_argument("--cancers", nargs="+", default=["kirc", "stad"])
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--curvature-interval", type=int, default=10)
    p.add_argument("--out", default="results/sgc_adamw_kirc_stad")
    p.add_argument("--optimizers", nargs="+", default=ALL_OPTIMIZERS)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()
    if args.log_every < 1:
        raise ValueError("--log-every must be >= 1")
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {args.device}", flush=True)
    print(f"Optimizers: {args.optimizers}", flush=True)
    print(f"Cancers: {args.cancers}", flush=True)
    print(f"Folds: {args.folds}, Epochs: {args.epochs}", flush=True)
    run_survival(args)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
