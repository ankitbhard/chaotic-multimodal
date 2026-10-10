"""
loss_landscape.py — Filter-normalized 2D/3D loss landscape visualization.

Generates 3D surface plots of the loss surface around trained model minima
using the filter normalization technique from:
  Li et al., "Visualizing the Loss Landscape of Neural Nets" (NeurIPS 2018).

Two modes:
  1. compute  — Run on EC2 (needs data + checkpoints). Saves .npz files.
  2. plot     — Run locally from precomputed .npz files. Generates figures.

Usage:
    # On EC2: compute landscapes and save .npz
    python loss_landscape.py compute \\
        --config configs/brca.yaml \\
        --checkpoint_dir results_msam_paper_landscape/brca/fold_0/checkpoints \\
        --optimizers cosine_ge sam msam \\
        --output_dir results_msam_paper_landscape/brca/npz \\
        --resolution 31

    # Locally: render 3D surface plots from .npz
    python loss_landscape.py plot \\
        --npz_dir results_msam_paper_landscape/brca/npz \\
        --optimizers cosine_ge sam msam \\
        --output_dir paper/msam_overleaf_sync/figures \\
        --cancer brca
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

try:
    import torch
except ImportError:
    torch = None  # Only needed for compute mode


# ── Filter normalization ─────────────────────────────────────────────────────

def _filter_normalize(direction, model_params):
    """
    Apply filter normalization to a random direction.
    For each parameter, scale the direction to have the same norm as
    the corresponding model parameter (per-filter for conv, per-row for linear).
    """
    normalized = []
    for d, p in zip(direction, model_params):
        if p.dim() <= 1:
            p_norm = p.norm()
            d_norm = d.norm()
            if d_norm > 1e-10:
                normalized.append(d * (p_norm / d_norm))
            else:
                normalized.append(d)
        else:
            for i in range(p.shape[0]):
                p_norm = p[i].norm()
                d_norm = d[i].norm()
                if d_norm > 1e-10:
                    d[i] = d[i] * (p_norm / d_norm)
            normalized.append(d)
    return normalized


def generate_directions(model, seed=42):
    """Generate two random filter-normalized directions for 2D plotting."""
    torch.manual_seed(seed)
    params = [p.data for p in model.parameters() if p.requires_grad]
    dir1 = [torch.randn_like(p) for p in params]
    dir2 = [torch.randn_like(p) for p in params]
    dir1 = _filter_normalize(dir1, params)
    dir2 = _filter_normalize(dir2, params)
    return dir1, dir2


# ── Loss evaluation ──────────────────────────────────────────────────────────

def _set_params(model, base_params, dir1, dir2, alpha, beta):
    """Set model parameters to: base + alpha*dir1 + beta*dir2."""
    idx = 0
    for p in model.parameters():
        if p.requires_grad:
            p.data.copy_(base_params[idx] + alpha * dir1[idx] + beta * dir2[idx])
            idx += 1


def evaluate_loss(model, loader, device):
    """Compute average Cox loss on a data loader."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for inputs, labels in loader:
            img = inputs["image"].to(device)
            gene = inputs["gene"].to(device)
            os_m = labels["os_months"].float().to(device)
            vs = labels["vital_status"].float().to(device)
            outputs = model({"image": img, "gene": gene})
            risk = outputs["survival"][:, 1]
            loss = cox_loss(risk, os_m, vs)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def compute_landscape(model, base_params, dir1, dir2, loader, device,
                      resolution=31, range_val=1.0):
    """Compute 2D loss surface on a grid. Returns (alphas, betas, loss_grid)."""
    alphas = np.linspace(-range_val, range_val, resolution)
    betas = np.linspace(-range_val, range_val, resolution)
    loss_grid = np.zeros((resolution, resolution))

    total = resolution * resolution
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            _set_params(model, base_params, dir1, dir2, alpha, beta)
            loss_grid[i, j] = evaluate_loss(model, loader, device)
            done = i * resolution + j + 1
            if done % 50 == 0 or done == total:
                print(f"  [{done}/{total}] alpha={alpha:.2f} beta={beta:.2f} "
                      f"loss={loss_grid[i, j]:.4f}")

    _set_params(model, base_params, dir1, dir2, 0.0, 0.0)
    return alphas, betas, loss_grid


# ── 3D Surface Plotting ─────────────────────────────────────────────────────

DISPLAY_NAMES = {
    "cosine_ge": "Cosine + OGM-GE",
    "sam": "SAM",
    "msam": "M-SAM (Ours)",
}

SURFACE_COLORS = {
    "cosine_ge": cm.Reds,
    "sam": cm.Oranges,
    "msam": cm.Greens,
}


def plot_3d_landscape(alphas, betas, loss_grid, title, output_path,
                      cmap=cm.coolwarm, elev=35, azim=-60):
    """Generate a single 3D surface plot."""
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(alphas, betas, indexing="ij")
    # Normalize loss for visualization
    Z = loss_grid - loss_grid.min()

    surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.9,
                           edgecolor='none', antialiased=True,
                           rstride=1, cstride=1)

    ax.set_xlabel(r'$\delta_1$', fontsize=12, labelpad=8)
    ax.set_ylabel(r'$\delta_2$', fontsize=12, labelpad=8)
    ax.set_zlabel('Loss', fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.view_init(elev=elev, azim=azim)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_3d_comparison(landscapes, output_path, cancer="BRCA"):
    """
    Plot 3D landscapes for multiple optimizers — two viewing angles per
    optimizer, arranged in a 2-row x N-column grid (like Li et al. 2018).
    Row 1: front view. Row 2: top-down angled view.
    """
    n = len(landscapes)
    fig = plt.figure(figsize=(6 * n, 10))

    views = [
        (35, -60),   # Row 1: front/side view
        (70, -60),   # Row 2: top-down angled view
    ]

    opt_names = list(landscapes.keys())

    for col, opt_name in enumerate(opt_names):
        alphas, betas, loss_grid = landscapes[opt_name]
        X, Y = np.meshgrid(alphas, betas, indexing="ij")
        Z = loss_grid - loss_grid.min()

        display = DISPLAY_NAMES.get(opt_name, opt_name)
        cmap = SURFACE_COLORS.get(opt_name, cm.coolwarm)

        for row, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(2, n, row * n + col + 1, projection='3d')

            ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.92,
                            edgecolor='none', antialiased=True,
                            rstride=1, cstride=1)

            # Mark minimum
            min_idx = np.unravel_index(loss_grid.argmin(), loss_grid.shape)
            ax.scatter([alphas[min_idx[0]]], [betas[min_idx[1]]], [0],
                       color='white', s=80, marker='*', edgecolors='black',
                       linewidths=0.8, zorder=10)

            ax.set_xlabel(r'$\delta_1$', fontsize=10, labelpad=6)
            ax.set_ylabel(r'$\delta_2$', fontsize=10, labelpad=6)
            if col == 0:
                ax.set_zlabel('Loss', fontsize=10, labelpad=6)

            if row == 0:
                ax.set_title(display, fontsize=14, fontweight='bold', pad=12)

            ax.view_init(elev=elev, azim=azim)

            # Clean panes
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('lightgray')
            ax.yaxis.pane.set_edgecolor('lightgray')
            ax.zaxis.pane.set_edgecolor('lightgray')
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=8)

    fig.suptitle(f"Loss Landscape Comparison — {cancer.upper()}\n"
                 f"(Filter-Normalized, Li et al. 2018)",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ── Compute mode (run on EC2) ────────────────────────────────────────────────

def cmd_compute(args):
    import yaml
    import torch

    try:
        from models.joint_model import build_model
        from train_survival import build_survival_dataloaders, cox_loss
    except ModuleNotFoundError:
        sys.path.insert(0, os.path.dirname(__file__))
        from models.joint_model import build_model
        from train_survival import build_survival_dataloaders, cox_loss

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

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
        seed=args.seed,
        fold=0,
        n_folds=5,
    )

    n_genes = loaders["n_genes"]
    image_dim = loaders["image_dim"]

    for opt_name in args.optimizers:
        print(f"\n{'='*60}")
        print(f"  Computing landscape: {opt_name}")
        print(f"{'='*60}")

        ckpt_path = None
        for suffix in ["_ep200.pt", "_best.pt"]:
            p = os.path.join(args.checkpoint_dir, f"{opt_name}{suffix}")
            if os.path.exists(p):
                ckpt_path = p
                break

        if ckpt_path is None:
            print(f"  [SKIP] No checkpoint found for {opt_name}")
            continue

        print(f"  Loading: {ckpt_path}")

        model = build_model(
            n_genes=n_genes, image_dim=image_dim,
            fusion_type=cfg.get("fusion_type", "attention"),
            dropout=cfg.get("dropout", 0.3),
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                         weights_only=True))
        model.to(device)

        base_params = [p.data.clone() for p in model.parameters()
                       if p.requires_grad]
        dir1, dir2 = generate_directions(model, seed=args.seed)

        alphas, betas, loss_grid = compute_landscape(
            model, base_params, dir1, dir2,
            loaders["train"], device,
            resolution=args.resolution,
            range_val=args.range_val,
        )

        np.savez(
            os.path.join(args.output_dir, f"landscape_{opt_name}.npz"),
            alphas=alphas, betas=betas, loss_grid=loss_grid,
        )
        print(f"  Saved .npz for {opt_name}")


# ── Plot mode (run locally from .npz) ────────────────────────────────────────

def cmd_plot(args):
    os.makedirs(args.output_dir, exist_ok=True)

    landscapes = {}
    for opt_name in args.optimizers:
        npz_path = os.path.join(args.npz_dir, f"landscape_{opt_name}.npz")
        if not os.path.exists(npz_path):
            print(f"  [SKIP] {npz_path} not found")
            continue
        data = np.load(npz_path)
        landscapes[opt_name] = (data["alphas"], data["betas"], data["loss_grid"])
        print(f"  Loaded: {npz_path}  "
              f"(grid={data['loss_grid'].shape}, "
              f"loss range=[{data['loss_grid'].min():.4f}, {data['loss_grid'].max():.4f}])")

        # Individual 3D plot
        cmap = SURFACE_COLORS.get(opt_name, cm.coolwarm)
        display = DISPLAY_NAMES.get(opt_name, opt_name)
        plot_3d_landscape(
            data["alphas"], data["betas"], data["loss_grid"],
            title=f"{display} — {args.cancer.upper()}",
            output_path=os.path.join(args.output_dir,
                                     f"landscape_3d_{opt_name}.pdf"),
            cmap=cmap,
        )

    # Comparison figure (2 rows x N cols)
    if len(landscapes) > 1:
        plot_3d_comparison(
            landscapes,
            os.path.join(args.output_dir, f"fig_loss_landscape_{args.cancer}.pdf"),
            cancer=args.cancer,
        )


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter-normalized loss landscape visualization",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Compute subcommand
    p_compute = subparsers.add_parser("compute", help="Compute landscapes (needs data)")
    p_compute.add_argument("--config", required=True)
    p_compute.add_argument("--checkpoint_dir", required=True)
    p_compute.add_argument("--optimizers", nargs="+",
                           default=["cosine_ge", "sam", "msam"])
    p_compute.add_argument("--output_dir", default="results_msam_paper_landscape/npz")
    p_compute.add_argument("--resolution", type=int, default=31)
    p_compute.add_argument("--range", type=float, default=1.0, dest="range_val")
    p_compute.add_argument("--seed", type=int, default=42)

    # Plot subcommand
    p_plot = subparsers.add_parser("plot", help="Plot from precomputed .npz files")
    p_plot.add_argument("--npz_dir", required=True)
    p_plot.add_argument("--optimizers", nargs="+",
                        default=["cosine_ge", "sam", "msam"])
    p_plot.add_argument("--output_dir", default="paper/msam_overleaf_sync/figures")
    p_plot.add_argument("--cancer", default="brca")

    args = parser.parse_args()

    if args.command == "compute":
        cmd_compute(args)
    elif args.command == "plot":
        cmd_plot(args)
    else:
        parser.print_help()
