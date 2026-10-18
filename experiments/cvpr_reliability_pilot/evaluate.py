"""
CVPR Reliability Pilot — Evaluation and Robustness Testing.

Evaluates trained models under:
  - Clean inputs
  - Gaussian noise on image embeddings (varying sigma)
  - Gaussian noise on gene features (varying sigma)
  - Missing image modality (zero embedding)
  - Missing gene modality (zero features)

Usage:
    python -m experiments.cvpr_reliability_pilot.evaluate \
        --config configs/brca.yaml \
        --results_dir results/cvpr_reliability_pilot/brca \
        --methods adamw aux_cox pcrf_aux
"""

import os, sys, json, argparse, copy
import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from train_survival import build_survival_dataloaders, concordance_index
from experiments.cvpr_reliability_pilot.models import build_pilot_model
from experiments.cvpr_reliability_pilot.losses import cox_loss


# ── Perturbation functions ───────────────────────────────────────────────────

def perturb_gaussian(tensor, sigma):
    """Add Gaussian noise to tensor."""
    return tensor + torch.randn_like(tensor) * sigma


def zero_modality(tensor):
    """Replace tensor with zeros (missing modality)."""
    return torch.zeros_like(tensor)


# ── Evaluation under perturbation ────────────────────────────────────────────

@torch.no_grad()
def evaluate_robustness(model, loader, device,
                        perturb_image=None, perturb_gene=None,
                        modality_mask=None):
    """
    Evaluate model with optional perturbations.

    Args:
        model: trained model
        loader: test DataLoader
        device: cuda/cpu
        perturb_image: None or callable(tensor) -> perturbed tensor
        perturb_gene: None or callable(tensor) -> perturbed tensor
        modality_mask: None or [2] mask (e.g., [1,0] = image only)
    Returns:
        (loss, c_index, n_patients)
    """
    model.eval()
    all_risk, all_time, all_event = [], [], []
    total_loss = 0.0

    for inputs, labels in loader:
        img = inputs["image"].to(device)
        gene = inputs["gene"].to(device)
        os_m = labels["os_months"].float().to(device)
        vs = labels["vital_status"].float().to(device)

        # Apply perturbations
        if perturb_image is not None:
            img = perturb_image(img)
        if perturb_gene is not None:
            gene = perturb_gene(gene)

        inp = {"image": img, "gene": gene}

        # Build modality mask
        mask = None
        if modality_mask is not None:
            B = img.size(0)
            mask = torch.tensor(modality_mask, device=device).float().unsqueeze(0).expand(B, -1)

        if mask is not None and hasattr(model, 'forward'):
            # Models that support modality_mask
            try:
                outputs = model(inp, modality_mask=mask)
            except TypeError:
                outputs = model(inp)
        else:
            outputs = model(inp)

        risk = outputs["survival"][:, 1]
        loss = cox_loss(risk, os_m, vs)
        total_loss += loss.item()

        all_risk.extend(risk.cpu().numpy().tolist())
        all_time.extend(os_m.cpu().numpy().tolist())
        all_event.extend(vs.cpu().numpy().tolist())

    ci = concordance_index(all_time, all_risk, all_event)
    return total_loss / max(len(loader), 1), ci, len(all_risk)


def run_robustness_suite(model, loader, device):
    """
    Run complete robustness evaluation suite.

    Returns dict of condition -> (loss, c_index)
    """
    results = {}

    # 1. Clean
    loss, ci, n = evaluate_robustness(model, loader, device)
    results["clean"] = {"loss": loss, "cindex": ci, "n_patients": n}
    print(f"    clean:              C-index = {ci:.4f}")

    # 2. Gaussian noise on image embeddings
    for sigma in [0.1, 0.3, 0.5, 1.0]:
        loss, ci, n = evaluate_robustness(
            model, loader, device,
            perturb_image=lambda t, s=sigma: perturb_gaussian(t, s),
        )
        key = f"noise_image_sigma{sigma}"
        results[key] = {"loss": loss, "cindex": ci, "sigma": sigma}
        print(f"    noise_image σ={sigma}: C-index = {ci:.4f}")

    # 3. Gaussian noise on gene features
    for sigma in [0.1, 0.3, 0.5, 1.0]:
        loss, ci, n = evaluate_robustness(
            model, loader, device,
            perturb_gene=lambda t, s=sigma: perturb_gaussian(t, s),
        )
        key = f"noise_gene_sigma{sigma}"
        results[key] = {"loss": loss, "cindex": ci, "sigma": sigma}
        print(f"    noise_gene  σ={sigma}: C-index = {ci:.4f}")

    # 4. Missing image modality
    loss, ci, n = evaluate_robustness(
        model, loader, device,
        perturb_image=zero_modality,
        modality_mask=[0, 1],
    )
    results["missing_image"] = {"loss": loss, "cindex": ci}
    print(f"    missing_image:      C-index = {ci:.4f}")

    # 5. Missing gene modality
    loss, ci, n = evaluate_robustness(
        model, loader, device,
        perturb_gene=zero_modality,
        modality_mask=[1, 0],
    )
    results["missing_gene"] = {"loss": loss, "cindex": ci}
    print(f"    missing_gene:       C-index = {ci:.4f}")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CVPR Pilot Robustness Evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cancer = cfg.get("cancer", "unknown")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Robustness Evaluation: {cancer} | device={device} ===")

    # Determine methods to evaluate
    if "all" in args.methods:
        from experiments.cvpr_reliability_pilot.models import ALL_METHODS
        methods = ALL_METHODS
    else:
        methods = args.methods

    all_robustness = {}

    for fold in range(args.folds):
        print(f"\n--- Fold {fold} ---")

        # Build data loaders
        loaders = build_survival_dataloaders(
            uni_dir=cfg["uni_dir"],
            gene_csv_paths=cfg["gene_csv_paths"],
            image_dim=cfg["image_dim"],
            n_top_genes=cfg.get("n_top_genes", 20000),
            batch_size=cfg.get("batch_size", 32),
            seed=args.seed,
            fold=fold,
            n_folds=args.folds,
        )

        n_genes = loaders["n_genes"]
        image_dim = loaders["image_dim"]

        for method in methods:
            fold_dir = os.path.join(args.results_dir, f"fold_{fold}")
            ckpt_path = os.path.join(fold_dir, f"{method}_best.pt")

            if not os.path.exists(ckpt_path):
                print(f"  [SKIP] {method} fold {fold}: no checkpoint")
                continue

            print(f"\n  {method} (fold {fold}):")

            # Build model and load checkpoint
            model, method_cfg = build_pilot_model(method, n_genes, image_dim)
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.to(device)

            # Run robustness suite
            results = run_robustness_suite(model, loaders["test"], device)

            key = f"{method}_fold{fold}"
            all_robustness[key] = results

    # Save robustness results
    out_path = os.path.join(args.results_dir, "robustness.json")
    with open(out_path, "w") as f:
        json.dump(all_robustness, f, indent=2)
    print(f"\nRobustness results saved: {out_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"ROBUSTNESS SUMMARY — {cancer}")
    print(f"{'='*70}")
    print(f"{'Method':18s} | {'Clean':7s} | {'Img σ=0.5':9s} | {'Gene σ=0.5':10s} | "
          f"{'No Image':8s} | {'No Gene':8s}")
    print("-" * 70)

    for method in methods:
        clean_cis = []
        img05_cis = []
        gene05_cis = []
        no_img_cis = []
        no_gene_cis = []

        for fold in range(args.folds):
            key = f"{method}_fold{fold}"
            if key in all_robustness:
                r = all_robustness[key]
                clean_cis.append(r.get("clean", {}).get("cindex", 0))
                img05_cis.append(r.get("noise_image_sigma0.5", {}).get("cindex", 0))
                gene05_cis.append(r.get("noise_gene_sigma0.5", {}).get("cindex", 0))
                no_img_cis.append(r.get("missing_image", {}).get("cindex", 0))
                no_gene_cis.append(r.get("missing_gene", {}).get("cindex", 0))

        if clean_cis:
            print(
                f"{method:18s} | {np.mean(clean_cis):.4f} | "
                f"{np.mean(img05_cis):.4f}    | "
                f"{np.mean(gene05_cis):.4f}     | "
                f"{np.mean(no_img_cis):.4f}   | "
                f"{np.mean(no_gene_cis):.4f}"
            )


if __name__ == "__main__":
    main()
