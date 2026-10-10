"""
visualize_msam.py — Generate tables and figures for M-SAM paper.

Reads results from TWO directories:
  - results_cindex_exp_b/<cancer>/fold_<i>/<opt>_survival.json  (reused baselines)
  - results_msam_paper/<cancer>/fold_<i>/<opt>_survival.json    (new M-SAM runs)

Generates:
  1. Table 1: Main results (7 opts x 5 cancers, C-index mean ± std)
  2. Table 2: Ablation (6 configs x 5 cancers)
  3. Table 3: Overfitting gap (best_val - test)
  4. Table 4: Sensitivity (rho and alpha sweeps)
  5. Figure: Val C-index training curves
  6. Figure: Shapley trajectories
  7. Figure: Overfitting gap bar chart

Usage:
    python visualize_msam.py --output_dir paper/figures
    python visualize_msam.py --output_dir paper/figures --latex
"""

import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict


# ── Configuration ─────────────────────────────────────────────────────────────

CANCERS = ["brca", "kirc", "luad", "lusc", "stad"]
N_FOLDS = 5

# Source directories
BASELINE_DIR = "results_cindex_exp_b"
MSAM_DIR = "results_msam_paper"
SENSITIVITY_DIR = "results_msam_paper_sensitivity"

# Optimizer display names and mapping
MAIN_OPTIMIZERS = {
    "sgd_mom": ("SGD+Mom", BASELINE_DIR),
    "adam": ("Adam", BASELINE_DIR),
    "cosine": ("Cosine", BASELINE_DIR),
    "cosine_ge": ("OGM-GE", MSAM_DIR),
    "sam": ("SAM", MSAM_DIR),
    "msam": ("M-SAM", MSAM_DIR),
    "msam_chaotic": ("M-SAM+Chaos", MSAM_DIR),
}

ABLATION_OPTIMIZERS = {
    "cosine": ("Cosine (base)", BASELINE_DIR),
    "cosine_ge": ("+ OGM-GE", MSAM_DIR),
    "sam": ("+ SAM (all)", MSAM_DIR),
    "msam_no_ogm": ("+ Shapley+SAM", MSAM_DIR),
    "msam_uniform": ("+ OGM-GE+SAM(rand)", MSAM_DIR),
    "msam": ("M-SAM (full)", MSAM_DIR),
}

PALETTE = {
    "sgd_mom": "#666666",
    "adam": "#4393c3",
    "cosine": "#2166ac",
    "cosine_ge": "#b2182b",
    "sam": "#f4a582",
    "msam": "#1a9850",
    "msam_chaotic": "#762a83",
    "msam_no_ogm": "#e08214",
    "msam_uniform": "#8073ac",
    "chaotic_adaptive": "#d6604d",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_result(cancer, fold, opt_name, source_dir):
    """Load a single result JSON. Returns None if not found."""
    path = os.path.join(source_dir, cancer, f"fold_{fold}",
                        f"{opt_name}_survival.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_all_folds(cancer, opt_name, source_dir, n_folds=N_FOLDS):
    """Load all folds for one cancer/optimizer. Returns list of dicts."""
    results = []
    for fold in range(n_folds):
        r = load_result(cancer, fold, opt_name, source_dir)
        if r is not None:
            results.append(r)
    return results


def get_cindex_stats(cancer, opt_name, source_dir, n_folds=N_FOLDS):
    """Get mean and std of test C-index across folds."""
    results = load_all_folds(cancer, opt_name, source_dir, n_folds)
    if not results:
        return None, None
    cindices = [r["test_cindex"] for r in results]
    return np.mean(cindices), np.std(cindices)


# ── Table 1: Main Results ────────────────────────────────────────────────────

def generate_main_table(output_dir, latex=False):
    """Generate Table 1: Main comparison across all cancers."""
    print("\n" + "=" * 80)
    print("TABLE 1: Main Results (Test C-index, 5-fold CV)")
    print("=" * 80)

    # Header
    header = f"{'Optimizer':<18}"
    for cancer in CANCERS:
        header += f" {cancer.upper():>14}"
    header += f" {'Mean':>10}"
    print(header)
    print("-" * 80)

    # Data collection for LaTeX
    all_data = {}
    best_per_cancer = {}

    for opt_name, (display, src_dir) in MAIN_OPTIMIZERS.items():
        row_means = []
        row = f"{display:<18}"
        for cancer in CANCERS:
            mean, std = get_cindex_stats(cancer, opt_name, src_dir)
            if mean is not None:
                row += f" {mean:.3f}±{std:.3f}"
                row_means.append(mean)
                if cancer not in best_per_cancer or mean > best_per_cancer[cancer][0]:
                    best_per_cancer[cancer] = (mean, opt_name)
            else:
                row += f" {'—':>14}"
        if row_means:
            row += f" {np.mean(row_means):>10.4f}"
        print(row)
        all_data[opt_name] = row_means

    print("-" * 80)
    print(f"Best per cancer: {best_per_cancer}")

    # LaTeX output
    if latex:
        _write_latex_table(all_data, MAIN_OPTIMIZERS, output_dir,
                           "table1_main.tex", best_per_cancer)

    return all_data


# ── Table 2: Ablation ────────────────────────────────────────────────────────

def generate_ablation_table(output_dir, latex=False):
    """Generate Table 2: Ablation study."""
    print("\n" + "=" * 80)
    print("TABLE 2: Ablation Study (Test C-index, 5-fold CV)")
    print("=" * 80)

    header = f"{'Config':<22}"
    for cancer in CANCERS:
        header += f" {cancer.upper():>14}"
    print(header)
    print("-" * 80)

    all_data = {}
    for opt_name, (display, src_dir) in ABLATION_OPTIMIZERS.items():
        row = f"{display:<22}"
        row_means = []
        for cancer in CANCERS:
            mean, std = get_cindex_stats(cancer, opt_name, src_dir)
            if mean is not None:
                row += f" {mean:.3f}±{std:.3f}"
                row_means.append(mean)
            else:
                row += f" {'—':>14}"
        print(row)
        all_data[opt_name] = row_means

    if latex:
        _write_latex_table(all_data, ABLATION_OPTIMIZERS, output_dir,
                           "table2_ablation.tex", {})


# ── Table 3: Overfitting Gap ─────────────────────────────────────────────────

def generate_overfitting_table(output_dir, latex=False):
    """Generate Table 3: Overfitting gap (best_val - test)."""
    print("\n" + "=" * 80)
    print("TABLE 3: Overfitting Gap (best_val_cindex - test_cindex)")
    print("=" * 80)

    header = f"{'Optimizer':<18}"
    for cancer in CANCERS:
        header += f" {cancer.upper():>10}"
    header += f" {'Mean':>10}"
    print(header)
    print("-" * 80)

    for opt_name, (display, src_dir) in MAIN_OPTIMIZERS.items():
        row = f"{display:<18}"
        gaps = []
        for cancer in CANCERS:
            results = load_all_folds(cancer, opt_name, src_dir)
            if results:
                fold_gaps = [r["best_val_cindex"] - r["test_cindex"]
                             for r in results]
                mean_gap = np.mean(fold_gaps)
                row += f" {mean_gap:>+10.4f}"
                gaps.append(mean_gap)
            else:
                row += f" {'—':>10}"
        if gaps:
            row += f" {np.mean(gaps):>+10.4f}"
        print(row)


# ── Table 4: Sensitivity ─────────────────────────────────────────────────────

def generate_sensitivity_table(output_dir, latex=False):
    """Generate Table 4: Sensitivity to rho and alpha."""
    print("\n" + "=" * 80)
    print("TABLE 4: Sensitivity Analysis (M-SAM, fold 0)")
    print("=" * 80)

    sens_cancers = ["brca", "kirc", "luad"]
    rhos = [0.01, 0.02, 0.05, 0.10, 0.20]
    alphas = [0.1, 0.25, 0.5, 1.0]

    # Rho sweep
    print(f"\n{'Rho sweep':}")
    header = f"{'rho':<8}"
    for cancer in sens_cancers:
        header += f" {cancer.upper():>10}"
    print(header)
    print("-" * 40)

    for rho in rhos:
        row = f"{rho:<8.2f}"
        for cancer in sens_cancers:
            rho_str = f"{rho:.2f}"
            path = os.path.join(SENSITIVITY_DIR, cancer, f"rho_{rho_str}",
                                "fold_0", "msam_survival.json")
            if os.path.exists(path):
                with open(path) as f:
                    r = json.load(f)
                row += f" {r['test_cindex']:>10.4f}"
            else:
                row += f" {'—':>10}"
        print(row)

    # Alpha sweep
    print(f"\n{'Alpha sweep':}")
    header = f"{'alpha':<8}"
    for cancer in sens_cancers:
        header += f" {cancer.upper():>10}"
    print(header)
    print("-" * 40)

    for alpha in alphas:
        row = f"{alpha:<8.2f}"
        for cancer in sens_cancers:
            path = os.path.join(SENSITIVITY_DIR, cancer, f"alpha_{alpha}",
                                "fold_0", "msam_survival.json")
            if os.path.exists(path):
                with open(path) as f:
                    r = json.load(f)
                row += f" {r['test_cindex']:>10.4f}"
            else:
                row += f" {'—':>10}"
        print(row)


# ── Figure: Training Curves ──────────────────────────────────────────────────

def plot_training_curves(output_dir):
    """Plot val C-index over epochs for top methods, per cancer."""
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)

    plot_opts = ["cosine", "cosine_ge", "sam", "msam"]

    for i, cancer in enumerate(CANCERS):
        ax = axes[i]
        for opt_name in plot_opts:
            _, src_dir = MAIN_OPTIMIZERS.get(opt_name, (opt_name, MSAM_DIR))
            results = load_all_folds(cancer, opt_name, src_dir)
            if not results:
                continue

            # Average val_cindex across folds
            max_len = max(len(r["val_cindex"]) for r in results)
            curves = np.full((len(results), max_len), np.nan)
            for j, r in enumerate(results):
                curves[j, :len(r["val_cindex"])] = r["val_cindex"]

            mean_curve = np.nanmean(curves, axis=0)
            std_curve = np.nanstd(curves, axis=0)

            display = MAIN_OPTIMIZERS.get(opt_name, (opt_name, ""))[0]
            color = PALETTE.get(opt_name, "#333")
            ax.plot(mean_curve, label=display, color=color, linewidth=2)
            ax.fill_between(range(len(mean_curve)),
                            mean_curve - std_curve,
                            mean_curve + std_curve,
                            alpha=0.15, color=color)

        ax.set_title(cancer.upper(), fontsize=13)
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel("Validation C-index")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")

    plt.suptitle("Validation C-index Over Training (5-fold CV mean ± std)",
                 fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "fig_training_curves.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Figure: Shapley Trajectories ─────────────────────────────────────────────

def plot_shapley_trajectories(output_dir):
    """Plot image vs gene Shapley contributions over training."""
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)

    for i, cancer in enumerate(CANCERS):
        ax = axes[i]
        results = load_all_folds(cancer, "msam", MSAM_DIR)

        if not results:
            ax.set_title(f"{cancer.upper()} (no data)")
            continue

        # Aggregate Shapley histories
        for j, r in enumerate(results):
            shapley = r.get("shapley_history", {})
            if "image" in shapley and "gene" in shapley:
                img_vals = shapley["image"]
                gene_vals = shapley["gene"]
                alpha_val = 0.3 if len(results) > 1 else 1.0
                ax.plot(img_vals, color="#d73027", alpha=alpha_val, linewidth=0.5)
                ax.plot(gene_vals, color="#4575b4", alpha=alpha_val, linewidth=0.5)

        # Plot mean if multiple folds
        if len(results) > 1:
            all_img = [r.get("shapley_history", {}).get("image", [])
                       for r in results]
            all_gene = [r.get("shapley_history", {}).get("gene", [])
                        for r in results]
            if all_img and all_img[0]:
                min_len = min(len(x) for x in all_img if x)
                img_mean = np.mean([x[:min_len] for x in all_img if x], axis=0)
                gene_mean = np.mean([x[:min_len] for x in all_gene if x], axis=0)
                ax.plot(img_mean, color="#d73027", linewidth=2.5, label="Image")
                ax.plot(gene_mean, color="#4575b4", linewidth=2.5, label="Gene")

        ax.set_title(cancer.upper(), fontsize=13)
        ax.set_xlabel("Training Step")
        if i == 0:
            ax.set_ylabel("Shapley Contribution")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("Shapley Value Trajectories (Image vs Gene)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "fig_shapley_trajectories.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Figure: Overfitting Gap Bar Chart ────────────────────────────────────────

def plot_overfitting_gap(output_dir):
    """Grouped bar chart of overfitting gaps."""
    plot_opts = ["cosine", "cosine_ge", "sam", "msam"]
    n_cancers = len(CANCERS)
    n_opts = len(plot_opts)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_cancers)
    width = 0.8 / n_opts

    for i, opt_name in enumerate(plot_opts):
        _, src_dir = MAIN_OPTIMIZERS.get(opt_name, (opt_name, MSAM_DIR))
        gaps = []
        for cancer in CANCERS:
            results = load_all_folds(cancer, opt_name, src_dir)
            if results:
                fold_gaps = [r["best_val_cindex"] - r["test_cindex"]
                             for r in results]
                gaps.append(np.mean(fold_gaps))
            else:
                gaps.append(0)

        display = MAIN_OPTIMIZERS.get(opt_name, (opt_name, ""))[0]
        color = PALETTE.get(opt_name, "#333")
        ax.bar(x + i * width - 0.4 + width / 2, gaps, width,
               label=display, color=color, alpha=0.8)

    ax.set_xlabel("Cancer Type")
    ax.set_ylabel("Overfitting Gap (val - test C-index)")
    ax.set_title("Generalization Gap: Lower = Better Generalization")
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in CANCERS])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig_overfitting_gap.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── LaTeX table helper ────────────────────────────────────────────────────────

def _write_latex_table(all_data, opt_map, output_dir, filename, best_map):
    """Write LaTeX table source."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{}")
    cols = "l" + "c" * len(CANCERS) + "c"
    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\toprule")

    # Header
    header = "Method"
    for c in CANCERS:
        header += f" & {c.upper()}"
    header += r" & Mean \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Find best per cancer
    best_vals = {}
    for cancer_idx, cancer in enumerate(CANCERS):
        best_ci = -1
        for opt_name, (display, src_dir) in opt_map.items():
            mean, _ = get_cindex_stats(cancer, opt_name, src_dir)
            if mean is not None and mean > best_ci:
                best_ci = mean
                best_vals[cancer] = (opt_name, mean)

    # Rows
    for opt_name, (display, src_dir) in opt_map.items():
        row = display
        row_means = []
        for cancer in CANCERS:
            mean, std = get_cindex_stats(cancer, opt_name, src_dir)
            if mean is not None:
                val_str = f"{mean:.3f}$\\pm${std:.3f}"
                if cancer in best_vals and best_vals[cancer][0] == opt_name:
                    val_str = r"\textbf{" + val_str + "}"
                row += f" & {val_str}"
                row_means.append(mean)
            else:
                row += " & ---"
        if row_means:
            row += f" & {np.mean(row_means):.4f}"
        else:
            row += " & ---"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved LaTeX: {path}")


# ── Dataset statistics table ─────────────────────────────────────────────────

def generate_dataset_table():
    """Print Table 5: Dataset statistics."""
    print("\n" + "=" * 80)
    print("TABLE 5: Dataset Statistics")
    print("=" * 80)

    stats = {
        "BRCA": {"patients": 1123, "events": "~15%", "dim": 1024, "encoder": "UNI"},
        "KIRC": {"patients": 519, "events": "~33%", "dim": 1536, "encoder": "UNI2-h"},
        "LUAD": {"patients": 522, "events": "~36%", "dim": 1536, "encoder": "UNI2-h"},
        "LUSC": {"patients": 500, "events": "~40%", "dim": 1536, "encoder": "UNI2-h"},
        "STAD": {"patients": 367, "events": "~35%", "dim": 1536, "encoder": "UNI2-h"},
    }

    print(f"{'Cancer':<8} {'Patients':>10} {'Events':>8} "
          f"{'Img Dim':>8} {'Encoder':>8}")
    print("-" * 50)
    for cancer, s in stats.items():
        print(f"{cancer:<8} {s['patients']:>10} {s['events']:>8} "
              f"{s['dim']:>8} {s['encoder']:>8}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate M-SAM paper tables and figures",
    )
    parser.add_argument("--output_dir", default="paper/figures")
    parser.add_argument("--latex", action="store_true",
                        help="Also generate LaTeX table source files")
    parser.add_argument("--tables_only", action="store_true")
    parser.add_argument("--figures_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.figures_only:
        generate_main_table(args.output_dir, latex=args.latex)
        generate_ablation_table(args.output_dir, latex=args.latex)
        generate_overfitting_table(args.output_dir, latex=args.latex)
        generate_sensitivity_table(args.output_dir, latex=args.latex)
        generate_dataset_table()

    if not args.tables_only:
        plot_training_curves(args.output_dir)
        plot_shapley_trajectories(args.output_dir)
        plot_overfitting_gap(args.output_dir)

    print("\nDone! All outputs in:", args.output_dir)
