"""
CVPR Reliability Pilot — Statistical Analysis and Visualization.

Generates:
  - Per-method, per-cancer C-index tables
  - Paired t-tests against baselines
  - Robustness degradation curves
  - Reliability weight distributions
  - Reliability collapse diagnostics
  - Publication-quality figures

Usage:
    python -m experiments.cvpr_reliability_pilot.analyze_results \
        --results_base results/cvpr_reliability_pilot \
        --cancers brca kirc luad --folds 3
"""

import os, sys, json, argparse
import numpy as np
from scipy import stats
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)


def load_results(results_base, cancers, folds, methods=None):
    """Load all experiment results into a structured dict."""
    data = {}
    for cancer in cancers:
        data[cancer] = {}
        for fold in range(folds):
            fold_dir = os.path.join(results_base, cancer, f"fold_{fold}")
            if not os.path.isdir(fold_dir):
                continue
            for fname in os.listdir(fold_dir):
                if fname.endswith(".json") and not fname.startswith("manifest"):
                    method = fname.replace(".json", "")
                    if methods and method not in methods:
                        continue
                    path = os.path.join(fold_dir, fname)
                    with open(path) as f:
                        result = json.load(f)
                    if method not in data[cancer]:
                        data[cancer][method] = {}
                    data[cancer][method][fold] = result
    return data


def compute_summary(data, folds):
    """Compute per-method, per-cancer summary statistics."""
    summary = {}
    for cancer, methods in data.items():
        summary[cancer] = {}
        for method, fold_data in methods.items():
            test_cis = [
                fold_data[f]["test_cindex"]
                for f in range(folds)
                if f in fold_data and "test_cindex" in fold_data[f]
            ]
            val_cis = [
                fold_data[f]["best_val_cindex"]
                for f in range(folds)
                if f in fold_data and "best_val_cindex" in fold_data[f]
            ]
            best_epochs = [
                fold_data[f]["best_epoch"]
                for f in range(folds)
                if f in fold_data and "best_epoch" in fold_data[f]
            ]
            elapsed = [
                fold_data[f].get("elapsed_seconds", 0)
                for f in range(folds)
                if f in fold_data
            ]

            if test_cis:
                summary[cancer][method] = {
                    "test_cindex_mean": np.mean(test_cis),
                    "test_cindex_std": np.std(test_cis),
                    "val_cindex_mean": np.mean(val_cis),
                    "val_cindex_std": np.std(val_cis),
                    "test_cindex_per_fold": test_cis,
                    "val_cindex_per_fold": val_cis,
                    "best_epoch_mean": np.mean(best_epochs),
                    "elapsed_mean": np.mean(elapsed),
                    "n_folds": len(test_cis),
                    "val_test_gap": np.mean(val_cis) - np.mean(test_cis),
                }
    return summary


def paired_tests(data, folds, baseline="adamw"):
    """Paired t-tests for each method vs baseline on each cancer."""
    results = {}
    for cancer, methods in data.items():
        results[cancer] = {}
        if baseline not in methods:
            continue
        base_cis = [
            methods[baseline][f]["test_cindex"]
            for f in range(folds)
            if f in methods[baseline]
        ]
        for method, fold_data in methods.items():
            if method == baseline:
                continue
            method_cis = [
                fold_data[f]["test_cindex"]
                for f in range(folds)
                if f in fold_data
            ]
            if len(method_cis) == len(base_cis) and len(method_cis) >= 2:
                diffs = [m - b for m, b in zip(method_cis, base_cis)]
                t_stat, p_val = stats.ttest_rel(method_cis, base_cis)
                results[cancer][method] = {
                    "delta": np.mean(diffs),
                    "t_statistic": t_stat,
                    "p_value": p_val,
                    "significant": p_val < 0.05,
                    "per_fold_diffs": diffs,
                }
    return results


def plot_performance_comparison(summary, save_dir):
    """Bar plot of test C-index per method per cancer."""
    cancers = sorted(summary.keys())
    methods_set = set()
    for c in cancers:
        methods_set.update(summary[c].keys())
    methods = sorted(methods_set)

    fig, axes = plt.subplots(1, len(cancers), figsize=(5 * len(cancers), 5), sharey=True)
    if len(cancers) == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    for ax, cancer in zip(axes, cancers):
        x = np.arange(len(methods))
        means = [summary[cancer].get(m, {}).get("test_cindex_mean", 0) for m in methods]
        stds = [summary[cancer].get(m, {}).get("test_cindex_std", 0) for m in methods]

        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black",
                      linewidth=0.5, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax.set_title(cancer.upper(), fontweight="bold")
        ax.set_ylabel("Test C-index" if cancer == cancers[0] else "")
        ax.set_ylim(0.45, 0.85)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "performance_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_robustness_curves(results_base, cancers, save_dir):
    """Plot C-index degradation under increasing noise."""
    fig, axes = plt.subplots(2, len(cancers), figsize=(5 * len(cancers), 8))
    if len(cancers) == 1:
        axes = axes.reshape(-1, 1)

    for col, cancer in enumerate(cancers):
        rob_path = os.path.join(results_base, cancer, "robustness.json")
        if not os.path.exists(rob_path):
            continue

        with open(rob_path) as f:
            rob_data = json.load(f)

        # Collect per-method robustness
        methods_data = defaultdict(lambda: defaultdict(list))
        for key, results in rob_data.items():
            parts = key.rsplit("_fold", 1)
            if len(parts) != 2:
                continue
            method = parts[0]
            for cond, vals in results.items():
                if isinstance(vals, dict) and "cindex" in vals:
                    methods_data[method][cond].append(vals["cindex"])

        sigmas = [0.0, 0.1, 0.3, 0.5, 1.0]

        for method, cond_data in methods_data.items():
            # Image noise curve
            img_cis = [np.mean(cond_data.get("clean", [0.5]))]
            for s in sigmas[1:]:
                key = f"noise_image_sigma{s}"
                img_cis.append(np.mean(cond_data.get(key, [0.5])))
            axes[0, col].plot(sigmas, img_cis, marker="o", label=method,
                            markersize=4, linewidth=1.5)

            # Gene noise curve
            gene_cis = [np.mean(cond_data.get("clean", [0.5]))]
            for s in sigmas[1:]:
                key = f"noise_gene_sigma{s}"
                gene_cis.append(np.mean(cond_data.get(key, [0.5])))
            axes[1, col].plot(sigmas, gene_cis, marker="s", label=method,
                            markersize=4, linewidth=1.5)

        for row in range(2):
            axes[row, col].set_xlabel("Noise σ")
            axes[row, col].axhline(0.5, color="gray", linestyle="--", alpha=0.3)
            axes[row, col].set_ylim(0.4, 0.8)
            if col == 0:
                label = "Image noise" if row == 0 else "Gene noise"
                axes[row, col].set_ylabel(f"Test C-index ({label})")
            axes[row, col].set_title(
                f"{cancer.upper()} — {'Image' if row == 0 else 'Gene'} Noise"
            )
            axes[row, col].legend(fontsize=6)

    plt.tight_layout()
    path = os.path.join(save_dir, "robustness_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_reliability_weights(data, folds, save_dir):
    """Plot reliability weight distributions for PCRF methods."""
    pcrf_methods = ["learned_gating", "pcrf", "pcrf_aux"]

    for cancer, methods in data.items():
        fig, axes = plt.subplots(1, len(pcrf_methods), figsize=(4 * len(pcrf_methods), 4))
        if len(pcrf_methods) == 1:
            axes = [axes]

        for ax, method in zip(axes, pcrf_methods):
            if method not in methods:
                ax.set_title(f"{method}\n(not found)")
                continue

            all_img_weights = []
            for fold in range(folds):
                if fold in methods[method]:
                    rel = methods[method][fold].get("test_reliability", {})
                    if rel:
                        ax.bar(
                            [f"Fold {fold}\nImg", f"Fold {fold}\nGene"],
                            [rel.get("alpha_img_mean", 0.5),
                             rel.get("alpha_gene_mean", 0.5)],
                            yerr=[rel.get("alpha_img_std", 0),
                                  rel.get("alpha_gene_std", 0)],
                            capsize=3, alpha=0.7,
                        )

            ax.set_title(method)
            ax.set_ylabel("Reliability weight")
            ax.set_ylim(0, 1)
            ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)

        plt.suptitle(f"{cancer.upper()} — Reliability Weights", fontweight="bold")
        plt.tight_layout()
        path = os.path.join(save_dir, f"reliability_weights_{cancer}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


def generate_report(summary, paired, data, folds, save_dir):
    """Generate markdown report."""
    lines = ["# CVPR Reliability Pilot — Results Summary\n"]
    lines.append(f"**Generated:** {__import__('datetime').datetime.now().isoformat()}\n")

    for cancer in sorted(summary.keys()):
        lines.append(f"\n## {cancer.upper()}\n")

        # Performance table
        lines.append("### Test C-index (mean ± std)\n")
        lines.append("| Method | Test C-index | Val C-index | Best Epoch | Gap | Time |")
        lines.append("|--------|-------------|-------------|------------|-----|------|")

        methods_sorted = sorted(
            summary[cancer].items(),
            key=lambda x: x[1]["test_cindex_mean"],
            reverse=True,
        )
        for method, s in methods_sorted:
            lines.append(
                f"| {method} | "
                f"{s['test_cindex_mean']:.4f} ± {s['test_cindex_std']:.4f} | "
                f"{s['val_cindex_mean']:.4f} ± {s['val_cindex_std']:.4f} | "
                f"{s['best_epoch_mean']:.0f} | "
                f"{s['val_test_gap']:.3f} | "
                f"{s['elapsed_mean']:.0f}s |"
            )

        # Per-fold table
        lines.append(f"\n### Per-Fold Test C-index\n")
        lines.append("| Method | " + " | ".join(f"Fold {f}" for f in range(folds)) + " |")
        lines.append("|--------" + "|-------" * folds + "|")
        for method, s in methods_sorted:
            fold_cis = s.get("test_cindex_per_fold", [])
            fold_str = " | ".join(f"{c:.4f}" for c in fold_cis)
            lines.append(f"| {method} | {fold_str} |")

        # Statistical tests
        if cancer in paired:
            lines.append(f"\n### Paired t-tests vs AdamW\n")
            lines.append("| Method | Δ C-index | t-stat | p-value | Sig? |")
            lines.append("|--------|-----------|--------|---------|------|")
            for method, t in sorted(paired[cancer].items()):
                sig = "Yes" if t["significant"] else "No"
                lines.append(
                    f"| {method} | {t['delta']:+.4f} | "
                    f"{t['t_statistic']:.3f} | {t['p_value']:.4f} | {sig} |"
                )

    report = "\n".join(lines)
    path = os.path.join(save_dir, "results_summary.md")
    with open(path, "w") as f:
        f.write(report)
    print(f"  Saved: {path}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_base", required=True)
    parser.add_argument("--cancers", nargs="+", default=["brca", "kirc", "luad"])
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()

    print("=== CVPR Pilot Analysis ===")

    # Load all results
    data = load_results(args.results_base, args.cancers, args.folds)

    # Compute summary
    summary = compute_summary(data, args.folds)

    # Statistical tests
    paired = paired_tests(data, args.folds, baseline="adamw")

    # Print summary
    for cancer in args.cancers:
        if cancer in summary:
            print(f"\n{cancer.upper()}:")
            for method, s in sorted(
                summary[cancer].items(),
                key=lambda x: x[1]["test_cindex_mean"],
                reverse=True,
            ):
                print(
                    f"  {method:18s}: "
                    f"{s['test_cindex_mean']:.4f}±{s['test_cindex_std']:.4f}  "
                    f"gap={s['val_test_gap']:.3f}"
                )

    # Generate figures
    fig_dir = os.path.join(args.results_base, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("\nGenerating figures...")
    plot_performance_comparison(summary, fig_dir)
    plot_robustness_curves(args.results_base, args.cancers, fig_dir)
    plot_reliability_weights(data, args.folds, fig_dir)

    # Generate report
    print("\nGenerating report...")
    generate_report(summary, paired, data, args.folds, args.results_base)

    # Save full analysis data
    analysis = {
        "summary": {
            cancer: {
                method: {k: v for k, v in s.items()
                        if not isinstance(v, np.floating)}
                for method, s in methods.items()
            }
            for cancer, methods in summary.items()
        },
        "paired_tests": paired,
    }
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    import json

    def json_serialize(obj):
        if isinstance(obj, dict):
            return {k: json_serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [json_serialize(v) for v in obj]
        return convert(obj)

    path = os.path.join(args.results_base, "analysis.json")
    with open(path, "w") as f:
        json.dump(json_serialize(analysis), f, indent=2)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
