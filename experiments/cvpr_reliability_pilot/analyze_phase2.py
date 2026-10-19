"""
Phase 2 analysis: Isolating reliability supervision effect.

Compares:
  - learned_gating_aux (architecture-matched baseline, no reliability)
  - pcrf_aux (full method with reliability)
  - pcrf_aux_warmup (curriculum reliability: 20-epoch warmup)

Generates:
  - PHASE2_RESULTS.csv
  - Comparison tables and paired tests
  - Reliability weight analysis
  - Figures
"""

import json, os, sys, csv
from datetime import datetime
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

PHASE2_METHODS = ["learned_gating_aux", "pcrf_aux", "pcrf_aux_warmup"]
CANCERS = ["brca", "kirc", "luad"]


def load_phase2_results(results_base, n_folds=3):
    """Load all Phase 2 results."""
    data = {}
    for cancer in CANCERS:
        data[cancer] = {}
        for method in PHASE2_METHODS:
            folds = []
            for fold in range(n_folds):
                path = os.path.join(results_base, cancer, f"fold_{fold}", f"{method}.json")
                if os.path.exists(path):
                    with open(path) as f:
                        folds.append(json.load(f))
            if folds:
                data[cancer][method] = folds
    return data


def paired_t_test(a, b):
    """Paired t-test. Returns (t_stat, p_value, mean_diff, ci_95)."""
    from scipy import stats
    a, b = np.array(a), np.array(b)
    diff = a - b
    n = len(diff)
    if n < 2:
        return 0.0, 1.0, 0.0, (0.0, 0.0)
    mean_d = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    if se < 1e-12:
        return float('inf') if mean_d > 0 else float('-inf'), 0.0, mean_d, (mean_d, mean_d)
    t = mean_d / se
    p = stats.t.sf(abs(t), n - 1) * 2
    t_crit = stats.t.ppf(0.975, n - 1)
    ci = (mean_d - t_crit * se, mean_d + t_crit * se)
    return float(t), float(p), float(mean_d), (float(ci[0]), float(ci[1]))


def analyze(results_base, n_folds=3, output_dir=None):
    """Run full Phase 2 analysis."""
    if output_dir is None:
        output_dir = results_base

    data = load_phase2_results(results_base, n_folds)

    report_lines = []
    csv_rows = []

    report_lines.append("# CVPR Reliability Pilot — Phase 2 Results\n")
    report_lines.append(f"**Generated:** {datetime.now().isoformat()}\n")
    report_lines.append(f"**Methods:** {', '.join(PHASE2_METHODS)}\n")
    report_lines.append(f"**Cancers:** {', '.join(c.upper() for c in CANCERS)}\n")
    report_lines.append(f"**Folds:** {n_folds}\n\n")

    # Per-cancer analysis
    for cancer in CANCERS:
        cancer_data = data.get(cancer, {})
        if not cancer_data:
            report_lines.append(f"\n## {cancer.upper()}: No results\n")
            continue

        report_lines.append(f"\n## {cancer.upper()}\n")

        # Summary table
        report_lines.append("### Test C-index (mean ± std)\n")
        report_lines.append("| Method | Test C-index | Val C-index | Best Epoch | Gap | Time | Warmup |\n")
        report_lines.append("|--------|-------------|-------------|------------|-----|------|--------|\n")

        method_cis = {}
        for method in PHASE2_METHODS:
            folds = cancer_data.get(method, [])
            if not folds:
                continue
            test_cis = [f["test_cindex"] for f in folds]
            val_cis = [f["best_val_cindex"] for f in folds]
            epochs = [f["best_epoch"] for f in folds]
            times = [f.get("elapsed_seconds", 0) for f in folds]
            warmup = folds[0].get("warmup_epochs", 0)

            method_cis[method] = test_cis

            mean_t = np.mean(test_cis)
            std_t = np.std(test_cis)
            mean_v = np.mean(val_cis)
            gap = mean_v - mean_t
            mean_ep = np.mean(epochs)
            mean_time = np.mean(times)

            report_lines.append(
                f"| {method} | {mean_t:.4f} ± {std_t:.4f} | "
                f"{mean_v:.4f} | {mean_ep:.0f} | {gap:.3f} | {mean_time:.0f}s | "
                f"{warmup} |\n"
            )

            csv_rows.append({
                "cancer": cancer.upper(),
                "method": method,
                "test_cindex_mean": f"{mean_t:.4f}",
                "test_cindex_std": f"{std_t:.4f}",
                "val_cindex_mean": f"{mean_v:.4f}",
                "gap": f"{gap:.3f}",
                "best_epoch_mean": f"{mean_ep:.0f}",
                "time_mean_s": f"{mean_time:.0f}",
                "warmup_epochs": str(warmup),
                **{f"fold_{i}": f"{test_cis[i]:.4f}" for i in range(len(test_cis))},
            })

        # Per-fold table
        report_lines.append("\n### Per-Fold Test C-index\n")
        report_lines.append("| Method | " + " | ".join(f"Fold {i}" for i in range(n_folds)) + " |\n")
        report_lines.append("|--------" + "|-------" * n_folds + "|\n")
        for method in PHASE2_METHODS:
            folds = cancer_data.get(method, [])
            if folds:
                cis = [f["test_cindex"] for f in folds]
                report_lines.append(f"| {method} | " + " | ".join(f"{c:.4f}" for c in cis) + " |\n")

        # Paired tests: pcrf_aux vs learned_gating_aux (THE critical comparison)
        report_lines.append("\n### Paired Comparisons\n")

        if "learned_gating_aux" in method_cis and "pcrf_aux" in method_cis:
            a = method_cis["pcrf_aux"]
            b = method_cis["learned_gating_aux"]
            t, p, d, ci = paired_t_test(a, b)
            sig = "**Yes**" if p < 0.05 else "No"
            report_lines.append(
                f"**PCRF+Aux vs Gating+Aux (reliability effect):**\n"
                f"- Δ C-index = {d:+.4f} (95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}])\n"
                f"- t = {t:.3f}, p = {p:.4f}, Significant: {sig}\n\n"
            )

        if "learned_gating_aux" in method_cis and "pcrf_aux_warmup" in method_cis:
            a = method_cis["pcrf_aux_warmup"]
            b = method_cis["learned_gating_aux"]
            t, p, d, ci = paired_t_test(a, b)
            sig = "**Yes**" if p < 0.05 else "No"
            report_lines.append(
                f"**PCRF+Aux+Warmup vs Gating+Aux (curriculum reliability effect):**\n"
                f"- Δ C-index = {d:+.4f} (95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}])\n"
                f"- t = {t:.3f}, p = {p:.4f}, Significant: {sig}\n\n"
            )

        if "pcrf_aux" in method_cis and "pcrf_aux_warmup" in method_cis:
            a = method_cis["pcrf_aux_warmup"]
            b = method_cis["pcrf_aux"]
            t, p, d, ci = paired_t_test(a, b)
            sig = "**Yes**" if p < 0.05 else "No"
            report_lines.append(
                f"**Warmup vs No-Warmup (curriculum effect):**\n"
                f"- Δ C-index = {d:+.4f} (95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}])\n"
                f"- t = {t:.3f}, p = {p:.4f}, Significant: {sig}\n\n"
            )

        # Reliability weights
        report_lines.append("\n### Reliability Weights (test set)\n")
        report_lines.append("| Method | Img Weight | Gene Weight |\n")
        report_lines.append("|--------|-----------|------------|\n")
        for method in PHASE2_METHODS:
            folds = cancer_data.get(method, [])
            if folds:
                img_ws = [f["test_reliability"]["alpha_img_mean"]
                         for f in folds if f.get("test_reliability")]
                gene_ws = [f["test_reliability"]["alpha_gene_mean"]
                          for f in folds if f.get("test_reliability")]
                if img_ws:
                    report_lines.append(
                        f"| {method} | {np.mean(img_ws):.3f} ± {np.std(img_ws):.3f} | "
                        f"{np.mean(gene_ws):.3f} ± {np.std(gene_ws):.3f} |\n"
                    )

    # Cross-cancer summary
    report_lines.append("\n\n## Cross-Cancer Summary\n")
    report_lines.append("| Cancer | Gating+Aux | PCRF+Aux | PCRF+Aux+Warmup | Best | Δ_reliability | p |\n")
    report_lines.append("|--------|-----------|---------|----------------|------|---------------|---|\n")

    wins = 0
    for cancer in CANCERS:
        cancer_data = data.get(cancer, {})
        row = f"| {cancer.upper()} "
        cis = {}
        for method in PHASE2_METHODS:
            folds = cancer_data.get(method, [])
            if folds:
                mean = np.mean([f["test_cindex"] for f in folds])
                cis[method] = mean
                row += f"| {mean:.4f} "
            else:
                row += "| — "

        if cis:
            best = max(cis, key=cis.get)
            row += f"| {best} "

            if "learned_gating_aux" in cis and "pcrf_aux" in cis:
                delta = cis["pcrf_aux"] - cis["learned_gating_aux"]
                a = [f["test_cindex"] for f in cancer_data["pcrf_aux"]]
                b = [f["test_cindex"] for f in cancer_data["learned_gating_aux"]]
                _, p, _, _ = paired_t_test(a, b)
                row += f"| {delta:+.4f} | {p:.4f} "
                if delta > 0:
                    wins += 1
            else:
                row += "| — | — "
        else:
            row += "| — | — | — "

        row += "|\n"
        report_lines.append(row)

    # Decision
    report_lines.append(f"\n\n## Decision\n")
    report_lines.append(f"Reliability supervision improves over architecture-matched baseline on "
                       f"**{wins}/3** cancers.\n\n")
    if wins >= 2:
        report_lines.append("**RECOMMENDATION: Expand to 5-fold experiments on all 5 cancers.**\n")
    else:
        report_lines.append("**RECOMMENDATION: Reliability hypothesis lacks sufficient support. "
                          "Consider pivoting to alternative approaches.**\n")

    # Write outputs
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "PHASE2_RESULTS_SUMMARY.md")
    with open(report_path, "w") as f:
        f.writelines(report_lines)
    print(f"Report: {report_path}")

    csv_path = os.path.join(output_dir, "PHASE2_RESULTS.csv")
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"CSV: {csv_path}")

    return data, report_lines


def generate_figures(data, output_dir):
    """Generate Phase 2 comparison figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figures")
        return

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Bar chart: per-cancer comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = {"learned_gating_aux": "#2196F3", "pcrf_aux": "#FF5722", "pcrf_aux_warmup": "#4CAF50"}
    labels = {"learned_gating_aux": "Gating+Aux", "pcrf_aux": "PCRF+Aux", "pcrf_aux_warmup": "PCRF+Warmup"}

    for idx, cancer in enumerate(CANCERS):
        ax = axes[idx]
        cancer_data = data.get(cancer, {})
        x = np.arange(len(PHASE2_METHODS))
        means, stds = [], []
        for method in PHASE2_METHODS:
            folds = cancer_data.get(method, [])
            if folds:
                cis = [f["test_cindex"] for f in folds]
                means.append(np.mean(cis))
                stds.append(np.std(cis))
            else:
                means.append(0)
                stds.append(0)

        bars = ax.bar(x, means, yerr=stds, capsize=5,
                     color=[colors[m] for m in PHASE2_METHODS])
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in PHASE2_METHODS], rotation=15, ha="right")
        ax.set_ylabel("Test C-index")
        ax.set_title(cancer.upper())
        ax.set_ylim(0.55, 0.75)

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "phase2_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure: {fig_dir}/phase2_comparison.png")

    # Reliability weight distribution
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for idx, cancer in enumerate(CANCERS):
        ax = axes[idx]
        cancer_data = data.get(cancer, {})
        for method in PHASE2_METHODS:
            folds = cancer_data.get(method, [])
            if folds:
                img_ws = [f["test_reliability"]["alpha_img_mean"]
                         for f in folds if f.get("test_reliability")]
                if img_ws:
                    ax.bar(labels[method], np.mean(img_ws),
                          yerr=np.std(img_ws) if len(img_ws) > 1 else 0,
                          color=colors[method], capsize=5, alpha=0.8)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel("Image Reliability Weight")
        ax.set_title(cancer.upper())
        ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "phase2_reliability_weights.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure: {fig_dir}/phase2_reliability_weights.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_base", default="results/cvpr_reliability_pilot/phase2")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_base
    data, _ = analyze(args.results_base, args.folds, output_dir)
    generate_figures(data, output_dir)
