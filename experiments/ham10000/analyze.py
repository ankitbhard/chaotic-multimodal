"""Post-run analysis for HAM10000 benchmark.

Loads all summary.json files, generates figures, runs statistical tests,
and writes a final report.

Run: python -m experiments.ham10000.analyze --results-dir results/ham10000_classification
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METHOD_ORDER = [
    "image_only_adamw", "meta_only_adamw",
    "adamw", "sam", "adaptive_aux", "samo", "ra_samo",
]
METHOD_LABELS = {
    "image_only_adamw": "Image-Only",
    "meta_only_adamw": "Meta-Only",
    "adamw": "AdamW",
    "sam": "SAM",
    "adaptive_aux": "Adaptive-Aux",
    "samo": "SAMO",
    "ra_samo": "RA-SAMO",
}

METRICS = ["test_macro_f1", "test_balanced_acc", "test_auroc", "test_accuracy"]


def load_results(results_dir):
    """Load all summary.json files into a dict: method -> list of result dicts."""
    results_dir = Path(results_dir)
    results = defaultdict(list)

    for summary_path in sorted(results_dir.rglob("summary.json")):
        try:
            data = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not data.get("completed"):
            continue
        method = data.get("method", summary_path.parent.parent.name)
        results[method].append(data)

    return dict(results)


def compute_summary_table(results):
    """Compute mean +/- std for each method and metric."""
    rows = []
    for method in METHOD_ORDER:
        if method not in results:
            continue
        runs = results[method]
        row = {"method": METHOD_LABELS.get(method, method), "n_runs": len(runs)}
        for metric in METRICS:
            values = [r[metric] for r in runs if metric in r]
            if values:
                row[f"{metric}_mean"] = np.mean(values)
                row[f"{metric}_std"] = np.std(values)
            else:
                row[f"{metric}_mean"] = float("nan")
                row[f"{metric}_std"] = float("nan")
        rows.append(row)
    return rows


def paired_ttest(results, metric="test_macro_f1"):
    """Run paired t-test between RA-SAMO and each other method (across seeds)."""
    if "ra_samo" not in results:
        return {}

    ra_samo_runs = sorted(results["ra_samo"], key=lambda r: r["seed"])
    ra_samo_vals = np.array([r[metric] for r in ra_samo_runs])
    n = len(ra_samo_vals)

    tests = {}
    for method in METHOD_ORDER:
        if method == "ra_samo" or method not in results:
            continue
        other_runs = sorted(results[method], key=lambda r: r["seed"])
        if len(other_runs) != n:
            continue
        other_vals = np.array([r[metric] for r in other_runs])
        diff = ra_samo_vals - other_vals
        mean_diff = np.mean(diff)
        if n > 1:
            se = np.std(diff, ddof=1) / np.sqrt(n)
            t_stat = mean_diff / max(se, 1e-12)
            # Two-tailed p-value approximation (n-1 df, using normal for n>=3)
            from math import erfc, sqrt
            p_value = erfc(abs(t_stat) / sqrt(2))
        else:
            t_stat = float("nan")
            p_value = float("nan")
        tests[method] = {
            "mean_diff": float(mean_diff),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "n_pairs": n,
        }
    return tests


def generate_figures(results, output_dir):
    """Generate matplotlib figures if available, otherwise skip."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figures")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Bar chart: macro-F1 per method
    methods_present = [m for m in METHOD_ORDER if m in results]
    f1_means = []
    f1_stds = []
    labels = []
    for method in methods_present:
        vals = [r["test_macro_f1"] for r in results[method] if "test_macro_f1" in r]
        f1_means.append(np.mean(vals))
        f1_stds.append(np.std(vals))
        labels.append(METHOD_LABELS.get(method, method))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, f1_means, yerr=f1_stds, capsize=4, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("HAM10000 Classification: Test Macro-F1 by Method")
    ax.set_ylim(0, max(f1_means) * 1.2 if f1_means else 1.0)
    fig.tight_layout()
    fig.savefig(output_dir / "macro_f1_bar.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {output_dir / 'macro_f1_bar.png'}")

    # Training curves (from epochs.csv if available)
    import csv
    for method in methods_present:
        method_dir = Path(output_dir).parent / method
        csv_files = sorted(method_dir.rglob("epochs.csv"))
        if not csv_files:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        for csv_path in csv_files:
            seed_name = csv_path.parent.name
            rows = list(csv.DictReader(csv_path.open()))
            epochs = [int(r["epoch"]) for r in rows]
            train_loss = [float(r["train_fusion_loss"]) for r in rows]
            val_f1 = [float(r["val_macro_f1"]) for r in rows]
            ax1.plot(epochs, train_loss, label=seed_name, alpha=0.7)
            ax2.plot(epochs, val_f1, label=seed_name, alpha=0.7)

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train Loss")
        ax1.set_title(f"{METHOD_LABELS.get(method, method)} - Training Loss")
        ax1.legend()
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Val Macro-F1")
        ax2.set_title(f"{METHOD_LABELS.get(method, method)} - Validation F1")
        ax2.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"curves_{method}.png", dpi=150)
        plt.close(fig)

    # Confusion matrices
    for method in methods_present:
        method_dir = Path(output_dir).parent / method
        cm_files = sorted(method_dir.rglob("confusion_matrix.npy"))
        if not cm_files:
            continue
        # Average confusion matrix across seeds
        cms = [np.load(f) for f in cm_files]
        avg_cm = np.mean(cms, axis=0)

        fig, ax = plt.subplots(figsize=(7, 6))
        from experiments.ham10000.dataset import CLASSES
        im = ax.imshow(avg_cm, interpolation="nearest", cmap="Blues")
        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES, rotation=45, ha="right")
        ax.set_yticklabels(CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{METHOD_LABELS.get(method, method)} - Confusion Matrix (avg)")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(output_dir / f"cm_{method}.png", dpi=150)
        plt.close(fig)

    print(f"Figures saved to {output_dir}")


def write_report(results, output_dir):
    """Write final_report.md summarizing all results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = compute_summary_table(results)
    ttests = paired_ttest(results, "test_macro_f1")

    lines = [
        "# HAM10000 Multimodal Classification Benchmark — Results",
        "",
        "## Summary",
        "",
        "Evaluation of RA-SAMO generalization from survival prediction to",
        "dermoscopic image + tabular metadata 7-class lesion classification.",
        "",
        "## Results Table",
        "",
        "| Method | N | Macro-F1 | Balanced Acc | AUROC | Accuracy |",
        "|--------|---|----------|-------------|-------|----------|",
    ]

    for row in summary_rows:
        def _fmt(key):
            m = row.get(f"{key}_mean", float("nan"))
            s = row.get(f"{key}_std", float("nan"))
            if np.isnan(m):
                return "N/A"
            return f"{m:.4f} +/- {s:.4f}"

        lines.append(
            f"| {row['method']} | {row['n_runs']} | "
            f"{_fmt('test_macro_f1')} | {_fmt('test_balanced_acc')} | "
            f"{_fmt('test_auroc')} | {_fmt('test_accuracy')} |"
        )

    lines.extend(["", "## Statistical Tests (RA-SAMO vs others, paired t-test on Macro-F1)", ""])

    if ttests:
        lines.append("| Comparison | Mean Diff | t-stat | p-value | N |")
        lines.append("|------------|-----------|--------|---------|---|")
        for method, test in ttests.items():
            label = METHOD_LABELS.get(method, method)
            sig = " *" if test["p_value"] < 0.05 else ""
            lines.append(
                f"| RA-SAMO vs {label} | {test['mean_diff']:+.4f} | "
                f"{test['t_stat']:.3f} | {test['p_value']:.4f}{sig} | "
                f"{test['n_pairs']} |"
            )
    else:
        lines.append("Not enough data for paired tests.")

    # Per-method details
    lines.extend(["", "## Per-Method Details", ""])
    for method in METHOD_ORDER:
        if method not in results:
            continue
        label = METHOD_LABELS.get(method, method)
        runs = results[method]
        lines.append(f"### {label}")
        lines.append("")
        for r in sorted(runs, key=lambda x: x.get("seed", 0)):
            lines.append(
                f"- Seed {r.get('seed', '?')}: "
                f"F1={r.get('test_macro_f1', 'N/A'):.4f}, "
                f"Acc={r.get('test_balanced_acc', 'N/A'):.4f}, "
                f"AUROC={r.get('test_auroc', 'N/A'):.4f}, "
                f"Best epoch={r.get('best_epoch', '?')}, "
                f"Time={r.get('wall_seconds', 0):.0f}s"
            )
        lines.append("")

    report_path = output_dir / "final_report.md"
    report_path.write_text("\n".join(lines))
    print(f"Report written to {report_path}")


def main():
    p = argparse.ArgumentParser(description="Analyze HAM10000 benchmark results")
    p.add_argument("--results-dir", required=True, help="Path to results directory")
    args = p.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No completed results found in {args.results_dir}")
        return

    print(f"Loaded results for {len(results)} methods:")
    for method, runs in sorted(results.items()):
        print(f"  {method}: {len(runs)} runs")

    figures_dir = Path(args.results_dir) / "figures"
    generate_figures(results, figures_dir)
    write_report(results, Path(args.results_dir))


if __name__ == "__main__":
    main()
