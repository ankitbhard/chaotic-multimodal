"""Post-run analysis for CLAM Attention MIL 2x3 experiment.

Loads summary.json and epochs.csv files, generates comparison tables,
training curves, and statistical tests.

Run: python -m experiments.clam_brca.analyze --results-dir results/clam_brca
"""

import argparse
import csv
import json
from collections import defaultdict
from math import erfc, sqrt
from pathlib import Path

import numpy as np


METHOD_ORDER = [
    "meanpool_adamw", "meanpool_aux_adamw", "meanpool_ra_samo",
    "clam_adamw", "clam_aux_adamw", "clam_ra_samo",
]
METHOD_LABELS = {
    "meanpool_adamw":     "MeanPool + AdamW",
    "meanpool_aux_adamw": "MeanPool + Aux + AdamW",
    "meanpool_ra_samo":   "MeanPool + RA-SAMO",
    "clam_adamw":         "CLAM + AdamW",
    "clam_aux_adamw":     "CLAM + Aux + AdamW",
    "clam_ra_samo":       "CLAM + RA-SAMO",
}


def load_results(results_dir):
    """Load all summary.json → method -> list of dicts."""
    results_dir = Path(results_dir)
    results = defaultdict(list)
    for path in sorted(results_dir.rglob("summary.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not data.get("completed"):
            continue
        method = data.get("method", path.parent.name)
        results[method].append(data)
    return dict(results)


def summary_table(results):
    """Print mean +/- std C-index table."""
    print("\n" + "=" * 70)
    print(f"{'Method':<28} {'Folds':>5}  {'Val C-index':>16}  {'Test C-index':>16}")
    print("-" * 70)
    for method in METHOD_ORDER:
        if method not in results:
            continue
        runs = results[method]
        val = [r["best_val_ci"] for r in runs]
        test = [r["test_ci"] for r in runs]
        label = METHOD_LABELS.get(method, method)
        print(f"{label:<28} {len(runs):>5}  "
              f"{np.mean(val):.4f} +/- {np.std(val):.4f}  "
              f"{np.mean(test):.4f} +/- {np.std(test):.4f}")
    print("=" * 70)


def paired_ttest(results, key="test_ci"):
    """Paired t-test: CLAM+RA-SAMO vs each other method."""
    if "clam_ra_samo" not in results:
        return
    ref = sorted(results["clam_ra_samo"], key=lambda r: r["fold"])
    ref_vals = np.array([r[key] for r in ref])
    n = len(ref_vals)

    print(f"\nPaired t-tests (CLAM+RA-SAMO vs others, metric={key}):")
    print(f"{'Comparison':<40} {'Diff':>8} {'t':>8} {'p':>8}")
    print("-" * 68)
    for method in METHOD_ORDER:
        if method == "clam_ra_samo" or method not in results:
            continue
        other = sorted(results[method], key=lambda r: r["fold"])
        if len(other) != n:
            continue
        other_vals = np.array([r[key] for r in other])
        diff = ref_vals - other_vals
        mean_d = np.mean(diff)
        if n > 1:
            se = np.std(diff, ddof=1) / np.sqrt(n)
            t_stat = mean_d / max(se, 1e-12)
            p_val = erfc(abs(t_stat) / sqrt(2))
        else:
            t_stat = p_val = float("nan")
        sig = " *" if p_val < 0.05 else ""
        label = METHOD_LABELS.get(method, method)
        print(f"  vs {label:<36} {mean_d:>+.4f} {t_stat:>8.3f} {p_val:>8.4f}{sig}")


def timing_table(results):
    """Print wall-clock time and peak GPU memory."""
    print(f"\n{'Method':<28} {'Time (s)':>10}  {'Peak GPU (MB)':>14}")
    print("-" * 58)
    for method in METHOD_ORDER:
        if method not in results:
            continue
        runs = results[method]
        times = [r.get("wall_seconds", 0) for r in runs]
        gpus = [r.get("peak_gpu_mb", 0) for r in runs]
        label = METHOD_LABELS.get(method, method)
        print(f"{label:<28} {np.mean(times):>8.0f} +/-{np.std(times):>4.0f}  "
              f"{np.mean(gpus):>10.0f} +/-{np.std(gpus):>4.0f}")


def generate_figures(results, results_dir):
    """Generate matplotlib figures if available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figures")
        return

    fig_dir = Path(results_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Bar chart: test C-index per method
    present = [m for m in METHOD_ORDER if m in results]
    means = [np.mean([r["test_ci"] for r in results[m]]) for m in present]
    stds = [np.std([r["test_ci"] for r in results[m]]) for m in present]
    labels = [METHOD_LABELS.get(m, m) for m in present]

    colors = ["#4DBEEE" if "meanpool" in m else "#D95319" for m in present]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Test C-index")
    ax.set_title("BRCA Survival: MeanPool vs CLAM Attention MIL")
    fig.tight_layout()
    fig.savefig(fig_dir / "cindex_bar.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_dir / 'cindex_bar.png'}")

    # Training curves: val C-index over epochs
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, prefix, title in [
        (axes[0], "meanpool", "MeanPool Methods"),
        (axes[1], "clam", "CLAM Methods"),
    ]:
        for method in present:
            if not method.startswith(prefix):
                continue
            label = METHOD_LABELS.get(method, method)
            method_dirs = sorted(
                Path(results_dir).rglob(f"*/{method}/epochs.csv"))
            for csv_path in method_dirs:
                try:
                    rows = list(csv.DictReader(csv_path.open()))
                    epochs = [int(r["epoch"]) for r in rows]
                    val_ci = [float(r["val_ci"]) for r in rows]
                    ax.plot(epochs, val_ci, label=label, alpha=0.7)
                except (KeyError, ValueError):
                    pass
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val C-index")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "val_ci_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_dir / 'val_ci_curves.png'}")


def write_report(results, results_dir):
    """Write final_report.md."""
    out = Path(results_dir) / "final_report.md"
    lines = [
        "# CLAM Attention MIL — BRCA Survival Results",
        "",
        "## Design",
        "2x3 controlled experiment: MeanPool vs CLAM x {AdamW, Aux+AdamW, RA-SAMO}",
        "",
        "## C-index Results",
        "",
        "| Method | Folds | Val C-index | Test C-index |",
        "|--------|-------|-------------|-------------|",
    ]
    for method in METHOD_ORDER:
        if method not in results:
            continue
        runs = results[method]
        val = [r["best_val_ci"] for r in runs]
        test = [r["test_ci"] for r in runs]
        label = METHOD_LABELS.get(method, method)
        lines.append(
            f"| {label} | {len(runs)} | "
            f"{np.mean(val):.4f} +/- {np.std(val):.4f} | "
            f"{np.mean(test):.4f} +/- {np.std(test):.4f} |"
        )
    lines.extend(["", "## Resource Usage", "",
                   "| Method | Wall (s) | Peak GPU (MB) |",
                   "|--------|----------|---------------|"])
    for method in METHOD_ORDER:
        if method not in results:
            continue
        runs = results[method]
        t = [r.get("wall_seconds", 0) for r in runs]
        g = [r.get("peak_gpu_mb", 0) for r in runs]
        label = METHOD_LABELS.get(method, method)
        lines.append(
            f"| {label} | {np.mean(t):.0f} +/- {np.std(t):.0f} | "
            f"{np.mean(g):.0f} +/- {np.std(g):.0f} |"
        )
    out.write_text("\n".join(lines))
    print(f"Report: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    args = p.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No completed results in {args.results_dir}")
        return

    print(f"Loaded {len(results)} methods:")
    for m, runs in sorted(results.items()):
        print(f"  {m}: {len(runs)} runs")

    summary_table(results)
    paired_ttest(results)
    timing_table(results)
    generate_figures(results, args.results_dir)
    write_report(results, args.results_dir)


if __name__ == "__main__":
    main()
