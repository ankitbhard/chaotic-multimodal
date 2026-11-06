#!/usr/bin/env python3
"""
RA-SAMO v4 Results Analysis.

Compares v4 probe-based controller against static methods and v3 variants.
Runs after v4 experiments complete.

Usage:
    python experiments/v4_results_analysis.py
    python experiments/v4_results_analysis.py --v4-dir results/ra_samo_v4_50ep
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import ttest_rel
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

REPO_ROOT = Path(__file__).resolve().parents[1]

CANCERS = ["brca", "kirc", "luad", "lusc", "stad"]
CANCER_LABELS = {"brca": "BRCA", "kirc": "KIRC", "luad": "LUAD",
                 "lusc": "LUSC", "stad": "STAD"}
STATIC_METHODS = ["baseline", "aux", "rank", "adaptive_aux", "aux_rank"]
FOLDS = list(range(5))

PER_CANCER_BEST = {
    "brca": "adaptive_aux",
    "stad": "adaptive_aux",
    "kirc": "baseline",
    "luad": "rank",
    "lusc": "rank",
}

PALETTE = [
    "#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800",
    "#00BCD4", "#E91E63", "#8BC34A",
]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3,
    "font.family": "sans-serif",
})


def load_summaries(result_dir):
    """Load summary.json files from a results directory.

    Returns: dict (cancer, fold, method) -> summary dict
    """
    summaries = {}
    if not result_dir.exists():
        print(f"  WARNING: {result_dir} does not exist")
        return summaries
    for cancer in CANCERS:
        cancer_dir = result_dir / cancer
        if not cancer_dir.exists():
            continue
        for fold in FOLDS:
            fold_dir = cancer_dir / f"fold_{fold}"
            if not fold_dir.exists():
                continue
            for method_dir in sorted(fold_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                summary_path = method_dir / "summary.json"
                if summary_path.exists():
                    try:
                        data = json.loads(summary_path.read_text())
                        if data.get("completed"):
                            summaries[(cancer, fold, method_dir.name)] = data
                    except (json.JSONDecodeError, OSError):
                        pass
    return summaries


def load_epochs(result_dir, method_filter=None):
    """Load epochs.csv files.

    Returns: dict (cancer, fold, method) -> list of row dicts
    """
    data = {}
    if not result_dir.exists():
        return data
    for cancer in CANCERS:
        cancer_dir = result_dir / cancer
        if not cancer_dir.exists():
            continue
        for fold in FOLDS:
            fold_dir = cancer_dir / f"fold_{fold}"
            if not fold_dir.exists():
                continue
            for method_dir in sorted(fold_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                if method_filter and method_dir.name not in method_filter:
                    continue
                epochs_path = method_dir / "epochs.csv"
                if epochs_path.exists():
                    rows = []
                    with open(epochs_path) as f:
                        for row in csv.DictReader(f):
                            converted = {}
                            for k, v in row.items():
                                if v == "":
                                    converted[k] = None
                                else:
                                    try:
                                        converted[k] = float(v)
                                    except (ValueError, TypeError):
                                        converted[k] = v
                            rows.append(converted)
                    data[(cancer, fold, method_dir.name)] = rows
    return data


def per_cancer_stats(summaries, methods):
    """Compute per-cancer mean +/- std test C-index for each method.

    Returns: dict cancer -> method -> (mean, std, values)
    """
    results = {}
    for cancer in CANCERS:
        results[cancer] = {}
        for method in methods:
            vals = []
            for fold in FOLDS:
                key = (cancer, fold, method)
                if key in summaries:
                    vals.append(summaries[key]["test_ci"])
            if vals:
                results[cancer][method] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "values": vals,
                    "n": len(vals),
                }
    return results


def paired_ttests(summaries, ref_method, other_methods):
    """Paired t-tests of ref_method vs each other per cancer.

    Returns: list of result dicts
    """
    if not HAS_SCIPY:
        print("  scipy not available, skipping t-tests")
        return []

    results = []
    for cancer in CANCERS:
        ref_vals = []
        for fold in FOLDS:
            key = (cancer, fold, ref_method)
            if key in summaries:
                ref_vals.append((fold, summaries[key]["test_ci"]))

        if len(ref_vals) < 2:
            continue

        for method in other_methods:
            other_vals = []
            for fold, _ in ref_vals:
                key = (cancer, fold, method)
                if key in summaries:
                    other_vals.append(summaries[key]["test_ci"])
                else:
                    other_vals = []
                    break

            if len(other_vals) != len(ref_vals):
                continue

            ref_arr = np.array([v for _, v in ref_vals])
            other_arr = np.array(other_vals)
            t_stat, p_val = ttest_rel(ref_arr, other_arr)
            results.append({
                "cancer": cancer,
                "reference": ref_method,
                "method": method,
                "mean_diff": float(np.mean(ref_arr - other_arr)),
                "t_stat": float(t_stat),
                "p_val": float(p_val),
                "n_folds": len(ref_vals),
                "significant": p_val < 0.05,
            })

    return results


def bootstrap_ci(values, n_boot=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    vals = np.array(values)
    n = len(vals)
    if n < 2:
        return float(np.mean(vals)), float(np.mean(vals)), float(np.mean(vals))
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(vals, size=n, replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.sort(boot_means)
    alpha = (1 - ci) / 2
    lo = float(boot_means[int(alpha * n_boot)])
    hi = float(boot_means[int((1 - alpha) * n_boot)])
    return float(np.mean(vals)), lo, hi


def probe_decision_summary(v4_summaries):
    """Summarize which probe candidate was selected per (cancer, fold).

    Returns: dict cancer -> {decisions: dict[fold->candidate], counts: dict[candidate->n]}
    """
    results = {}
    for cancer in CANCERS:
        decisions = {}
        counts = defaultdict(int)
        for fold in FOLDS:
            key = (cancer, fold, "ra_samo_v4")
            if key not in v4_summaries:
                continue
            v4_info = v4_summaries[key].get("v4_probe", {})
            committed = v4_info.get("committed_candidate", "unknown")
            decisions[fold] = committed
            counts[committed] += 1
        if decisions:
            results[cancer] = {"decisions": decisions, "counts": dict(counts)}
    return results


def _save_fig(fig, name, fig_dir):
    fig.savefig(fig_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png")


def plot_comparison_bar(stats, fig_dir):
    """Bar chart: v4 vs static methods per cancer."""
    all_methods = list(dict.fromkeys(
        ["ra_samo_v4"] + STATIC_METHODS
    ))
    available_methods = [m for m in all_methods
                         if any(m in stats.get(c, {}) for c in CANCERS)]
    if not available_methods:
        return

    x = np.arange(len(CANCERS))
    width = 0.8 / len(available_methods)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for j, method in enumerate(available_methods):
        means = []
        stds = []
        for cancer in CANCERS:
            entry = stats.get(cancer, {}).get(method)
            if entry:
                means.append(entry["mean"])
                stds.append(entry["std"])
            else:
                means.append(0)
                stds.append(0)
        offset = (j - len(available_methods) / 2 + 0.5) * width
        color = PALETTE[j % len(PALETTE)]
        lw = 2 if method == "ra_samo_v4" else 0
        ax.bar(x + offset, means, width, yerr=stds, capsize=2,
               label=method, color=color, alpha=0.85, edgecolor="black", linewidth=lw)

    ax.set_xticks(x)
    ax.set_xticklabels([CANCER_LABELS[c] for c in CANCERS])
    ax.set_ylabel("Test C-index")
    ax.set_title("v4 vs Static Methods")
    ax.legend(fontsize=5, ncol=3, loc="lower right")
    fig.tight_layout()
    _save_fig(fig, "v4_comparison_bar", fig_dir)


def plot_probe_decisions(probe_results, fig_dir):
    """Stacked bar chart of probe decisions per cancer."""
    if not probe_results:
        return

    candidates = sorted(set(
        c for info in probe_results.values() for c in info["counts"]
    ))
    cancers = [c for c in CANCERS if c in probe_results]
    if not cancers or not candidates:
        return

    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(cancers))
    bottom = np.zeros(len(cancers))

    for k, cand in enumerate(candidates):
        vals = [probe_results[c]["counts"].get(cand, 0) for c in cancers]
        ax.bar(x, vals, 0.6, bottom=bottom, label=cand,
               color=PALETTE[k % len(PALETTE)])
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([CANCER_LABELS[c] for c in cancers])
    ax.set_ylabel("Number of folds")
    ax.set_title("v4 Probe Decisions by Cancer")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, "probe_decisions", fig_dir)


def generate_report(stats, ttest_results, bootstrap_results, probe_results,
                    output_dir):
    """Generate v4_results_report.md."""
    lines = [
        "# RA-SAMO v4 Results Report",
        "",
    ]

    # ── Main results table ──
    lines.append("## 1. Test C-index (mean +/- std, 5 folds)")
    lines.append("")
    all_methods = list(dict.fromkeys(
        ["ra_samo_v4"] + STATIC_METHODS
    ))
    available = [m for m in all_methods
                 if any(m in stats.get(c, {}) for c in CANCERS)]

    header = "| Method | " + " | ".join(CANCER_LABELS[c] for c in CANCERS) + " | Mean |"
    sep = "|--------|" + "|".join("------" for _ in CANCERS) + "|------|"
    lines.extend([header, sep])

    for method in available:
        cells = []
        cross_means = []
        for cancer in CANCERS:
            entry = stats.get(cancer, {}).get(method)
            if entry:
                cells.append(f"{entry['mean']:.4f} +/- {entry['std']:.4f}")
                cross_means.append(entry["mean"])
            else:
                cells.append("—")
        overall = f"{np.mean(cross_means):.4f}" if cross_means else "—"
        marker = " **" if method == "ra_samo_v4" else ""
        lines.append(f"| {method}{marker} | " + " | ".join(cells) + f" | {overall} |")
    lines.append("")

    # ── Per-cancer best comparison ──
    lines.append("## 2. v4 vs Per-Cancer Best Static")
    lines.append("")
    lines.append("| Cancer | Best Static | Static CI | v4 CI | Diff |")
    lines.append("|--------|------------|-----------|-------|------|")
    for cancer in CANCERS:
        best_m = PER_CANCER_BEST.get(cancer, "baseline")
        static_entry = stats.get(cancer, {}).get(best_m)
        v4_entry = stats.get(cancer, {}).get("ra_samo_v4")
        if static_entry and v4_entry:
            diff = v4_entry["mean"] - static_entry["mean"]
            lines.append(
                f"| {CANCER_LABELS[cancer]} | {best_m} | {static_entry['mean']:.4f} "
                f"| {v4_entry['mean']:.4f} | {diff:+.4f} |"
            )
    lines.append("")

    # ── Paired t-tests ──
    if ttest_results:
        lines.append("## 3. Paired t-tests (v4 vs each static)")
        lines.append("")
        lines.append("| Cancer | vs Method | Diff | t-stat | p-value | Sig |")
        lines.append("|--------|-----------|------|--------|---------|-----|")
        for t in sorted(ttest_results, key=lambda x: (x["cancer"], x["method"])):
            sig = "*" if t["significant"] else ""
            lines.append(
                f"| {CANCER_LABELS[t['cancer']]} | {t['method']} "
                f"| {t['mean_diff']:+.4f} | {t['t_stat']:.3f} "
                f"| {t['p_val']:.4f} | {sig} |"
            )
        lines.append("")

    # ── Bootstrap CIs ──
    if bootstrap_results:
        lines.append("## 4. Bootstrap 95% Confidence Intervals")
        lines.append("")
        lines.append("| Cancer | Method | Mean | 95% CI |")
        lines.append("|--------|--------|------|--------|")
        for cancer in CANCERS:
            if cancer not in bootstrap_results:
                continue
            for method in ["ra_samo_v4"] + [PER_CANCER_BEST.get(cancer, "baseline")]:
                entry = bootstrap_results.get(cancer, {}).get(method)
                if entry:
                    lines.append(
                        f"| {CANCER_LABELS[cancer]} | {method} "
                        f"| {entry[0]:.4f} | [{entry[1]:.4f}, {entry[2]:.4f}] |"
                    )
        lines.append("")

    # ── Probe decisions ──
    if probe_results:
        lines.append("## 5. Probe Decisions")
        lines.append("")
        lines.append("Which weight configuration was selected per (cancer, fold)?")
        lines.append("")
        lines.append("| Cancer | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |")
        lines.append("|--------|--------|--------|--------|--------|--------|")
        for cancer in CANCERS:
            if cancer not in probe_results:
                continue
            decisions = probe_results[cancer]["decisions"]
            cells = [decisions.get(f, "—") for f in FOLDS]
            lines.append(f"| {CANCER_LABELS[cancer]} | " + " | ".join(str(c) for c in cells) + " |")
        lines.append("")

        lines.append("### Summary counts")
        lines.append("")
        for cancer in CANCERS:
            if cancer not in probe_results:
                continue
            counts = probe_results[cancer]["counts"]
            parts = [f"{k}: {v}" for k, v in sorted(counts.items())]
            lines.append(f"- **{CANCER_LABELS[cancer]}**: {', '.join(parts)}")
        lines.append("")

    # ── Interpretation ──
    lines.append("## 6. Interpretation")
    lines.append("")

    # Count wins/ties/losses vs per-cancer best
    wins = 0
    ties = 0
    losses = 0
    for cancer in CANCERS:
        v4_entry = stats.get(cancer, {}).get("ra_samo_v4")
        best_m = PER_CANCER_BEST.get(cancer, "baseline")
        static_entry = stats.get(cancer, {}).get(best_m)
        if v4_entry and static_entry:
            diff = v4_entry["mean"] - static_entry["mean"]
            if diff > 0.005:
                wins += 1
            elif diff < -0.005:
                losses += 1
            else:
                ties += 1

    lines.append(f"v4 vs per-cancer-best-static: {wins}W / {ties}T / {losses}L "
                 f"(threshold: +/-0.005)")
    lines.append("")

    if wins + ties >= 4:
        lines.append("v4 probe mechanism successfully identifies competitive configurations "
                     "without per-cancer tuning.")
    elif losses >= 3:
        lines.append("v4 underperforms — probe phase may be too costly or val-CI signals "
                     "too noisy for reliable selection.")
    else:
        lines.append("Mixed results — v4 matches some cancers but not others. "
                     "The method-selection problem remains hard.")
    lines.append("")

    report_path = output_dir / "v4_results_report.md"
    report_path.write_text("\n".join(lines))
    print(f"\nReport: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="RA-SAMO v4 results analysis")
    parser.add_argument("--v4-dir", type=Path,
                        default=REPO_ROOT / "results" / "ra_samo_v4_50ep")
    parser.add_argument("--static-dir", type=Path,
                        default=REPO_ROOT / "results" / "samo_all_cancers_50ep")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "results" / "v4_results")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load summaries
    print("Loading v4 results...")
    v4_summaries = load_summaries(args.v4_dir)
    print(f"  {len(v4_summaries)} v4 runs")

    print("Loading static results...")
    static_summaries = load_summaries(args.static_dir)
    print(f"  {len(static_summaries)} static runs")

    all_summaries = {**static_summaries, **v4_summaries}

    if not v4_summaries:
        print("No v4 results found. Run experiments first.")
        return

    # Per-cancer stats
    print("\nComputing per-cancer stats...")
    all_methods = STATIC_METHODS + ["ra_samo_v4"]
    stats = per_cancer_stats(all_summaries, all_methods)

    # Paired t-tests
    print("Running paired t-tests...")
    ttest_results = paired_ttests(all_summaries, "ra_samo_v4", STATIC_METHODS)

    # Bootstrap CIs
    print("Computing bootstrap CIs...")
    bootstrap_results = {}
    for cancer in CANCERS:
        bootstrap_results[cancer] = {}
        for method in ["ra_samo_v4", PER_CANCER_BEST.get(cancer, "baseline")]:
            entry = stats.get(cancer, {}).get(method)
            if entry and entry["values"]:
                bootstrap_results[cancer][method] = bootstrap_ci(entry["values"])

    # Probe decisions
    print("Summarizing probe decisions...")
    probe_results = probe_decision_summary(v4_summaries)

    # Figures
    print("\nGenerating figures...")
    plot_comparison_bar(stats, fig_dir)
    plot_probe_decisions(probe_results, fig_dir)

    # Report
    print("\nGenerating report...")
    generate_report(stats, ttest_results, bootstrap_results, probe_results, args.out)

    # Save raw data
    raw = {
        "stats": {c: {m: {"mean": v["mean"], "std": v["std"], "n": v["n"]}
                       for m, v in methods.items()}
                  for c, methods in stats.items()},
        "ttest_results": ttest_results,
        "probe_decisions": {c: v["counts"] for c, v in probe_results.items()},
    }
    (args.out / "raw_results.json").write_text(json.dumps(raw, indent=2))

    print("\nDone.")


if __name__ == "__main__":
    main()
