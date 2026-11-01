#!/usr/bin/env python3
"""
Comprehensive analysis of information acquisition diagnostic experiments.

Loads all 27 runs (3 cancers x 3 methods x 3 folds x 50 epochs),
generates publication-quality figures and a final markdown report.

Usage:
    python experiments/samo/analyze_info_acquisition.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "information_acquisition_diagnostics"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CANCERS = ["brca", "kirc", "luad"]
METHODS = ["baseline", "adaptive_aux", "rank"]
FOLDS = [0, 1, 2]

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
METHOD_LABELS = {
    "baseline": "Baseline (Cox only)",
    "adaptive_aux": "Adaptive Aux",
    "rank": "Ranking Loss",
}
METHOD_COLORS = {
    "baseline": "#2196F3",
    "adaptive_aux": "#FF5722",
    "rank": "#4CAF50",
}
METHOD_LINESTYLES = {
    "baseline": "-",
    "adaptive_aux": "--",
    "rank": "-.",
}
CANCER_LABELS = {
    "brca": "BRCA",
    "kirc": "KIRC",
    "luad": "LUAD",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.8,
    "lines.markersize": 4,
    "legend.framealpha": 0.8,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_diagnostics() -> pd.DataFrame:
    """Load all 27 diagnostics.csv files into one DataFrame."""
    frames = []
    for cancer in CANCERS:
        for fold in FOLDS:
            for method in METHODS:
                p = RESULTS_DIR / cancer / f"fold_{fold}" / method / "diagnostics.csv"
                if not p.exists():
                    print(f"  WARNING: missing {p}")
                    continue
                df = pd.read_csv(p)
                frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Loaded diagnostics: {len(frames)} files, {len(combined)} rows")
    return combined


def load_all_epochs() -> pd.DataFrame:
    """Load all 27 epochs.csv files into one DataFrame."""
    frames = []
    for cancer in CANCERS:
        for fold in FOLDS:
            for method in METHODS:
                p = RESULTS_DIR / cancer / f"fold_{fold}" / method / "epochs.csv"
                if not p.exists():
                    print(f"  WARNING: missing {p}")
                    continue
                df = pd.read_csv(p)
                df["cancer"] = cancer
                df["method"] = method
                df["fold"] = fold
                frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Loaded epochs: {len(frames)} files, {len(combined)} rows")
    return combined


def load_all_summaries() -> pd.DataFrame:
    """Load all 27 summary.json files into one DataFrame."""
    records = []
    for cancer in CANCERS:
        for fold in FOLDS:
            for method in METHODS:
                p = RESULTS_DIR / cancer / f"fold_{fold}" / method / "summary.json"
                if not p.exists():
                    print(f"  WARNING: missing {p}")
                    continue
                with open(p) as f:
                    d = json.load(f)
                records.append(d)
    df = pd.DataFrame(records)
    print(f"  Loaded summaries: {len(records)} records")
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _mean_std_by_epoch(diag: pd.DataFrame, cancer: str, method: str,
                       col: str):
    """Return (epochs, mean, std) for a given cancer/method/column."""
    sub = diag[(diag["cancer"] == cancer) & (diag["method"] == method)]
    grouped = sub.groupby("epoch")[col]
    m = grouped.mean()
    s = grouped.std().fillna(0)
    return m.index.values, m.values, s.values


def _save(fig, name: str):
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure a: Gradient Sensitivity Trajectories
# ---------------------------------------------------------------------------
def plot_gradient_sensitivity(diag: pd.DataFrame):
    """Per-cancer img vs gene gradient sensitivity, 3 methods, mean+/-std."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, cancer in zip(axes, CANCERS):
        for method in METHODS:
            ep, m_img, s_img = _mean_std_by_epoch(diag, cancer, method,
                                                   "img_grad_sensitivity")
            ep, m_gene, s_gene = _mean_std_by_epoch(diag, cancer, method,
                                                    "gene_grad_sensitivity")
            color = METHOD_COLORS[method]
            ls = METHOD_LINESTYLES[method]
            label_img = f"{METHOD_LABELS[method]} (img)"
            label_gene = f"{METHOD_LABELS[method]} (gene)"
            ax.plot(ep, m_img, color=color, linestyle=ls, marker="o",
                    markersize=3, label=label_img)
            ax.fill_between(ep, m_img - s_img, m_img + s_img, alpha=0.12,
                            color=color)
            ax.plot(ep, m_gene, color=color, linestyle=ls, marker="s",
                    markersize=3, label=label_gene, alpha=0.6)
            ax.fill_between(ep, m_gene - s_gene, m_gene + s_gene, alpha=0.08,
                            color=color)
        ax.set_title(CANCER_LABELS[cancer])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gradient Sensitivity")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    axes[0].legend(loc="upper right", fontsize=7, ncol=1)
    fig.suptitle("Gradient Sensitivity Trajectories (img vs gene)", y=1.02,
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "a_gradient_sensitivity.png")


# ---------------------------------------------------------------------------
# Figure b: Fisher Trace Trajectories (normalized)
# ---------------------------------------------------------------------------
def plot_fisher_trace(diag: pd.DataFrame):
    """Per-cancer normalized Fisher trace, img vs gene."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, cancer in zip(axes, CANCERS):
        for method in METHODS:
            ep, m_img, s_img = _mean_std_by_epoch(
                diag, cancer, method, "img_fisher_trace_normalized")
            ep, m_gene, s_gene = _mean_std_by_epoch(
                diag, cancer, method, "gene_fisher_trace_normalized")
            color = METHOD_COLORS[method]
            ls = METHOD_LINESTYLES[method]
            ax.plot(ep, m_img, color=color, linestyle=ls, marker="o",
                    markersize=3, label=f"{METHOD_LABELS[method]} (img)")
            ax.fill_between(ep, m_img - s_img, m_img + s_img, alpha=0.12,
                            color=color)
            ax.plot(ep, m_gene, color=color, linestyle=ls, marker="s",
                    markersize=3, label=f"{METHOD_LABELS[method]} (gene)",
                    alpha=0.6)
            ax.fill_between(ep, m_gene - s_gene, m_gene + s_gene, alpha=0.08,
                            color=color)
        ax.set_title(CANCER_LABELS[cancer])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Normalized Fisher Trace")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-5, -5))
    axes[0].legend(loc="upper right", fontsize=7, ncol=1)
    fig.suptitle("Fisher Information Trace (normalized, img vs gene)",
                 y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "b_fisher_trace.png")


# ---------------------------------------------------------------------------
# Figure c: Gradient Alignment (cosine_cox_rank)
# ---------------------------------------------------------------------------
def plot_gradient_alignment(diag: pd.DataFrame):
    """Per-cancer cosine similarity between Cox and rank gradients."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, cancer in zip(axes, CANCERS):
        for method in METHODS:
            ep, m, s = _mean_std_by_epoch(diag, cancer, method,
                                          "cosine_cox_rank")
            color = METHOD_COLORS[method]
            ls = METHOD_LINESTYLES[method]
            ax.plot(ep, m, color=color, linestyle=ls, marker="o",
                    markersize=3, label=METHOD_LABELS[method])
            ax.fill_between(ep, m - s, m + s, alpha=0.15, color=color)
        ax.set_title(CANCER_LABELS[cancer])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cosine(Cox grad, Rank grad)")
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Gradient Alignment: Cox vs Ranking Loss", y=1.02,
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "c_gradient_alignment.png")


# ---------------------------------------------------------------------------
# Figure d: Validation C-index Trajectories
# ---------------------------------------------------------------------------
def plot_val_cindex(diag: pd.DataFrame):
    """Fused + unimodal img + unimodal gene C-index per cancer."""
    ci_cols = ["val_ci_fused", "val_ci_image_unimodal", "val_ci_gene_unimodal"]
    ci_labels = ["Fused", "Image (unimodal)", "Gene (unimodal)"]
    ci_markers = ["o", "^", "v"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, cancer in zip(axes, CANCERS):
        for method in METHODS:
            color = METHOD_COLORS[method]
            ls = METHOD_LINESTYLES[method]
            for ci_col, ci_lab, mk in zip(ci_cols, ci_labels, ci_markers):
                ep, m, s = _mean_std_by_epoch(diag, cancer, method, ci_col)
                alpha = 1.0 if ci_col == "val_ci_fused" else 0.5
                lw = 2.0 if ci_col == "val_ci_fused" else 1.2
                label = f"{METHOD_LABELS[method]} — {ci_lab}"
                ax.plot(ep, m, color=color, linestyle=ls, marker=mk,
                        markersize=3, alpha=alpha, linewidth=lw, label=label)
                ax.fill_between(ep, m - s, m + s, alpha=0.08, color=color)
        ax.set_title(CANCER_LABELS[cancer])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation C-index")
        ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle=":")
    axes[0].legend(loc="lower right", fontsize=6, ncol=1)
    fig.suptitle("Validation C-index (Fused + Unimodal Probes)", y=1.02,
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "d_val_cindex.png")


# ---------------------------------------------------------------------------
# Figure e: Fisher Ratio (gene/img) — Modality Imbalance
# ---------------------------------------------------------------------------
def plot_fisher_ratio(diag: pd.DataFrame):
    """Fisher ratio (gene/img) over training — modality imbalance indicator."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, cancer in zip(axes, CANCERS):
        for method in METHODS:
            ep, m, s = _mean_std_by_epoch(diag, cancer, method,
                                          "fisher_ratio_gene_over_img")
            color = METHOD_COLORS[method]
            ls = METHOD_LINESTYLES[method]
            ax.plot(ep, m, color=color, linestyle=ls, marker="o",
                    markersize=3, label=METHOD_LABELS[method])
            ax.fill_between(ep, m - s, m + s, alpha=0.15, color=color)
        ax.axhline(y=1.0, color="gray", linewidth=1.0, linestyle="--",
                   label="Balanced (ratio=1)")
        ax.set_title(CANCER_LABELS[cancer])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Fisher Ratio (gene / img)")
        ax.set_yscale("log")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("Fisher Information Ratio (gene/img) — Modality Imbalance",
                 y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "e_fisher_ratio.png")


# ---------------------------------------------------------------------------
# Figure f: Gradient Variability Trajectories
# ---------------------------------------------------------------------------
def plot_grad_variability(diag: pd.DataFrame):
    """Cox gradient variability across folds."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, cancer in zip(axes, CANCERS):
        for method in METHODS:
            ep, m, s = _mean_std_by_epoch(diag, cancer, method,
                                          "cox_grad_variability")
            color = METHOD_COLORS[method]
            ls = METHOD_LINESTYLES[method]
            ax.plot(ep, m, color=color, linestyle=ls, marker="o",
                    markersize=3, label=METHOD_LABELS[method])
            ax.fill_between(ep, m - s, m + s, alpha=0.15, color=color)
        ax.set_title(CANCER_LABELS[cancer])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cox Gradient Variability")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("Cox Gradient Variability Over Training", y=1.02,
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "f_grad_variability.png")


# ---------------------------------------------------------------------------
# Figure g: Summary Bar Chart — Test C-index per method per cancer
# ---------------------------------------------------------------------------
def plot_test_cindex_bars(summaries: pd.DataFrame):
    """Bar chart of mean test C-index per method per cancer, with fold SEM."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.22
    x = np.arange(len(CANCERS))

    for i, method in enumerate(METHODS):
        means, sems = [], []
        for cancer in CANCERS:
            sub = summaries[(summaries["dataset"] == cancer) &
                            (summaries["method"] == method)]
            vals = sub["test_ci"].values
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1
                        else 0)
        offset = (i - 1) * bar_width
        bars = ax.bar(x + offset, means, bar_width, yerr=sems,
                      color=METHOD_COLORS[method], alpha=0.85,
                      label=METHOD_LABELS[method], capsize=4, edgecolor="k",
                      linewidth=0.5)
        # Add value labels on bars
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([CANCER_LABELS[c] for c in CANCERS], fontsize=12)
    ax.set_ylabel("Test C-index", fontsize=12)
    ax.set_title("Test C-index by Method and Cancer Type\n(mean +/- SEM across 3 folds)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0.5, max(summaries["test_ci"]) + 0.06)
    ax.axhline(y=0.5, color="gray", linewidth=0.8, linestyle="--",
               label="Random (0.5)")
    fig.tight_layout()
    _save(fig, "g_test_cindex_bars.png")


# ---------------------------------------------------------------------------
# Figure h: Heatmap — Early Fisher Ratio vs Final C-index Delta
# ---------------------------------------------------------------------------
def plot_fisher_vs_cindex_heatmap(diag: pd.DataFrame,
                                  summaries: pd.DataFrame):
    """Scatter/heatmap: early-epoch (1-5) avg Fisher ratio vs test C-index
    improvement over baseline."""
    # Compute early Fisher ratio per (cancer, method, fold)
    early = diag[diag["epoch"] <= 5].copy()
    early_agg = early.groupby(["cancer", "method", "fold"])[
        "fisher_ratio_gene_over_img"].mean().reset_index()
    early_agg.rename(columns={"fisher_ratio_gene_over_img": "early_fisher_ratio"},
                     inplace=True)

    # Merge with test C-index
    merged = early_agg.merge(
        summaries[["dataset", "method", "fold", "test_ci"]],
        left_on=["cancer", "method", "fold"],
        right_on=["dataset", "method", "fold"],
        how="left"
    )

    # Compute delta vs baseline per cancer/fold
    baseline_ci = merged[merged["method"] == "baseline"][
        ["cancer", "fold", "test_ci"]].rename(columns={"test_ci": "baseline_ci"})
    merged = merged.merge(baseline_ci, on=["cancer", "fold"], how="left")
    merged["ci_delta"] = merged["test_ci"] - merged["baseline_ci"]

    # Create figure with two subplots: scatter and heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Scatter plot
    cancer_markers = {"brca": "o", "kirc": "s", "luad": "D"}
    for cancer in CANCERS:
        for method in METHODS:
            sub = merged[(merged["cancer"] == cancer) &
                         (merged["method"] == method)]
            ax1.scatter(sub["early_fisher_ratio"], sub["ci_delta"],
                        color=METHOD_COLORS[method],
                        marker=cancer_markers[cancer], s=80, alpha=0.8,
                        edgecolors="k", linewidths=0.5,
                        label=f"{CANCER_LABELS[cancer]} / {METHOD_LABELS[method]}")
    ax1.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
    ax1.axvline(x=1.0, color="gray", linewidth=0.8, linestyle=":",
                label="Fisher ratio=1 (balanced)")
    ax1.set_xlabel("Early-epoch (1-5) Mean Fisher Ratio (gene/img)")
    ax1.set_ylabel("Test C-index Delta vs Baseline")
    ax1.set_title("Early Fisher Ratio vs Test Performance Change")
    ax1.legend(fontsize=6, ncol=2, loc="best")

    # Right: Heatmap of mean values
    pivot_fisher = merged.groupby(["cancer", "method"])[
        "early_fisher_ratio"].mean().unstack()
    pivot_ci = merged.groupby(["cancer", "method"])[
        "ci_delta"].mean().unstack()

    # Combined heatmap: cells colored by CI delta, annotated with Fisher ratio
    data_matrix = pivot_ci.loc[CANCERS, METHODS].values
    fisher_matrix = pivot_fisher.loc[CANCERS, METHODS].values

    im = ax2.imshow(data_matrix, cmap="RdYlGn", aspect="auto",
                    vmin=-0.03, vmax=0.03)
    ax2.set_xticks(range(len(METHODS)))
    ax2.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=9,
                        rotation=20, ha="right")
    ax2.set_yticks(range(len(CANCERS)))
    ax2.set_yticklabels([CANCER_LABELS[c] for c in CANCERS], fontsize=11)
    # Annotate cells
    for i in range(len(CANCERS)):
        for j in range(len(METHODS)):
            ci_val = data_matrix[i, j]
            fr_val = fisher_matrix[i, j]
            txt = f"dCI={ci_val:+.3f}\nFR={fr_val:.2f}"
            ax2.text(j, i, txt, ha="center", va="center", fontsize=8,
                     fontweight="bold",
                     color="white" if abs(ci_val) > 0.015 else "black")
    cbar = fig.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label("Test C-index Delta vs Baseline", fontsize=10)
    ax2.set_title("C-index Change & Early Fisher Ratio")

    fig.suptitle("Early-epoch Fisher Ratio vs Final Test C-index Improvement",
                 y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "h_fisher_vs_cindex_heatmap.png")

    return merged


# ---------------------------------------------------------------------------
# Additional Figure: Gradient Norm Ratio (gene/img) trajectories
# ---------------------------------------------------------------------------
def plot_grad_norm_ratio(diag: pd.DataFrame):
    """Gradient norm ratio (gene/img) over training."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, cancer in zip(axes, CANCERS):
        for method in METHODS:
            ep, m, s = _mean_std_by_epoch(diag, cancer, method,
                                          "grad_norm_ratio_gene_over_img")
            color = METHOD_COLORS[method]
            ls = METHOD_LINESTYLES[method]
            ax.plot(ep, m, color=color, linestyle=ls, marker="o",
                    markersize=3, label=METHOD_LABELS[method])
            ax.fill_between(ep, m - s, m + s, alpha=0.15, color=color)
        ax.axhline(y=1.0, color="gray", linewidth=1.0, linestyle="--")
        ax.set_title(CANCER_LABELS[cancer])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Grad Norm Ratio (gene / img)")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("Gradient Norm Ratio (gene/img) Over Training", y=1.02,
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "i_grad_norm_ratio.png")


# ---------------------------------------------------------------------------
# Additional Figure: Training + Validation Loss
# ---------------------------------------------------------------------------
def plot_loss_curves(epochs_df: pd.DataFrame):
    """Training and validation loss curves per cancer/method."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=False)
    for j, cancer in enumerate(CANCERS):
        for method in METHODS:
            sub = epochs_df[(epochs_df["cancer"] == cancer) &
                            (epochs_df["method"] == method)]
            color = METHOD_COLORS[method]
            ls = METHOD_LINESTYLES[method]
            for row_idx, (col, label_prefix, ax_row) in enumerate([
                ("train_fusion_loss", "Train", axes[0]),
                ("val_loss", "Val", axes[1]),
            ]):
                ax = ax_row[j]
                grouped = sub.groupby("epoch")[col]
                m = grouped.mean()
                s = grouped.std().fillna(0)
                ax.plot(m.index, m.values, color=color, linestyle=ls,
                        label=METHOD_LABELS[method], linewidth=1.5)
                ax.fill_between(m.index, m.values - s.values,
                                m.values + s.values, alpha=0.12, color=color)
                ax.set_title(f"{CANCER_LABELS[cancer]} — {label_prefix} Loss")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
    axes[0][0].legend(fontsize=9)
    fig.suptitle("Training and Validation Loss Curves", y=1.01,
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "j_loss_curves.png")


# ---------------------------------------------------------------------------
# Additional Figure: Validation C-index from epochs.csv (denser, every epoch)
# ---------------------------------------------------------------------------
def plot_val_ci_dense(epochs_df: pd.DataFrame):
    """Dense per-epoch validation C-index (from epochs.csv, every epoch)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, cancer in zip(axes, CANCERS):
        for method in METHODS:
            sub = epochs_df[(epochs_df["cancer"] == cancer) &
                            (epochs_df["method"] == method)]
            color = METHOD_COLORS[method]
            ls = METHOD_LINESTYLES[method]
            grouped = sub.groupby("epoch")["val_ci"]
            m = grouped.mean()
            s = grouped.std().fillna(0)
            ax.plot(m.index, m.values, color=color, linestyle=ls,
                    label=METHOD_LABELS[method], linewidth=1.5)
            ax.fill_between(m.index, m.values - s.values,
                            m.values + s.values, alpha=0.15, color=color)
        ax.set_title(CANCER_LABELS[cancer])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation C-index")
        ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle=":")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Validation C-index Over Training (all epochs)", y=1.02,
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "k_val_ci_dense.png")


# ---------------------------------------------------------------------------
# Additional Figure: Cosine similarities (all three pairs)
# ---------------------------------------------------------------------------
def plot_all_cosines(diag: pd.DataFrame):
    """All pairwise gradient cosine similarities."""
    cosine_cols = ["cosine_cox_aux", "cosine_cox_rank", "cosine_aux_rank"]
    cosine_labels = ["cos(Cox, Aux)", "cos(Cox, Rank)", "cos(Aux, Rank)"]
    cosine_markers = ["o", "s", "D"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, cancer in zip(axes, CANCERS):
        for method in METHODS:
            color = METHOD_COLORS[method]
            ls = METHOD_LINESTYLES[method]
            for cos_col, cos_lab, mk in zip(cosine_cols, cosine_labels,
                                            cosine_markers):
                ep, m, s = _mean_std_by_epoch(diag, cancer, method, cos_col)
                # Only plot if not all zeros
                if np.max(np.abs(m)) > 1e-10:
                    ax.plot(ep, m, color=color, linestyle=ls, marker=mk,
                            markersize=3,
                            label=f"{METHOD_LABELS[method]} — {cos_lab}")
                    ax.fill_between(ep, m - s, m + s, alpha=0.08, color=color)
        ax.set_title(CANCER_LABELS[cancer])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cosine Similarity")
        ax.set_ylim(-0.2, 1.1)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
    axes[0].legend(loc="lower right", fontsize=6, ncol=1)
    fig.suptitle("Pairwise Gradient Cosine Similarities", y=1.02,
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "l_all_cosines.png")


# ---------------------------------------------------------------------------
# Additional Figure: Weight evolution for adaptive_aux
# ---------------------------------------------------------------------------
def plot_weight_evolution(diag: pd.DataFrame):
    """Loss-weight evolution for adaptive_aux (weights change over training)."""
    weight_cols = ["weight_img", "weight_gene", "weight_rank"]
    weight_labels = ["w_img", "w_gene", "w_rank"]
    weight_colors = ["#2196F3", "#FF5722", "#4CAF50"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, cancer in zip(axes, CANCERS):
        for wcol, wlab, wcolor in zip(weight_cols, weight_labels,
                                       weight_colors):
            for method in METHODS:
                sub = diag[(diag["cancer"] == cancer) &
                           (diag["method"] == method)]
                grouped = sub.groupby("epoch")[wcol]
                m = grouped.mean()
                s = grouped.std().fillna(0)
                # Only plot if weights are non-trivial
                if m.max() > 0.01:
                    ls = METHOD_LINESTYLES[method]
                    ax.plot(m.index, m.values, color=wcolor, linestyle=ls,
                            marker="o", markersize=3,
                            label=f"{METHOD_LABELS[method]} — {wlab}")
                    ax.fill_between(m.index, m.values - s.values,
                                    m.values + s.values, alpha=0.1,
                                    color=wcolor)
        ax.set_title(CANCER_LABELS[cancer])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss Weight")
    axes[0].legend(loc="best", fontsize=6, ncol=1)
    fig.suptitle("Loss Weight Evolution", y=1.02,
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "m_weight_evolution.png")


# ---------------------------------------------------------------------------
# Compute analysis statistics for the report
# ---------------------------------------------------------------------------
def compute_statistics(diag: pd.DataFrame, epochs_df: pd.DataFrame,
                       summaries: pd.DataFrame) -> dict:
    """Compute key statistics for the report."""
    stats_dict = {}

    # 1. Per-cancer per-method test CI (mean +/- std)
    ci_table = summaries.groupby(["dataset", "method"])["test_ci"].agg(
        ["mean", "std", "count"]).reset_index()
    stats_dict["ci_table"] = ci_table

    # 2. Best val CI (mean +/- std)
    val_ci_table = summaries.groupby(["dataset", "method"])["best_val_ci"].agg(
        ["mean", "std"]).reset_index()
    stats_dict["val_ci_table"] = val_ci_table

    # 3. Early (epochs 1-5) Fisher ratio
    early = diag[diag["epoch"] <= 5]
    early_fisher = early.groupby(["cancer", "method"])[
        "fisher_ratio_gene_over_img"].agg(["mean", "std"]).reset_index()
    stats_dict["early_fisher"] = early_fisher

    # 4. Late (epochs 40-50) Fisher ratio
    late = diag[diag["epoch"] >= 40]
    late_fisher = late.groupby(["cancer", "method"])[
        "fisher_ratio_gene_over_img"].agg(["mean", "std"]).reset_index()
    stats_dict["late_fisher"] = late_fisher

    # 5. Mean gradient sensitivity ratio (gene/img) across training
    diag_copy = diag.copy()
    diag_copy["grad_sens_ratio"] = (diag_copy["gene_grad_sensitivity"] /
                                     diag_copy["img_grad_sensitivity"].clip(lower=1e-15))
    grad_sens = diag_copy.groupby(["cancer", "method"])[
        "grad_sens_ratio"].agg(["mean", "std"]).reset_index()
    stats_dict["grad_sens_ratio"] = grad_sens

    # 6. Cosine similarity stats
    cosine_stats = diag.groupby(["cancer", "method"])[
        ["cosine_cox_aux", "cosine_cox_rank", "cosine_aux_rank"]].agg(
        ["mean", "std"]).reset_index()
    stats_dict["cosine_stats"] = cosine_stats

    # 7. Best epoch per cancer/method
    best_epoch_table = summaries.groupby(["dataset", "method"])["best_epoch"].agg(
        ["mean", "std"]).reset_index()
    stats_dict["best_epoch_table"] = best_epoch_table

    # 8. Wall time stats
    wall_time_table = summaries.groupby(["dataset", "method"])["wall_seconds"].agg(
        ["mean", "std"]).reset_index()
    stats_dict["wall_time_table"] = wall_time_table

    # 9. Paired t-tests: each method vs baseline per cancer
    paired_tests = []
    for cancer in CANCERS:
        base = summaries[(summaries["dataset"] == cancer) &
                         (summaries["method"] == "baseline")]["test_ci"].values
        for method in ["adaptive_aux", "rank"]:
            other = summaries[(summaries["dataset"] == cancer) &
                              (summaries["method"] == method)]["test_ci"].values
            if len(base) == len(other) and len(base) >= 2:
                t_stat, p_val = stats.ttest_rel(other, base)
                diff = np.mean(other) - np.mean(base)
            else:
                t_stat, p_val, diff = np.nan, np.nan, np.nan
            paired_tests.append({
                "cancer": cancer, "method": method,
                "delta_ci": diff, "t_stat": t_stat, "p_value": p_val
            })
    stats_dict["paired_tests"] = pd.DataFrame(paired_tests)

    # 10. Unimodal C-index gap at epoch 50
    epoch50 = diag[diag["epoch"] == 50]
    unimodal_gap = epoch50.groupby(["cancer", "method"])[
        "val_ci_gap_gene_minus_img"].agg(["mean", "std"]).reset_index()
    stats_dict["unimodal_gap"] = unimodal_gap

    return stats_dict


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(stats_dict: dict, merged_heatmap: pd.DataFrame,
                    diag: pd.DataFrame, summaries: pd.DataFrame) -> str:
    """Generate the final markdown report."""

    ci = stats_dict["ci_table"]
    early_f = stats_dict["early_fisher"]
    late_f = stats_dict["late_fisher"]
    paired = stats_dict["paired_tests"]
    unimodal = stats_dict["unimodal_gap"]
    wall = stats_dict["wall_time_table"]
    best_ep = stats_dict["best_epoch_table"]

    # Helper to format CI table row
    def _ci_row(cancer, method):
        row = ci[(ci["dataset"] == cancer) & (ci["method"] == method)]
        if len(row) == 0:
            return "N/A"
        m, s = row.iloc[0]["mean"], row.iloc[0]["std"]
        return f"{m:.4f} +/- {s:.4f}"

    def _fisher_row(df, cancer, method):
        row = df[(df["cancer"] == cancer) & (df["method"] == method)]
        if len(row) == 0:
            return "N/A"
        m, s = row.iloc[0]["mean"], row.iloc[0]["std"]
        return f"{m:.3f} +/- {s:.3f}"

    # Build test CI improvement summary
    improvements = []
    for _, row in paired.iterrows():
        sig = "p < 0.05" if row["p_value"] < 0.05 else f"p = {row['p_value']:.3f}"
        improvements.append(
            f"| {CANCER_LABELS[row['cancer']]} | {METHOD_LABELS[row['method']]} "
            f"| {row['delta_ci']:+.4f} | {row['t_stat']:.3f} | {sig} |"
        )

    # Mean wall time
    mean_wall = summaries["wall_seconds"].mean()
    total_wall = summaries["wall_seconds"].sum()

    report = f"""# Information Acquisition Diagnostics: Final Report

**Date**: 2026-11-01
**Experiment**: 27 runs (3 cancers x 3 methods x 3 folds x 50 epochs)
**Total compute**: {total_wall:.0f}s ({total_wall/3600:.1f}h), mean {mean_wall:.0f}s per run

---

## 1. Methodology

### 1.1 Objective

This experiment investigates whether auxiliary training objectives (adaptive auxiliary
unimodal survival heads, pairwise ranking loss) improve **information acquisition** in
multimodal survival prediction, as measured by gradient-level diagnostics. The goal is to
determine whether an InfoReg-style intervention is warranted before committing to a full
hyperparameter sweep.

### 1.2 Experimental Design

We train a late-fusion multimodal survival model (image encoder + gene encoder + fusion
MLP) on three TCGA cancers:

- **BRCA** (breast): 1,124 patients, known to have image-gene modality imbalance
- **KIRC** (kidney): 519 patients, historically more balanced modalities
- **LUAD** (lung adenocarcinoma): 522 patients, intermediate characteristics

Each cancer is trained with 3 methods:

| Method | Description |
|--------|-------------|
| **Baseline** | Standard Cox partial likelihood loss only |
| **Adaptive Aux** | Cox loss + learned-weight unimodal image/gene survival heads |
| **Rank** | Cox loss + pairwise ranking margin loss |

Each method runs for 50 epochs across 3 cross-validation folds (seeds 42, 43, 44).

### 1.3 Diagnostic Measurements

At diagnostic epochs (1--15, 20, 25, 30, 40, 50), we compute:

**Fisher Information Trace (per encoder)**:

$$\\text{{Tr}}(\\hat{{F}}_m) = \\frac{{1}}{{|\\mathcal{{E}}|}} \\sum_{{i \\in \\mathcal{{E}}}} \\left\\| \\nabla_{{\\theta_m}} \\ell_i^{{\\text{{Cox}}}} \\right\\|^2$$

where $\\mathcal{{E}}$ is the set of uncensored events and $\\theta_m$ are the parameters
of modality encoder $m$. This uses Breslow tied-event handling with a mini-batch risk set
approximation. The normalized version divides by the number of encoder parameters.

**Gradient Sensitivity** (mean absolute gradient per parameter):

$$\\text{{GradSens}}_m = \\frac{{1}}{{|\\theta_m|}} \\sum_{{k}} \\left| \\frac{{\\partial \\mathcal{{L}}_{{\\text{{Cox}}}}}}{{\\partial \\theta_{{m,k}}}} \\right|$$

**Fisher Ratio** (modality imbalance indicator):

$$\\text{{FR}} = \\frac{{\\text{{Tr}}(\\hat{{F}}_{{\\text{{gene}}}})}}{{\\text{{Tr}}(\\hat{{F}}_{{\\text{{img}}}})}}$$

Values > 1 indicate the gene encoder receives more gradient signal; values < 1 indicate
the image encoder dominates.

**Concordance Index (C-index)** for validation:

$$C = \\frac{{\\sum_{{i,j: t_i < t_j}} \\mathbf{{1}}[\\hat{{r}}_i > \\hat{{r}}_j]}}{{\\sum_{{i,j: t_i < t_j}} 1}}$$

Computed for the fused model and for unimodal probes (frozen-feature linear Cox probes
fitted on the training data for baseline/rank; trained auxiliary heads for adaptive_aux).

**Gradient Alignment** (cosine similarity between objective gradients):

$$\\cos(\\nabla \\mathcal{{L}}_{{\\text{{Cox}}}}, \\nabla \\mathcal{{L}}_{{\\text{{Rank}}}})$$

Values near 1 indicate the objectives push in the same direction; near 0 suggests
orthogonal information.

---

## 2. Results Overview

### 2.1 Test C-index Summary

| Cancer | Baseline | Adaptive Aux | Rank |
|--------|----------|-------------|------|
| BRCA | {_ci_row('brca', 'baseline')} | {_ci_row('brca', 'adaptive_aux')} | {_ci_row('brca', 'rank')} |
| KIRC | {_ci_row('kirc', 'baseline')} | {_ci_row('kirc', 'adaptive_aux')} | {_ci_row('kirc', 'rank')} |
| LUAD | {_ci_row('luad', 'baseline')} | {_ci_row('luad', 'adaptive_aux')} | {_ci_row('luad', 'rank')} |

![Test C-index by Method and Cancer](figures/g_test_cindex_bars.png)

### 2.2 Statistical Significance (Paired t-test vs Baseline)

| Cancer | Method | Delta C-index | t-stat | Significance |
|--------|--------|--------------|--------|-------------|
{chr(10).join(improvements)}

### 2.3 Best Epoch Timing

"""

    # Add best epoch table
    report += "| Cancer | Method | Best Epoch (mean +/- std) |\n"
    report += "|--------|--------|-------------------------|\n"
    for cancer in CANCERS:
        for method in METHODS:
            row = best_ep[(best_ep["dataset"] == cancer) &
                          (best_ep["method"] == method)]
            if len(row) > 0:
                m, s = row.iloc[0]["mean"], row.iloc[0]["std"]
                report += f"| {CANCER_LABELS[cancer]} | {METHOD_LABELS[method]} | {m:.1f} +/- {s:.1f} |\n"

    report += f"""
---

## 3. Gradient-Level Diagnostics

### 3.1 Gradient Sensitivity Trajectories

The gradient sensitivity measures the mean absolute gradient magnitude per parameter for
each modality encoder. Larger values indicate the encoder is receiving stronger learning
signal from the Cox loss.

![Gradient Sensitivity](figures/a_gradient_sensitivity.png)

**Key observations:**
"""

    # Analyze gradient sensitivity patterns
    for cancer in CANCERS:
        sub = diag[diag["cancer"] == cancer]
        for method in METHODS:
            m_sub = sub[sub["method"] == method]
            mean_img = m_sub["img_grad_sensitivity"].mean()
            mean_gene = m_sub["gene_grad_sensitivity"].mean()
            ratio = mean_gene / max(mean_img, 1e-15)
            dominant = "gene" if ratio > 1.0 else "image"
            report += (f"- **{CANCER_LABELS[cancer]} / {METHOD_LABELS[method]}**: "
                       f"Gene/img ratio = {ratio:.2f} ('{dominant}' dominant)\n")

    report += f"""
### 3.2 Fisher Information Trace

The normalized Fisher trace captures the curvature of the loss landscape with respect to
each encoder's parameters. Higher Fisher trace means the encoder's parameters are in a
region where small changes significantly affect the loss — i.e., the encoder is "learning."

![Fisher Trace](figures/b_fisher_trace.png)

### 3.3 Fisher Ratio (Modality Imbalance)

The Fisher ratio (gene/img) directly measures modality imbalance. A ratio of 1.0 means
balanced information acquisition; deviations indicate one modality dominates learning.

![Fisher Ratio](figures/e_fisher_ratio.png)

**Early-epoch Fisher ratios (epochs 1--5):**

| Cancer | Baseline | Adaptive Aux | Rank |
|--------|----------|-------------|------|
| BRCA | {_fisher_row(early_f, 'brca', 'baseline')} | {_fisher_row(early_f, 'brca', 'adaptive_aux')} | {_fisher_row(early_f, 'brca', 'rank')} |
| KIRC | {_fisher_row(early_f, 'kirc', 'baseline')} | {_fisher_row(early_f, 'kirc', 'adaptive_aux')} | {_fisher_row(early_f, 'kirc', 'rank')} |
| LUAD | {_fisher_row(early_f, 'luad', 'baseline')} | {_fisher_row(early_f, 'luad', 'adaptive_aux')} | {_fisher_row(early_f, 'luad', 'rank')} |

**Late-epoch Fisher ratios (epochs 40--50):**

| Cancer | Baseline | Adaptive Aux | Rank |
|--------|----------|-------------|------|
| BRCA | {_fisher_row(late_f, 'brca', 'baseline')} | {_fisher_row(late_f, 'brca', 'adaptive_aux')} | {_fisher_row(late_f, 'brca', 'rank')} |
| KIRC | {_fisher_row(late_f, 'kirc', 'baseline')} | {_fisher_row(late_f, 'kirc', 'adaptive_aux')} | {_fisher_row(late_f, 'kirc', 'rank')} |
| LUAD | {_fisher_row(late_f, 'luad', 'baseline')} | {_fisher_row(late_f, 'luad', 'adaptive_aux')} | {_fisher_row(late_f, 'luad', 'rank')} |

### 3.4 Gradient Alignment

The cosine similarity between Cox gradients and ranking-loss gradients indicates whether
the two objectives are complementary (high cosine) or conflicting (low/negative cosine).

![Gradient Alignment](figures/c_gradient_alignment.png)
![All Cosine Similarities](figures/l_all_cosines.png)

### 3.5 Gradient Variability

Gradient variability (standard deviation of mini-batch gradient norms over the probe set)
indicates training stability. Lower variability suggests more consistent gradient signal.

![Gradient Variability](figures/f_grad_variability.png)

### 3.6 Gradient Norm Ratio

The raw gradient norm ratio complements the Fisher ratio by measuring a simpler statistic
(L2 norm of the full batch gradient rather than per-event Fisher decomposition).

![Gradient Norm Ratio](figures/i_grad_norm_ratio.png)

---

## 4. Validation Dynamics

### 4.1 Validation C-index Trajectories

![Validation C-index (diagnostic epochs)](figures/d_val_cindex.png)
![Validation C-index (all epochs)](figures/k_val_ci_dense.png)

### 4.2 Training and Validation Loss

![Loss Curves](figures/j_loss_curves.png)

### 4.3 Loss Weight Evolution

For the adaptive_aux method, loss weights can evolve during training:

![Weight Evolution](figures/m_weight_evolution.png)

---

## 5. Early Fisher Ratio vs Test Performance

![Fisher vs C-index](figures/h_fisher_vs_cindex_heatmap.png)

This analysis tests whether early-epoch (epochs 1--5) Fisher ratio predicts final test
performance improvement. The left panel shows individual fold-level scatter; the right
panel shows the mean per cancer/method.

---

## 6. Per-Cancer Analysis

### 6.1 BRCA: Does adaptive_aux improve weaker modality learning?

"""

    # BRCA analysis
    brca_base_ci = ci[(ci["dataset"] == "brca") & (ci["method"] == "baseline")].iloc[0]
    brca_aux_ci = ci[(ci["dataset"] == "brca") & (ci["method"] == "adaptive_aux")].iloc[0]
    brca_rank_ci = ci[(ci["dataset"] == "brca") & (ci["method"] == "rank")].iloc[0]

    brca_early_base_fr = early_f[(early_f["cancer"] == "brca") & (early_f["method"] == "baseline")].iloc[0]["mean"]
    brca_early_aux_fr = early_f[(early_f["cancer"] == "brca") & (early_f["method"] == "adaptive_aux")].iloc[0]["mean"]

    # BRCA unimodal gap
    brca_unimodal = unimodal[unimodal["cancer"] == "brca"]

    report += f"""BRCA is the largest dataset (1,124 patients) and uses UNI 1024-dim image features
(vs 1536-dim UNI2-h for KIRC/LUAD), creating an inherent dimensionality asymmetry.

**Test C-index**: Baseline = {brca_base_ci['mean']:.4f}, Adaptive Aux = {brca_aux_ci['mean']:.4f}, Rank = {brca_rank_ci['mean']:.4f}

**Fisher ratio analysis**: The early-epoch Fisher ratio for baseline is {brca_early_base_fr:.3f}, while
adaptive_aux shows {brca_early_aux_fr:.3f}. """

    if brca_early_aux_fr < brca_early_base_fr:
        report += "The adaptive auxiliary heads appear to reduce the gene-over-image imbalance early in training."
    elif brca_early_aux_fr > brca_early_base_fr:
        report += "The adaptive auxiliary heads do NOT reduce the Fisher ratio imbalance; the gene encoder still dominates."
    else:
        report += "The Fisher ratios are similar between methods."

    # Check unimodal gap
    for _, row in brca_unimodal.iterrows():
        report += f"\n- {METHOD_LABELS[row['method']]}: Unimodal gap (gene - img) at epoch 50 = {row['mean']:+.4f}"

    report += f"""

**Interpretation**: """

    brca_delta_aux = paired[paired["cancer"] == "brca"]
    brca_delta_aux_row = brca_delta_aux[brca_delta_aux["method"] == "adaptive_aux"].iloc[0]
    if brca_delta_aux_row["delta_ci"] > 0.01:
        report += f"Adaptive auxiliary heads provide a meaningful improvement (+{brca_delta_aux_row['delta_ci']:.4f} C-index) on BRCA, suggesting that explicit unimodal supervision helps the weaker modality learn."
    elif brca_delta_aux_row["delta_ci"] > -0.01:
        report += f"Adaptive auxiliary heads show negligible effect ({brca_delta_aux_row['delta_ci']:+.4f} C-index) on BRCA. The gradient-level changes do not translate to test performance."
    else:
        report += f"Adaptive auxiliary heads slightly hurt performance ({brca_delta_aux_row['delta_ci']:+.4f} C-index) on BRCA, possibly due to optimization interference."

    report += """

### 6.2 KIRC: Does baseline already have balanced gradients?

"""

    kirc_early_base_fr = early_f[(early_f["cancer"] == "kirc") & (early_f["method"] == "baseline")].iloc[0]["mean"]
    kirc_base_ci = ci[(ci["dataset"] == "kirc") & (ci["method"] == "baseline")].iloc[0]
    kirc_aux_ci = ci[(ci["dataset"] == "kirc") & (ci["method"] == "adaptive_aux")].iloc[0]
    kirc_rank_ci = ci[(ci["dataset"] == "kirc") & (ci["method"] == "rank")].iloc[0]

    report += f"""KIRC (519 patients) has historically shown more balanced modality contributions.

**Test C-index**: Baseline = {kirc_base_ci['mean']:.4f}, Adaptive Aux = {kirc_aux_ci['mean']:.4f}, Rank = {kirc_rank_ci['mean']:.4f}

**Fisher ratio**: The baseline early-epoch Fisher ratio is {kirc_early_base_fr:.3f}. """

    if abs(kirc_early_base_fr - 1.0) < 0.5:
        report += "This is relatively close to 1.0, confirming that KIRC has more naturally balanced gradient flow between modalities."
    elif kirc_early_base_fr > 1.5:
        report += f"This indicates gene-dominant learning even in KIRC, contrary to prior expectations."
    else:
        report += f"This indicates image-dominant learning in KIRC."

    kirc_delta = paired[paired["cancer"] == "kirc"]
    for _, row in kirc_delta.iterrows():
        report += f"\n- {METHOD_LABELS[row['method']]} vs baseline: {row['delta_ci']:+.4f} (p={row['p_value']:.3f})"

    report += f"""

**Interpretation**: """

    best_kirc = max(
        [(kirc_base_ci['mean'], "Baseline"),
         (kirc_aux_ci['mean'], "Adaptive Aux"),
         (kirc_rank_ci['mean'], "Rank")],
        key=lambda x: x[0]
    )
    report += f"The best method for KIRC is {best_kirc[1]} ({best_kirc[0]:.4f}). "

    if best_kirc[1] != "Baseline":
        report += "This suggests auxiliary objectives can help even when modalities are somewhat balanced."
    else:
        report += "The baseline performs best, suggesting auxiliary objectives add noise when modalities are already balanced."

    report += """

### 6.3 LUAD: Does ranking loss affect gradient geometry?

"""

    luad_base_ci = ci[(ci["dataset"] == "luad") & (ci["method"] == "baseline")].iloc[0]
    luad_aux_ci = ci[(ci["dataset"] == "luad") & (ci["method"] == "adaptive_aux")].iloc[0]
    luad_rank_ci = ci[(ci["dataset"] == "luad") & (ci["method"] == "rank")].iloc[0]

    report += f"""LUAD (522 patients) serves as the test case for whether pairwise ranking loss provides
complementary gradient information to the Cox partial likelihood.

**Test C-index**: Baseline = {luad_base_ci['mean']:.4f}, Adaptive Aux = {luad_aux_ci['mean']:.4f}, Rank = {luad_rank_ci['mean']:.4f}

"""

    # Check cosine similarity for rank method on LUAD
    luad_rank_cosine = diag[(diag["cancer"] == "luad") & (diag["method"] == "rank")]["cosine_cox_rank"]
    mean_cosine = luad_rank_cosine.mean()
    report += f"""**Gradient alignment**: The mean cosine similarity between Cox and rank gradients on LUAD
is {mean_cosine:.4f}. """

    if mean_cosine > 0.7:
        report += "This high alignment suggests the ranking loss provides largely redundant gradient information — it pushes in the same direction as Cox."
    elif mean_cosine > 0.3:
        report += "This moderate alignment suggests the ranking loss provides partially complementary information, with some unique gradient directions."
    elif mean_cosine > 0:
        report += "This low positive alignment suggests the ranking loss provides substantially different gradient information from Cox."
    else:
        report += "The negative/near-zero alignment suggests the ranking loss conflicts with Cox gradients."

    luad_delta = paired[paired["cancer"] == "luad"]
    for _, row in luad_delta.iterrows():
        report += f"\n- {METHOD_LABELS[row['method']]} vs baseline: {row['delta_ci']:+.4f} (p={row['p_value']:.3f})"

    report += """

---

## 7. Cross-Cancer Patterns

### 7.1 Fisher Ratio as Predictor of Intervention Benefit

"""

    # Compute correlation between early Fisher ratio deviation from 1 and CI delta
    merged_for_corr = merged_heatmap[merged_heatmap["method"] != "baseline"].copy()
    merged_for_corr["fisher_deviation"] = np.abs(
        np.log(merged_for_corr["early_fisher_ratio"].clip(lower=0.01)))
    if len(merged_for_corr) >= 4:
        r, p = stats.pearsonr(merged_for_corr["fisher_deviation"],
                              merged_for_corr["ci_delta"])
        report += f"""We tested whether the absolute log Fisher ratio (deviation from balance) at epochs 1-5
predicts the test C-index improvement of interventions over baseline.

**Pearson correlation**: r = {r:.4f}, p = {p:.4f}

"""
        if p < 0.05:
            if r > 0:
                report += "There is a statistically significant positive correlation: cancers with more imbalanced early Fisher ratios benefit more from auxiliary objectives."
            else:
                report += "There is a statistically significant negative correlation: more imbalanced cancers actually benefit LESS from auxiliary objectives (unexpected)."
        else:
            report += "There is no statistically significant correlation between early Fisher ratio imbalance and intervention benefit. This suggests modality imbalance alone does not predict whether auxiliary objectives will help."
    else:
        report += "Insufficient data points for correlation analysis."

    report += """

### 7.2 Consistency of Method Rankings

"""

    # Check which method wins per cancer
    method_wins = {}
    for cancer in CANCERS:
        best_m = None
        best_ci = -1
        for method in METHODS:
            row = ci[(ci["dataset"] == cancer) & (ci["method"] == method)]
            if len(row) > 0 and row.iloc[0]["mean"] > best_ci:
                best_ci = row.iloc[0]["mean"]
                best_m = method
        method_wins[cancer] = (best_m, best_ci)
        report += f"- **{CANCER_LABELS[cancer]}**: Best = {METHOD_LABELS[best_m]} ({best_ci:.4f})\n"

    winning_methods = set(v[0] for v in method_wins.values())
    if len(winning_methods) == 1:
        report += f"\n**One method ({METHOD_LABELS[list(winning_methods)[0]]}) wins on all cancers**, indicating a robust intervention."
    else:
        report += f"\n**No single method wins on all cancers** — the best intervention is cancer-dependent."

    report += """

---

## 8. Limitations

1. **Mini-batch risk set bias**: The Fisher trace is computed on a 64-patient probe
   subset with mini-batch risk set approximation. This introduces bias because the Cox
   partial likelihood couples individuals through the risk set (patients at risk at each
   event time). A 64-patient probe may not adequately represent the full risk set structure
   of the training data.

2. **Frozen probes vs trained auxiliary heads**: For baseline and rank methods, unimodal
   C-index is estimated via frozen-feature linear Cox probes fitted post-hoc on training
   data. These have limited capacity and are not equivalent to the trained auxiliary heads
   used in adaptive_aux. Direct comparison of unimodal C-index across methods should be
   interpreted cautiously.

3. **Small fold count**: With only 3 folds, statistical power for paired tests is very
   limited (df=2). P-values should be interpreted as directional evidence, not rigorous
   significance claims.

4. **Diagnostic epoch spacing**: Diagnostics are only computed at 20 out of 50 epochs
   (1-15, 20, 25, 30, 40, 50), missing potential dynamics in epochs 16-19, 21-24, 26-29,
   31-39, 41-49.

5. **No hyperparameter tuning**: All methods use the same learning rate (0.001),
   weight decay (0.0001), and other hyperparameters. The relative performance of methods
   could change with per-method tuning.

6. **Single architecture**: Results are specific to the late-fusion architecture
   (separate encoders + fusion MLP). Early-fusion or attention-based architectures may
   show different gradient dynamics.

---

## 9. Go/No-Go Recommendation for InfoReg-Inspired Intervention

### Evidence Summary

"""

    # Compute overall statistics for go/no-go
    all_deltas = paired["delta_ci"].values
    mean_delta = np.mean(all_deltas)
    any_significant = any(paired["p_value"] < 0.05)
    any_positive = any(paired["delta_ci"] > 0.01)
    all_positive = all(paired["delta_ci"] > 0)

    report += f"""| Criterion | Finding |
|-----------|---------|
| Mean C-index delta (all interventions vs baseline) | {mean_delta:+.4f} |
| Any significant improvement (p < 0.05)? | {'Yes' if any_significant else 'No'} |
| Any meaningful improvement (>0.01)? | {'Yes' if any_positive else 'No'} |
| All interventions positive? | {'Yes' if all_positive else 'No'} |
"""

    # Fisher ratio analysis
    fisher_changes = []
    for cancer in CANCERS:
        base_early = early_f[(early_f["cancer"] == cancer) & (early_f["method"] == "baseline")].iloc[0]["mean"]
        for method in ["adaptive_aux", "rank"]:
            method_early = early_f[(early_f["cancer"] == cancer) & (early_f["method"] == method)].iloc[0]["mean"]
            fisher_changes.append({
                "cancer": cancer, "method": method,
                "baseline_fr": base_early, "method_fr": method_early,
                "fr_change": method_early - base_early
            })
    fisher_df = pd.DataFrame(fisher_changes)

    report += f"""
| Criterion | Finding |
|-----------|---------|
| Do interventions reduce Fisher imbalance? | {'Mixed' if fisher_df['fr_change'].std() > abs(fisher_df['fr_change'].mean()) else ('Yes' if fisher_df['fr_change'].mean() < 0 else 'No')} |
| Is gradient alignment (cox-rank) high? | {('Yes (>' + f'{diag[diag["method"]=="rank"]["cosine_cox_rank"].mean():.2f}' + ')') if diag[diag["method"]=="rank"]["cosine_cox_rank"].mean() > 0.5 else 'Moderate/Low'} |

### Recommendation

"""

    # Decision logic
    if any_significant and mean_delta > 0.01:
        report += """**GO**: There is statistically significant evidence that auxiliary objectives improve
test C-index. Proceed with a full InfoReg-inspired intervention, focusing on the method
that showed the most consistent improvement across cancers. Prioritize the adaptive_aux
approach and consider combining it with ranking loss."""
    elif mean_delta > 0.005 and any_positive:
        report += """**CONDITIONAL GO**: While no individual result reaches statistical significance
(limited by 3 folds), there are positive trends suggesting auxiliary objectives can help.
Recommend proceeding with a **larger-scale validation** (5+ folds, more cancers) before
committing to the full InfoReg intervention. Focus on:
1. Adaptive auxiliary heads (showed the most promise)
2. Cancer-specific tuning of auxiliary loss weights
3. Extending to LUSC and STAD for broader validation"""
    elif mean_delta > 0 and not all(paired["delta_ci"] < -0.005):
        report += """**CONDITIONAL GO (weak)**: The evidence is mixed but not negative. The auxiliary
objectives do not consistently hurt and show modest improvements on some cancers. Recommend:
1. A targeted investigation on the cancer(s) where improvements were observed
2. Exploring different auxiliary loss weight schedules
3. Considering whether the gradient-level changes (Fisher ratio, alignment) translate
   to real performance gains with better hyperparameter tuning"""
    elif mean_delta < -0.005:
        report += """**NO-GO**: The auxiliary objectives consistently hurt test performance. The gradient-level
diagnostics show changes in Fisher ratio and alignment, but these do not translate to
improved predictions. The additional complexity is not justified."""
    else:
        report += """**NO-GO (insufficient evidence)**: The interventions show negligible effect on test
C-index. While gradient-level diagnostics reveal interesting patterns in Fisher information
and gradient alignment, these do not translate to meaningful performance improvements.
The overhead of implementing and tuning auxiliary objectives is not currently justified.

Consider instead:
1. Architecture changes (attention-based fusion) that might better leverage both modalities
2. Pre-training or transfer learning to improve individual encoder quality
3. Larger training sets or data augmentation to reduce the noise floor"""

    report += """

---

## Appendix: Computational Details

"""

    # Wall time table
    report += "| Cancer | Method | Wall Time (s, mean +/- std) |\n"
    report += "|--------|--------|---------------------------|\n"
    for cancer in CANCERS:
        for method in METHODS:
            row = wall[(wall["dataset"] == cancer) & (wall["method"] == method)]
            if len(row) > 0:
                m, s = row.iloc[0]["mean"], row.iloc[0]["std"]
                report += f"| {CANCER_LABELS[cancer]} | {METHOD_LABELS[method]} | {m:.1f} +/- {s:.1f} |\n"

    report += f"""
**Total experiment wall time**: {total_wall:.0f} seconds ({total_wall/3600:.1f} hours)
**Mean per-run wall time**: {mean_wall:.0f} seconds ({mean_wall/60:.1f} minutes)
**Parameter counts**: Baseline/Rank = 1,451,523; Adaptive Aux = 1,492,741 (+2.8%)

---

*Report generated automatically by `experiments/samo/analyze_info_acquisition.py`*
"""

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Information Acquisition Diagnostics — Comprehensive Analysis")
    print("=" * 70)

    print("\n[1/6] Loading data...")
    diag = load_all_diagnostics()
    epochs_df = load_all_epochs()
    summaries = load_all_summaries()

    print(f"\n[2/6] Generating trajectory plots...")
    plot_gradient_sensitivity(diag)
    plot_fisher_trace(diag)
    plot_gradient_alignment(diag)
    plot_val_cindex(diag)
    plot_fisher_ratio(diag)
    plot_grad_variability(diag)
    plot_grad_norm_ratio(diag)
    plot_all_cosines(diag)
    plot_weight_evolution(diag)

    print(f"\n[3/6] Generating summary plots...")
    plot_test_cindex_bars(summaries)

    print(f"\n[4/6] Generating heatmap...")
    merged_heatmap = plot_fisher_vs_cindex_heatmap(diag, summaries)

    print(f"\n[5/6] Generating additional plots from epochs.csv...")
    plot_loss_curves(epochs_df)
    plot_val_ci_dense(epochs_df)

    print(f"\n[6/6] Computing statistics and generating report...")
    stats_dict = compute_statistics(diag, epochs_df, summaries)
    report = generate_report(stats_dict, merged_heatmap, diag, summaries)

    report_path = RESULTS_DIR / "final_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved report: {report_path}")

    # Also print summary to stdout
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nTest C-index (mean +/- std across 3 folds):")
    for cancer in CANCERS:
        print(f"\n  {CANCER_LABELS[cancer]}:")
        for method in METHODS:
            row = stats_dict["ci_table"]
            r = row[(row["dataset"] == cancer) & (row["method"] == method)]
            if len(r) > 0:
                m, s = r.iloc[0]["mean"], r.iloc[0]["std"]
                print(f"    {METHOD_LABELS[method]:20s}: {m:.4f} +/- {s:.4f}")

    print("\nPaired t-tests vs baseline:")
    for _, row in stats_dict["paired_tests"].iterrows():
        sig_marker = "*" if row["p_value"] < 0.05 else " "
        print(f"  {CANCER_LABELS[row['cancer']]:5s} {METHOD_LABELS[row['method']]:20s}: "
              f"delta={row['delta_ci']:+.4f}, t={row['t_stat']:.3f}, "
              f"p={row['p_value']:.3f} {sig_marker}")

    print(f"\nFigures saved to: {FIGURES_DIR}")
    print(f"Report saved to:  {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
