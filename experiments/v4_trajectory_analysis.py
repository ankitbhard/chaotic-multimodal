#!/usr/bin/env python3
"""
RA-SAMO v4 Trajectory Analysis.

Analyzes 350 existing runs (250 static + 100 v3) to determine whether
dynamic objective switching has measurable headroom over per-cancer-best static.

Outputs: results/v4_analysis/ with figures + analysis_report.md

Usage:
    python experiments/v4_trajectory_analysis.py
    python experiments/v4_trajectory_analysis.py --out results/v4_analysis
"""

import argparse
import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

try:
    from scipy.stats import pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

REPO_ROOT = Path(__file__).resolve().parents[1]

CANCERS = ["brca", "kirc", "luad", "lusc", "stad"]
CANCER_LABELS = {"brca": "BRCA", "kirc": "KIRC", "luad": "LUAD",
                 "lusc": "LUSC", "stad": "STAD"}
STATIC_METHODS = ["baseline", "aux", "rank", "adaptive_aux", "aux_rank"]
V3_METHODS = ["ra_samo_v3_0", "ra_samo_v3_1", "ra_samo_v3_2", "ra_samo_v3_3"]
FOLDS = list(range(5))

# Known per-cancer best static methods (from prior results)
PER_CANCER_BEST = {
    "brca": "adaptive_aux",
    "stad": "adaptive_aux",
    "kirc": "baseline",
    "luad": "rank",
    "lusc": "rank",
}

# v3 mode -> static method mapping
V3_MODE_MAP = {"cox": "baseline", "aux": "adaptive_aux", "rank": "rank"}

PALETTE = [
    "#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800",
    "#00BCD4", "#E91E63", "#8BC34A", "#3F51B5", "#795548",
]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 1.5,
    "font.family": "sans-serif",
})


# ═══════════════════════════════════════════════════════════════════════════
# Section 1: Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_epochs_csv(path):
    """Load a single epochs.csv into list of dicts with numeric conversion."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
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
    return rows


def load_all_data(static_dir, v3_dir):
    """Load all epoch-level data from both result directories.

    Returns dict: (cancer, fold, method) -> list of epoch dicts (sorted by epoch)
    """
    data = {}
    summaries = {}

    for base_dir, methods in [(static_dir, None), (v3_dir, None)]:
        if not base_dir.exists():
            print(f"  WARNING: {base_dir} does not exist, skipping")
            continue
        for cancer in CANCERS:
            cancer_dir = base_dir / cancer
            if not cancer_dir.exists():
                continue
            for fold in FOLDS:
                fold_dir = cancer_dir / f"fold_{fold}"
                if not fold_dir.exists():
                    continue
                for method_dir in sorted(fold_dir.iterdir()):
                    if not method_dir.is_dir():
                        continue
                    method = method_dir.name
                    epochs_path = method_dir / "epochs.csv"
                    summary_path = method_dir / "summary.json"
                    if epochs_path.exists():
                        rows = load_epochs_csv(epochs_path)
                        rows.sort(key=lambda r: r.get("epoch", 0))
                        data[(cancer, fold, method)] = rows
                    if summary_path.exists():
                        try:
                            sdata = json.loads(summary_path.read_text())
                            if sdata.get("completed"):
                                summaries[(cancer, fold, method)] = sdata
                        except (json.JSONDecodeError, OSError):
                            pass

    print(f"Loaded {len(data)} epoch trajectories, {len(summaries)} summaries")
    return data, summaries


# ═══════════════════════════════════════════════════════════════════════════
# Section 2: Per-Fold Oracle Analysis (1B)
# ═══════════════════════════════════════════════════════════════════════════

def oracle_analysis(data, summaries):
    """Compare val_ci across static methods per (cancer, fold).

    Returns:
        oracle_table: dict cancer -> {headroom_per_fold, headroom_val_oracle,
                                       leader_changes, per_fold_best}
    """
    results = {}

    for cancer in CANCERS:
        fold_test_cis = {m: [] for m in STATIC_METHODS}
        leader_change_counts = []
        val_oracle_test_cis = []
        per_fold_best_test_cis = []
        per_cancer_best_test_cis = []

        best_method = PER_CANCER_BEST.get(cancer, "baseline")

        for fold in FOLDS:
            # Collect val_ci trajectories for static methods
            trajectories = {}
            for method in STATIC_METHODS:
                key = (cancer, fold, method)
                if key in data:
                    epochs = data[key]
                    trajectories[method] = {
                        int(r["epoch"]): r["val_ci"]
                        for r in epochs if r.get("val_ci") is not None
                    }

            if not trajectories:
                continue

            # Leader at each epoch — how often does it change?
            all_epochs = sorted(set().union(*[set(t.keys()) for t in trajectories.values()]))
            prev_leader = None
            changes = 0
            for ep in all_epochs:
                ep_vals = {m: trajectories[m].get(ep) for m in trajectories
                           if trajectories[m].get(ep) is not None}
                if ep_vals:
                    leader = max(ep_vals, key=ep_vals.get)
                    if prev_leader is not None and leader != prev_leader:
                        changes += 1
                    prev_leader = leader
            leader_change_counts.append(changes)

            # (a) Per-cancer best static test_ci
            key_a = (cancer, fold, best_method)
            test_ci_a = summaries.get(key_a, {}).get("test_ci")

            # (b) Per-fold best static test_ci (cheating upper bound)
            fold_tests = {}
            for method in STATIC_METHODS:
                key_b = (cancer, fold, method)
                if key_b in summaries:
                    fold_tests[method] = summaries[key_b]["test_ci"]
            test_ci_b = max(fold_tests.values()) if fold_tests else None

            # (c) Val-ci oracle: pick method with best val_ci, use its test_ci
            best_val_methods = {}
            for method in STATIC_METHODS:
                key_c = (cancer, fold, method)
                if key_c in summaries:
                    best_val_methods[method] = summaries[key_c]["best_val_ci"]
            if best_val_methods:
                oracle_method = max(best_val_methods, key=best_val_methods.get)
                test_ci_c = summaries.get((cancer, fold, oracle_method), {}).get("test_ci")
            else:
                test_ci_c = None

            if test_ci_a is not None:
                per_cancer_best_test_cis.append(test_ci_a)
            if test_ci_b is not None:
                per_fold_best_test_cis.append(test_ci_b)
            if test_ci_c is not None:
                val_oracle_test_cis.append(test_ci_c)

            for method in STATIC_METHODS:
                key_m = (cancer, fold, method)
                if key_m in summaries:
                    fold_test_cis[method].append(summaries[key_m]["test_ci"])

        mean_a = np.mean(per_cancer_best_test_cis) if per_cancer_best_test_cis else 0
        mean_b = np.mean(per_fold_best_test_cis) if per_fold_best_test_cis else 0
        mean_c = np.mean(val_oracle_test_cis) if val_oracle_test_cis else 0

        results[cancer] = {
            "per_cancer_best": best_method,
            "per_cancer_best_mean_ci": float(mean_a),
            "per_fold_best_mean_ci": float(mean_b),
            "val_oracle_mean_ci": float(mean_c),
            "headroom_per_fold": float(mean_b - mean_a),
            "headroom_val_oracle": float(mean_c - mean_a),
            "leader_changes_per_fold": [float(x) for x in leader_change_counts],
            "mean_leader_changes": float(np.mean(leader_change_counts)) if leader_change_counts else 0,
            "method_means": {m: float(np.mean(v)) if v else 0
                             for m, v in fold_test_cis.items()},
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Section 3: Feature-Outcome Correlation (1C)
# ═══════════════════════════════════════════════════════════════════════════

NUMERIC_FEATURES = [
    "train_fusion_loss", "train_img_loss", "train_gene_loss", "train_rank_loss",
    "val_loss", "weight_img", "weight_gene", "weight_rank", "lr",
    "val_ci_fused", "val_ci_image_aux", "val_ci_gene_aux",
    "val_ci_image_ablated", "val_ci_gene_ablated",
    "val_image_ablation_delta", "val_gene_ablation_delta",
    "progress_scale_img", "progress_scale_gene",
    "epoch",
]

V3_FEATURES = [
    "v3_event_rate", "v3_pair_density",
    "v3_align_aux", "v3_align_rank",
    "v3_align_aux_rel", "v3_align_rank_rel",
]


def fisher_z_combine(rs, ns):
    """Combine Pearson r values across folds via Fisher z-transform."""
    zs = [np.arctanh(np.clip(r, -0.999, 0.999)) for r in rs]
    weights = [n - 3 for n in ns]
    if sum(weights) <= 0:
        return 0.0
    z_combined = sum(z * w for z, w in zip(zs, weights)) / sum(weights)
    return float(np.tanh(z_combined))


def feature_outcome_correlations(data):
    """Correlate features with delta_val_ci for each (cancer, method).

    Returns: dict cancer -> method -> feature -> {raw_r, partial_r, n_obs}
    """
    if not HAS_SCIPY:
        print("  scipy not available, skipping feature correlations")
        return {}

    results = {}
    all_features = NUMERIC_FEATURES + V3_FEATURES

    for cancer in CANCERS:
        results[cancer] = {}
        # Pool across all methods that have rich features
        all_methods = set()
        for (c, f, m), rows in data.items():
            if c == cancer:
                all_methods.add(m)

        for method in sorted(all_methods):
            fold_correlations = defaultdict(lambda: {"rs": [], "partial_rs": [], "ns": []})

            for fold in FOLDS:
                key = (cancer, fold, method)
                if key not in data:
                    continue
                rows = data[key]
                if len(rows) < 5:
                    continue

                # Compute delta_val_ci
                val_cis = [r.get("val_ci") for r in rows]
                delta_val_ci = []
                for i in range(len(val_cis) - 1):
                    if val_cis[i] is not None and val_cis[i + 1] is not None:
                        delta_val_ci.append(val_cis[i + 1] - val_cis[i])
                    else:
                        delta_val_ci.append(None)

                # For each feature, compute correlation with delta_val_ci
                for feat in all_features:
                    feat_vals = [r.get(feat) for r in rows[:-1]]  # align with delta
                    cur_val_ci = val_cis[:-1]

                    # Filter to valid pairs
                    valid = [(fv, dv, cv) for fv, dv, cv in zip(feat_vals, delta_val_ci, cur_val_ci)
                             if fv is not None and dv is not None and cv is not None
                             and np.isfinite(fv) and np.isfinite(dv) and np.isfinite(cv)]

                    if len(valid) < 5:
                        continue

                    fv_arr = np.array([v[0] for v in valid])
                    dv_arr = np.array([v[1] for v in valid])
                    cv_arr = np.array([v[2] for v in valid])

                    # Skip constant arrays
                    if np.std(fv_arr) < 1e-12 or np.std(dv_arr) < 1e-12:
                        continue

                    # Raw Pearson r
                    r_raw, _ = pearsonr(fv_arr, dv_arr)

                    # Partial correlation controlling for current val_ci
                    r_partial = r_raw
                    if np.std(cv_arr) > 1e-12:
                        # Residualize both feature and delta on current val_ci
                        coef_f = np.polyfit(cv_arr, fv_arr, 1)
                        coef_d = np.polyfit(cv_arr, dv_arr, 1)
                        resid_f = fv_arr - np.polyval(coef_f, cv_arr)
                        resid_d = dv_arr - np.polyval(coef_d, cv_arr)
                        if np.std(resid_f) > 1e-12 and np.std(resid_d) > 1e-12:
                            r_partial, _ = pearsonr(resid_f, resid_d)

                    fold_correlations[feat]["rs"].append(r_raw)
                    fold_correlations[feat]["partial_rs"].append(r_partial)
                    fold_correlations[feat]["ns"].append(len(valid))

            # Combine across folds
            method_results = {}
            for feat, vals in fold_correlations.items():
                if vals["rs"]:
                    method_results[feat] = {
                        "raw_r": fisher_z_combine(vals["rs"], vals["ns"]),
                        "partial_r": fisher_z_combine(vals["partial_rs"], vals["ns"]),
                        "n_folds": len(vals["rs"]),
                        "n_obs": sum(vals["ns"]),
                    }
            if method_results:
                results[cancer][method] = method_results

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Section 4: v3 Mode Decision Audit (1D)
# ═══════════════════════════════════════════════════════════════════════════

def v3_mode_audit(data, summaries):
    """For each v3 run, check if its mode choice was optimal vs static methods.

    Returns: dict cancer -> v3_method -> {optimality_rate, mean_gap, details}
    """
    results = {}

    for cancer in CANCERS:
        results[cancer] = {}
        for v3_method in V3_METHODS:
            fold_optimality = []
            fold_gaps = []
            fold_details = []

            for fold in FOLDS:
                v3_key = (cancer, fold, v3_method)
                if v3_key not in data:
                    continue

                v3_rows = data[v3_key]
                # Build static val_ci lookup per epoch
                static_by_epoch = {}
                for method in STATIC_METHODS:
                    s_key = (cancer, fold, method)
                    if s_key in data:
                        for r in data[s_key]:
                            ep = int(r["epoch"])
                            if ep not in static_by_epoch:
                                static_by_epoch[ep] = {}
                            if r.get("val_ci") is not None:
                                static_by_epoch[ep][method] = r["val_ci"]

                optimal_count = 0
                total_count = 0
                gaps = []

                for r in v3_rows:
                    v3_mode = r.get("v3_mode")
                    if not v3_mode or v3_mode not in V3_MODE_MAP:
                        continue
                    ep = int(r["epoch"])
                    mapped_method = V3_MODE_MAP[v3_mode]

                    # What did each static method achieve at this epoch?
                    ep_statics = static_by_epoch.get(ep, {})
                    if not ep_statics:
                        continue

                    best_static = max(ep_statics, key=ep_statics.get)
                    best_val = ep_statics[best_static]
                    chosen_val = ep_statics.get(mapped_method, None)

                    total_count += 1
                    if mapped_method == best_static:
                        optimal_count += 1
                    elif chosen_val is not None:
                        gaps.append(best_val - chosen_val)

                if total_count > 0:
                    fold_optimality.append(optimal_count / total_count)
                    fold_gaps.extend(gaps)

            if fold_optimality:
                results[cancer][v3_method] = {
                    "optimality_rate": float(np.mean(fold_optimality)),
                    "mean_gap_when_suboptimal": float(np.mean(fold_gaps)) if fold_gaps else 0.0,
                    "n_folds": len(fold_optimality),
                }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Section 5: Transition Point Analysis (1E)
# ═══════════════════════════════════════════════════════════════════════════

def transition_analysis(data):
    """Detect plateaus, divergence points, and benefit windows.

    Returns: dict cancer -> {plateaus, divergences, benefit_windows}
    """
    results = {}

    for cancer in CANCERS:
        plateau_epochs = defaultdict(list)  # fold -> list of plateau start epochs
        divergence_epochs = defaultdict(list)
        aux_benefit_epochs = defaultdict(list)
        rank_benefit_epochs = defaultdict(list)

        for fold in FOLDS:
            # Collect static trajectories
            trajectories = {}
            for method in STATIC_METHODS:
                key = (cancer, fold, method)
                if key in data:
                    traj = {}
                    for r in data[key]:
                        ep = int(r["epoch"])
                        if r.get("val_ci") is not None:
                            traj[ep] = r["val_ci"]
                    if traj:
                        trajectories[method] = traj

            if len(trajectories) < 2:
                continue

            all_epochs = sorted(set().union(*[set(t.keys()) for t in trajectories.values()]))

            # Plateau detection: val_ci improves < 0.001 for 5 consecutive epochs
            for method, traj in trajectories.items():
                sorted_eps = sorted(traj.keys())
                streak = 0
                for i in range(1, len(sorted_eps)):
                    improvement = traj[sorted_eps[i]] - traj[sorted_eps[i - 1]]
                    if improvement < 0.001:
                        streak += 1
                        if streak >= 5:
                            plateau_epochs[fold].append(sorted_eps[i - 4])
                            streak = 0  # reset after recording
                    else:
                        streak = 0

            # Divergence: gap between best/worst static > 0.02
            for ep in all_epochs:
                vals = [trajectories[m][ep] for m in trajectories if ep in trajectories[m]]
                if len(vals) >= 2 and max(vals) - min(vals) > 0.02:
                    divergence_epochs[fold].append(ep)

            # Aux/rank benefit windows
            aux_methods = {"aux", "adaptive_aux", "aux_rank"}
            rank_methods = {"rank", "aux_rank"}
            for ep in all_epochs:
                ep_vals = {m: trajectories[m][ep] for m in trajectories if ep in trajectories[m]}
                if not ep_vals:
                    continue
                leader = max(ep_vals, key=ep_vals.get)
                if leader in aux_methods:
                    aux_benefit_epochs[fold].append(ep)
                if leader in rank_methods:
                    rank_benefit_epochs[fold].append(ep)

        results[cancer] = {
            "plateau_epochs_per_fold": {f: sorted(set(v)) for f, v in plateau_epochs.items()},
            "divergence_epochs_per_fold": {f: v for f, v in divergence_epochs.items()},
            "aux_benefit_epochs_per_fold": {f: v for f, v in aux_benefit_epochs.items()},
            "rank_benefit_epochs_per_fold": {f: v for f, v in rank_benefit_epochs.items()},
            "mean_divergence_onset": float(np.mean([
                min(v) for v in divergence_epochs.values() if v
            ])) if any(divergence_epochs.values()) else None,
            "aux_benefit_fraction": float(np.mean([
                len(v) / 50 for v in aux_benefit_epochs.values()
            ])) if aux_benefit_epochs else 0,
            "rank_benefit_fraction": float(np.mean([
                len(v) / 50 for v in rank_benefit_epochs.values()
            ])) if rank_benefit_epochs else 0,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Section 6: Cross-Fold Consistency (1F)
# ═══════════════════════════════════════════════════════════════════════════

def cross_fold_consistency(data):
    """Compute inter-fold correlation of val_ci trajectories.

    Returns: dict cancer -> method -> {mean_corr, std_corr, n_folds}
    """
    results = {}

    for cancer in CANCERS:
        results[cancer] = {}
        all_methods = set()
        for (c, f, m) in data:
            if c == cancer:
                all_methods.add(m)

        for method in sorted(all_methods):
            fold_trajectories = []
            for fold in FOLDS:
                key = (cancer, fold, method)
                if key in data:
                    vals = [r.get("val_ci") for r in data[key]]
                    if all(v is not None for v in vals) and len(vals) > 0:
                        fold_trajectories.append(np.array(vals))

            if len(fold_trajectories) < 2:
                continue

            # Ensure same length (trim to min)
            min_len = min(len(t) for t in fold_trajectories)
            fold_trajectories = [t[:min_len] for t in fold_trajectories]

            # Pairwise correlations
            corrs = []
            for i in range(len(fold_trajectories)):
                for j in range(i + 1, len(fold_trajectories)):
                    if np.std(fold_trajectories[i]) > 1e-12 and np.std(fold_trajectories[j]) > 1e-12:
                        c = np.corrcoef(fold_trajectories[i], fold_trajectories[j])[0, 1]
                        if np.isfinite(c):
                            corrs.append(c)

            if corrs:
                results[cancer][method] = {
                    "mean_corr": float(np.mean(corrs)),
                    "std_corr": float(np.std(corrs)),
                    "n_folds": len(fold_trajectories),
                }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Section 7: Figure Generation
# ═══════════════════════════════════════════════════════════════════════════

def _save_fig(fig, name, fig_dir):
    fig.savefig(fig_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png")


def plot_oracle_headroom(oracle_results, fig_dir):
    """Bar chart: per-cancer oracle headroom (per-fold vs val-oracle)."""
    cancers = [c for c in CANCERS if c in oracle_results]
    if not cancers:
        return

    x = np.arange(len(cancers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 3))
    headroom_fold = [oracle_results[c]["headroom_per_fold"] for c in cancers]
    headroom_val = [oracle_results[c]["headroom_val_oracle"] for c in cancers]

    ax.bar(x - width / 2, headroom_fold, width, label="Per-fold best", color="#2196F3")
    ax.bar(x + width / 2, headroom_val, width, label="Val-CI oracle", color="#FF5722")
    ax.set_xticks(x)
    ax.set_xticklabels([CANCER_LABELS[c] for c in cancers])
    ax.set_ylabel("Headroom over per-cancer best (C-index)")
    ax.set_title("Oracle Headroom Analysis")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save_fig(fig, "oracle_headroom", fig_dir)


def plot_leader_changes(oracle_results, fig_dir):
    """Box/scatter plot of leader changes per fold per cancer."""
    cancers = [c for c in CANCERS if c in oracle_results]
    if not cancers:
        return

    fig, ax = plt.subplots(figsize=(5, 3))
    data_boxes = []
    labels = []
    for c in cancers:
        changes = oracle_results[c]["leader_changes_per_fold"]
        if changes:
            data_boxes.append(changes)
            labels.append(CANCER_LABELS[c])

    if data_boxes:
        bp = ax.boxplot(data_boxes, labels=labels, patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(PALETTE[i % len(PALETTE)])
            patch.set_alpha(0.7)
        ax.set_ylabel("Leader changes per fold (50 epochs)")
        ax.set_title("Method Leadership Volatility")
    fig.tight_layout()
    _save_fig(fig, "leader_changes", fig_dir)


def plot_feature_heatmap(correlations, fig_dir):
    """Heatmap of feature-delta_val_ci correlations by cancer."""
    if not correlations:
        return

    # Collect all features that appear in any cancer (for the most data-rich method)
    all_feats = set()
    cancer_method_map = {}
    for cancer in CANCERS:
        if cancer not in correlations:
            continue
        # Pick the method with most features (likely ra_samo or adaptive_aux)
        best_method = max(correlations[cancer],
                         key=lambda m: len(correlations[cancer][m]),
                         default=None)
        if best_method:
            cancer_method_map[cancer] = best_method
            all_feats.update(correlations[cancer][best_method].keys())

    if not all_feats or not cancer_method_map:
        return

    feats = sorted(all_feats)
    cancers = [c for c in CANCERS if c in cancer_method_map]

    for corr_type, title_suffix in [("raw_r", "Raw"), ("partial_r", "Partial")]:
        matrix = np.full((len(feats), len(cancers)), np.nan)
        for j, cancer in enumerate(cancers):
            method = cancer_method_map[cancer]
            for i, feat in enumerate(feats):
                entry = correlations[cancer].get(method, {}).get(feat)
                if entry:
                    matrix[i, j] = entry[corr_type]

        fig, ax = plt.subplots(figsize=(max(4, len(cancers) * 1.2), max(4, len(feats) * 0.35)))
        mask = np.isnan(matrix)
        plot_data = np.where(mask, 0, matrix)
        im = ax.imshow(plot_data, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        ax.set_xticks(np.arange(len(cancers)))
        ax.set_xticklabels([CANCER_LABELS[c] for c in cancers])
        ax.set_yticks(np.arange(len(feats)))
        ax.set_yticklabels(feats, fontsize=5)
        # Annotate
        for i in range(len(feats)):
            for j in range(len(cancers)):
                if not mask[i, j]:
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                            fontsize=4, color="white" if abs(matrix[i, j]) > 0.3 else "black")
        ax.set_title(f"Feature → delta_val_ci Correlation ({title_suffix})")
        fig.colorbar(im, ax=ax, shrink=0.6)
        fig.tight_layout()
        _save_fig(fig, f"feature_correlation_{corr_type}", fig_dir)


def plot_val_ci_trajectories(data, fig_dir):
    """Val CI trajectories: static methods, one panel per cancer, folds as thin lines."""
    n = len(CANCERS)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3), squeeze=False)

    for i, cancer in enumerate(CANCERS):
        ax = axes[0, i]
        for j, method in enumerate(STATIC_METHODS):
            color = PALETTE[j % len(PALETTE)]
            fold_plotted = False
            for fold in FOLDS:
                key = (cancer, fold, method)
                if key not in data:
                    continue
                epochs = [r["epoch"] for r in data[key]]
                vals = [r["val_ci"] for r in data[key]]
                label = method if not fold_plotted else None
                ax.plot(epochs, vals, color=color, alpha=0.3, linewidth=0.8, label=label)
                fold_plotted = True
        ax.set_title(CANCER_LABELS[cancer])
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel("Val C-index")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 5),
                   bbox_to_anchor=(0.5, -0.08), fontsize=6)
    fig.suptitle("Val C-index Trajectories (Static Methods)", fontsize=10, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "val_ci_trajectories", fig_dir)


def plot_cross_fold_stability(consistency, fig_dir):
    """Heatmap of cross-fold correlation by cancer x method."""
    all_methods = sorted(set(
        m for cancer in consistency.values() for m in cancer
    ))
    cancers = [c for c in CANCERS if c in consistency]
    if not all_methods or not cancers:
        return

    matrix = np.full((len(cancers), len(all_methods)), np.nan)
    for i, cancer in enumerate(cancers):
        for j, method in enumerate(all_methods):
            entry = consistency.get(cancer, {}).get(method)
            if entry:
                matrix[i, j] = entry["mean_corr"]

    fig, ax = plt.subplots(figsize=(max(4, len(all_methods) * 0.7), 3))
    plot_data = np.where(np.isnan(matrix), 0, matrix)
    im = ax.imshow(plot_data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(all_methods)))
    ax.set_xticklabels(all_methods, rotation=45, ha="right", fontsize=5)
    ax.set_yticks(np.arange(len(cancers)))
    ax.set_yticklabels([CANCER_LABELS[c] for c in cancers])
    for i in range(len(cancers)):
        for j in range(len(all_methods)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=4)
    ax.set_title("Cross-Fold Trajectory Correlation")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    _save_fig(fig, "cross_fold_stability", fig_dir)


# ═══════════════════════════════════════════════════════════════════════════
# Section 8: Report Generation (1G)
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(oracle_results, correlations, v3_audit, transitions,
                    consistency, output_dir):
    """Generate analysis_report.md with go/no-go recommendation."""
    lines = [
        "# RA-SAMO v4 Trajectory Analysis Report",
        "",
        "Analysis of 350 existing runs (250 static + 100 v3) across 5 cancers, 5 folds, 50 epochs.",
        "",
    ]

    # ── Oracle Analysis ──
    lines.append("## 1. Oracle Headroom Analysis")
    lines.append("")
    lines.append("| Cancer | Per-Cancer Best | Mean CI | Per-Fold Best CI | Val Oracle CI | Headroom (fold) | Headroom (val) | Leader Changes |")
    lines.append("|--------|----------------|---------|------------------|---------------|-----------------|----------------|----------------|")
    total_headroom_fold = []
    total_headroom_val = []
    for cancer in CANCERS:
        if cancer not in oracle_results:
            continue
        o = oracle_results[cancer]
        total_headroom_fold.append(o["headroom_per_fold"])
        total_headroom_val.append(o["headroom_val_oracle"])
        lines.append(
            f"| {CANCER_LABELS[cancer]} | {o['per_cancer_best']} | {o['per_cancer_best_mean_ci']:.4f} "
            f"| {o['per_fold_best_mean_ci']:.4f} | {o['val_oracle_mean_ci']:.4f} "
            f"| {o['headroom_per_fold']:+.4f} | {o['headroom_val_oracle']:+.4f} "
            f"| {o['mean_leader_changes']:.1f} |"
        )
    if total_headroom_fold:
        lines.append(f"| **Mean** | | | | | **{np.mean(total_headroom_fold):+.4f}** | **{np.mean(total_headroom_val):+.4f}** | |")
    lines.append("")

    lines.append("**Interpretation**: Per-fold headroom is the theoretical maximum if we could cheat and "
                 "pick the best method per fold. Val-oracle headroom is what we get by picking the method "
                 "with best validation CI (achievable without cheating).")
    lines.append("")

    # ── Per-cancer method means ──
    lines.append("### Static Method Mean Test C-index")
    lines.append("")
    header = "| Method | " + " | ".join(CANCER_LABELS[c] for c in CANCERS) + " | Mean |"
    sep = "|--------|" + "|".join("------" for _ in CANCERS) + "|------|"
    lines.extend([header, sep])
    for method in STATIC_METHODS:
        vals = []
        cells = []
        for cancer in CANCERS:
            v = oracle_results.get(cancer, {}).get("method_means", {}).get(method, 0)
            vals.append(v)
            cells.append(f"{v:.4f}")
        lines.append(f"| {method} | " + " | ".join(cells) + f" | {np.mean(vals):.4f} |")
    lines.append("")

    # ── v3 Mode Audit ──
    lines.append("## 2. v3 Mode Decision Audit")
    lines.append("")
    lines.append("| Cancer | v3 Variant | Optimality Rate | Mean Gap When Sub-optimal |")
    lines.append("|--------|------------|-----------------|---------------------------|")
    for cancer in CANCERS:
        if cancer not in v3_audit:
            continue
        for v3m in V3_METHODS:
            entry = v3_audit[cancer].get(v3m)
            if entry:
                lines.append(
                    f"| {CANCER_LABELS[cancer]} | {v3m} | {entry['optimality_rate']:.1%} "
                    f"| {entry['mean_gap_when_suboptimal']:.4f} |"
                )
    lines.append("")

    # ── Transition Points ──
    lines.append("## 3. Transition Point Analysis")
    lines.append("")
    lines.append("| Cancer | Divergence Onset | Aux Benefit % | Rank Benefit % |")
    lines.append("|--------|-----------------|---------------|----------------|")
    for cancer in CANCERS:
        if cancer not in transitions:
            continue
        t = transitions[cancer]
        onset = f"{t['mean_divergence_onset']:.1f}" if t["mean_divergence_onset"] is not None else "N/A"
        lines.append(
            f"| {CANCER_LABELS[cancer]} | ep {onset} "
            f"| {t['aux_benefit_fraction']:.1%} | {t['rank_benefit_fraction']:.1%} |"
        )
    lines.append("")

    # ── Cross-Fold Consistency ──
    lines.append("## 4. Cross-Fold Consistency")
    lines.append("")
    lines.append("Mean pairwise correlation of val_ci trajectories across folds.")
    lines.append("")
    # Show just static methods
    header = "| Method | " + " | ".join(CANCER_LABELS[c] for c in CANCERS) + " |"
    sep = "|--------|" + "|".join("------" for _ in CANCERS) + "|"
    lines.extend([header, sep])
    for method in STATIC_METHODS:
        cells = []
        for cancer in CANCERS:
            entry = consistency.get(cancer, {}).get(method)
            if entry:
                cells.append(f"{entry['mean_corr']:.3f}")
            else:
                cells.append("—")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.append("")

    # ── Feature Correlations Summary ──
    if correlations:
        lines.append("## 5. Feature-Outcome Correlations (Top 5)")
        lines.append("")
        lines.append("Top features correlated with delta_val_ci (partial r, controlling for current val_ci).")
        lines.append("")
        for cancer in CANCERS:
            if cancer not in correlations or not correlations[cancer]:
                continue
            lines.append(f"### {CANCER_LABELS[cancer]}")
            lines.append("")
            # Pick richest method
            best_method = max(correlations[cancer],
                             key=lambda m: len(correlations[cancer][m]),
                             default=None)
            if not best_method:
                continue
            feats = correlations[cancer][best_method]
            ranked = sorted(feats.items(), key=lambda x: abs(x[1]["partial_r"]), reverse=True)[:5]
            lines.append(f"Method: {best_method}")
            lines.append("")
            lines.append("| Feature | Partial r | Raw r | N obs |")
            lines.append("|---------|-----------|-------|-------|")
            for feat, info in ranked:
                lines.append(f"| {feat} | {info['partial_r']:.3f} | {info['raw_r']:.3f} | {info['n_obs']} |")
            lines.append("")

    # ── Go/No-Go Recommendation ──
    lines.append("## 6. Go/No-Go Recommendation")
    lines.append("")

    mean_headroom = np.mean(total_headroom_fold) if total_headroom_fold else 0
    mean_val_headroom = np.mean(total_headroom_val) if total_headroom_val else 0

    if mean_headroom > 0.01:
        lines.append(f"**GO**: Mean per-fold headroom is {mean_headroom:+.4f}, suggesting meaningful room "
                     "for per-fold method selection. The v4 probe mechanism is worth testing.")
    elif mean_headroom > 0.003:
        lines.append(f"**CONDITIONAL GO**: Mean per-fold headroom is {mean_headroom:+.4f} — marginal but "
                     "non-zero. v4 may match per-cancer-best if probes are reliable. "
                     "Success = automated per-cancer selection without per-cancer tuning.")
    else:
        lines.append(f"**NO-GO**: Mean per-fold headroom is {mean_headroom:+.4f} — too small for dynamic "
                     "switching to reliably improve. Per-cancer-best static is the practical ceiling.")
    lines.append("")

    if mean_val_headroom > 0:
        lines.append(f"Val-oracle headroom ({mean_val_headroom:+.4f}) suggests that picking the method with "
                     "best validation CI is a reasonable strategy — this is exactly what v4's probe does.")
    else:
        lines.append(f"Val-oracle headroom ({mean_val_headroom:+.4f}) is non-positive, meaning validation CI "
                     "does not reliably predict test CI advantage. Probe mechanism may not generalize.")
    lines.append("")

    # Quantify v3 failure modes
    v3_rates = []
    for cancer in CANCERS:
        for v3m in V3_METHODS:
            entry = v3_audit.get(cancer, {}).get(v3m)
            if entry:
                v3_rates.append(entry["optimality_rate"])
    if v3_rates:
        lines.append(f"v3 mode optimality: {np.mean(v3_rates):.1%} mean across cancers/variants. "
                     "v4's direct val-CI probing should improve on v3's loss-progress heuristic.")
    lines.append("")

    report_path = output_dir / "analysis_report.md"
    report_path.write_text("\n".join(lines))
    print(f"\nReport: {report_path}")
    return report_path


# ═══════════════════════════════════════════════════════════════════════════
# Section 9: Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RA-SAMO v4 trajectory analysis")
    parser.add_argument("--static-dir", type=Path,
                        default=REPO_ROOT / "results" / "samo_all_cancers_50ep")
    parser.add_argument("--v3-dir", type=Path,
                        default=REPO_ROOT / "results" / "ra_samo_v3_50ep")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "results" / "v4_analysis")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1A: Load data
    print("Loading data...")
    data, summaries = load_all_data(args.static_dir, args.v3_dir)
    if not data:
        print("No data found. Check paths.")
        return

    # 1B: Oracle analysis
    print("\nRunning oracle analysis...")
    oracle_results = oracle_analysis(data, summaries)

    # 1C: Feature-outcome correlations
    print("\nComputing feature-outcome correlations...")
    correlations = feature_outcome_correlations(data)

    # 1D: v3 mode audit
    print("\nAuditing v3 mode decisions...")
    v3_audit = v3_mode_audit(data, summaries)

    # 1E: Transition analysis
    print("\nAnalyzing transition points...")
    transitions = transition_analysis(data)

    # 1F: Cross-fold consistency
    print("\nComputing cross-fold consistency...")
    consistency = cross_fold_consistency(data)

    # Figures
    print("\nGenerating figures...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_oracle_headroom(oracle_results, fig_dir)
        plot_leader_changes(oracle_results, fig_dir)
        plot_feature_heatmap(correlations, fig_dir)
        plot_val_ci_trajectories(data, fig_dir)
        plot_cross_fold_stability(consistency, fig_dir)

    # 1G: Report
    print("\nGenerating report...")
    generate_report(oracle_results, correlations, v3_audit, transitions,
                    consistency, args.out)

    # Save raw results as JSON
    raw = {
        "oracle": oracle_results,
        "v3_audit": v3_audit,
        "transitions": {k: {kk: vv for kk, vv in v.items()
                            if not isinstance(vv, dict) or not any(isinstance(x, list) for x in vv.values())}
                        for k, v in transitions.items()},
        "consistency": consistency,
    }
    (args.out / "raw_analysis.json").write_text(json.dumps(raw, indent=2, default=str))

    print("\nDone.")


if __name__ == "__main__":
    main()
