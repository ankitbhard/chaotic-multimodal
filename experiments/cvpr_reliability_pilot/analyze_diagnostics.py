"""
Analyze Phase 2 training diagnostics.

Produces PHASE2_DIAGNOSTICS_REPORT.md with:
1. Gradient dynamics comparison (pcrf_aux vs learned_gating_aux)
2. Adam moment evolution
3. Update stability and oscillation
4. Effective learning rate analysis
5. Hessian curvature (BRCA only)
6. Epoch-level gradient noise and consistency
7. Moment adaptation speed
8. Cross-cancer comparison
"""

import os, json, csv, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

BASE = Path(__file__).resolve().parent.parent.parent
DIAG_DIR = BASE / "results" / "cvpr_reliability_pilot" / "phase2_diagnostics"

CANCERS = ["brca", "kirc", "luad"]
METHODS = ["learned_gating_aux", "pcrf_aux", "pcrf_aux_warmup"]
FOLDS = [0, 1, 2]

# Key parameter groups to focus on
KEY_GROUPS = ["image_encoder", "gene_encoder", "shared", "reliability_img", "reliability_gene"]
ENCODER_GROUPS = ["image_encoder", "gene_encoder"]


def read_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                try:
                    row[k] = float(v)
                except (ValueError, TypeError):
                    pass
            rows.append(row)
    return rows


def read_json(path):
    with open(path) as f:
        return json.load(f)


def aggregate_step_data(cancer, method, groups=None):
    """Load step data across folds, optionally filtered to specific groups."""
    all_rows = []
    for fold in FOLDS:
        path = DIAG_DIR / cancer / f"fold_{fold}" / f"{method}_step_diagnostics.csv"
        if not path.exists():
            continue
        rows = read_csv(path)
        for r in rows:
            r["fold"] = fold
            if groups is None or r["group"] in groups:
                all_rows.append(r)
    return all_rows


def aggregate_epoch_data(cancer, method, groups=None):
    """Load epoch data across folds."""
    all_rows = []
    for fold in FOLDS:
        path = DIAG_DIR / cancer / f"fold_{fold}" / f"{method}_epoch_diagnostics.csv"
        if not path.exists():
            continue
        rows = read_csv(path)
        for r in rows:
            r["fold"] = fold
            if groups is None or r["group"] in groups:
                all_rows.append(r)
    return all_rows


def aggregate_hessian_data(cancer, method):
    """Load hessian data across folds."""
    all_rows = []
    for fold in FOLDS:
        path = DIAG_DIR / cancer / f"fold_{fold}" / f"{method}_hessian_diagnostics.csv"
        if not path.exists():
            continue
        rows = read_csv(path)
        for r in rows:
            r["fold"] = fold
        all_rows.extend(rows)
    return all_rows


def epoch_means(rows, key, group=None):
    """Get per-epoch mean of a key across folds, optionally for one group."""
    by_epoch = defaultdict(list)
    for r in rows:
        if group and r["group"] != group:
            continue
        ep = int(r["epoch"])
        v = r.get(key)
        if v is not None and not np.isnan(v):
            by_epoch[ep].append(v)
    epochs = sorted(by_epoch.keys())
    return epochs, [np.mean(by_epoch[e]) for e in epochs]


def phase_means(rows, key, group, phases=None):
    """Get mean of key in early/mid/late training phases."""
    if phases is None:
        phases = [(1, 20, "early"), (21, 60, "mid"), (61, 100, "late")]
    results = {}
    for start, end, name in phases:
        vals = [r[key] for r in rows
                if r["group"] == group
                and start <= int(r.get("epoch", r.get("step", 0) // 100 + 1)) <= end
                and not np.isnan(r[key])]
        results[name] = np.mean(vals) if vals else float("nan")
    return results


def fmt(x, decimals=4):
    if isinstance(x, float) and np.isnan(x):
        return "—"
    if isinstance(x, float):
        if abs(x) < 0.001 and x != 0:
            return f"{x:.2e}"
        return f"{x:.{decimals}f}"
    return str(x)


def main():
    report = []
    W = report.append

    W("# Phase 2 Training Diagnostics Report")
    W("")
    W(f"**Date:** 2026-10-19")
    W(f"**Experiment:** PCRF Phase 2 (3 methods × 3 cancers × 3 folds)")
    W(f"**Diagnostics:** 18 metric categories across step/epoch/Hessian granularity")
    W("")
    W("---")
    W("")

    # ── 1. Summary statistics ────────────────────────────────────────────────
    W("## 1. Training Outcome Summary")
    W("")
    W("| Cancer | Method | Test C-index | Val C-index | Gen Gap |")
    W("|--------|--------|:------------:|:-----------:|:-------:|")

    for cancer in CANCERS:
        for method in METHODS:
            test_cis = []
            val_cis = []
            for fold in FOLDS:
                jpath = DIAG_DIR / cancer / f"fold_{fold}" / f"{method}.json"
                if jpath.exists():
                    j = read_json(jpath)
                    test_cis.append(j.get("test_cindex", float("nan")))
                    val_cis.append(j.get("best_val_cindex", float("nan")))
            if test_cis:
                tc = np.mean(test_cis)
                vc = np.mean(val_cis)
                gap = vc - tc
                W(f"| {cancer.upper()} | {method} | {tc:.4f}±{np.std(test_cis):.3f} | {vc:.4f} | {gap:.3f} |")

    W("")
    W("---")
    W("")

    # ── 2. Gradient dynamics ─────────────────────────────────────────────────
    W("## 2. Gradient Dynamics")
    W("")
    W("### 2.1 Gradient Norm Evolution (Encoders)")
    W("")
    W("Mean gradient norm across training phases (averaged over folds).")
    W("")
    W("| Cancer | Method | Group | Early (1-20) | Mid (21-60) | Late (61-100) | Trend |")
    W("|--------|--------|-------|:------------:|:-----------:|:-------------:|:-----:|")

    for cancer in CANCERS:
        for method in ["learned_gating_aux", "pcrf_aux"]:
            step_rows = aggregate_step_data(cancer, method, ENCODER_GROUPS)
            if not step_rows:
                continue
            for group in ENCODER_GROUPS:
                # Compute per-epoch means first
                by_epoch = defaultdict(list)
                for r in step_rows:
                    if r["group"] == group:
                        by_epoch[int(r["epoch"])].append(r["grad_norm"])
                epoch_norms = {e: np.mean(v) for e, v in by_epoch.items()}
                early = np.mean([v for e, v in epoch_norms.items() if 1 <= e <= 20]) if epoch_norms else float("nan")
                mid = np.mean([v for e, v in epoch_norms.items() if 21 <= e <= 60]) if epoch_norms else float("nan")
                late = np.mean([v for e, v in epoch_norms.items() if 61 <= e <= 100]) if epoch_norms else float("nan")
                if not np.isnan(early) and not np.isnan(late) and early > 0:
                    ratio = late / early
                    trend = "↓" if ratio < 0.8 else ("↑" if ratio > 1.2 else "→")
                    trend += f" ({ratio:.2f}x)"
                else:
                    trend = "—"
                short_group = group.replace("_encoder", "")
                W(f"| {cancer.upper()} | {method} | {short_group} | {fmt(early)} | {fmt(mid)} | {fmt(late)} | {trend} |")

    W("")

    # ── 2.2 Gradient-to-weight ratio ────────────────────────────────────────
    W("### 2.2 Gradient-to-Weight Ratio (Encoder Health)")
    W("")
    W("Healthy range: 1e-4 to 1e-2. Values too small → vanishing; too large → unstable.")
    W("")
    W("| Cancer | Method | Group | Epoch 1 | Epoch 50 | Epoch 100 | Status |")
    W("|--------|--------|-------|:-------:|:--------:|:---------:|:------:|")

    for cancer in CANCERS:
        for method in ["learned_gating_aux", "pcrf_aux"]:
            step_rows = aggregate_step_data(cancer, method, ENCODER_GROUPS)
            if not step_rows:
                continue
            for group in ENCODER_GROUPS:
                by_epoch = defaultdict(list)
                for r in step_rows:
                    if r["group"] == group:
                        by_epoch[int(r["epoch"])].append(r["grad_weight_ratio"])
                e1 = np.mean(by_epoch.get(1, [float("nan")])) if 1 in by_epoch else float("nan")
                e50 = np.mean(by_epoch.get(50, [float("nan")])) if 50 in by_epoch else float("nan")
                e100 = np.mean(by_epoch.get(100, [float("nan")])) if 100 in by_epoch else float("nan")
                # Status check
                vals = [v for v in [e1, e50, e100] if not np.isnan(v)]
                if vals:
                    avg = np.mean(vals)
                    status = "OK" if 1e-4 < avg < 1e-1 else ("LOW" if avg <= 1e-4 else "HIGH")
                else:
                    status = "—"
                short_group = group.replace("_encoder", "")
                W(f"| {cancer.upper()} | {method} | {short_group} | {fmt(e1)} | {fmt(e50)} | {fmt(e100)} | {status} |")

    W("")
    W("---")
    W("")

    # ── 3. Adam Moment Analysis ──────────────────────────────────────────────
    W("## 3. Adam Moment Analysis")
    W("")
    W("### 3.1 First Moment (m̂) Norm — Gradient Signal Strength")
    W("")
    W("| Cancer | Method | Group | Early | Mid | Late |")
    W("|--------|--------|-------|:-----:|:---:|:----:|")

    for cancer in CANCERS:
        for method in ["learned_gating_aux", "pcrf_aux"]:
            step_rows = aggregate_step_data(cancer, method, ENCODER_GROUPS)
            if not step_rows:
                continue
            for group in ENCODER_GROUPS:
                by_epoch = defaultdict(list)
                for r in step_rows:
                    if r["group"] == group:
                        by_epoch[int(r["epoch"])].append(r["m_norm"])
                epoch_vals = {e: np.mean(v) for e, v in by_epoch.items()}
                early = np.mean([v for e, v in epoch_vals.items() if 1 <= e <= 20]) if epoch_vals else float("nan")
                mid = np.mean([v for e, v in epoch_vals.items() if 21 <= e <= 60]) if epoch_vals else float("nan")
                late = np.mean([v for e, v in epoch_vals.items() if 61 <= e <= 100]) if epoch_vals else float("nan")
                short_group = group.replace("_encoder", "")
                W(f"| {cancer.upper()} | {method} | {short_group} | {fmt(early)} | {fmt(mid)} | {fmt(late)} |")

    W("")

    W("### 3.2 Second Moment (v̂) Mean — Gradient Magnitude Scale")
    W("")
    W("| Cancer | Method | Group | Early | Mid | Late |")
    W("|--------|--------|-------|:-----:|:---:|:----:|")

    for cancer in CANCERS:
        for method in ["learned_gating_aux", "pcrf_aux"]:
            step_rows = aggregate_step_data(cancer, method, ENCODER_GROUPS)
            if not step_rows:
                continue
            for group in ENCODER_GROUPS:
                by_epoch = defaultdict(list)
                for r in step_rows:
                    if r["group"] == group:
                        by_epoch[int(r["epoch"])].append(r["v_mean"])
                epoch_vals = {e: np.mean(v) for e, v in by_epoch.items()}
                early = np.mean([v for e, v in epoch_vals.items() if 1 <= e <= 20]) if epoch_vals else float("nan")
                mid = np.mean([v for e, v in epoch_vals.items() if 21 <= e <= 60]) if epoch_vals else float("nan")
                late = np.mean([v for e, v in epoch_vals.items() if 61 <= e <= 100]) if epoch_vals else float("nan")
                short_group = group.replace("_encoder", "")
                W(f"| {cancer.upper()} | {method} | {short_group} | {fmt(early)} | {fmt(mid)} | {fmt(late)} |")

    W("")
    W("---")
    W("")

    # ── 4. Effective Learning Rate ───────────────────────────────────────────
    W("## 4. Effective Learning Rate Distribution")
    W("")
    W("Adam's per-parameter effective LR = nominal_lr / (√v̂ + ε). Spread (p90/p10) shows adaptivity.")
    W("")
    W("| Cancer | Method | Group | p10 (ep 10) | p50 (ep 10) | p90 (ep 10) | Spread | p50 (ep 50) |")
    W("|--------|--------|-------|:-----------:|:-----------:|:-----------:|:------:|:-----------:|")

    for cancer in CANCERS:
        for method in ["learned_gating_aux", "pcrf_aux"]:
            step_rows = aggregate_step_data(cancer, method, ENCODER_GROUPS)
            if not step_rows:
                continue
            for group in ENCODER_GROUPS:
                ep10 = [r for r in step_rows if r["group"] == group and int(r["epoch"]) == 10]
                ep50 = [r for r in step_rows if r["group"] == group and int(r["epoch"]) == 50]
                if ep10:
                    p10 = np.mean([r["effective_lr_p10"] for r in ep10])
                    p50 = np.mean([r["effective_lr_p50"] for r in ep10])
                    p90 = np.mean([r["effective_lr_p90"] for r in ep10])
                    spread = p90 / (p10 + 1e-12)
                else:
                    p10 = p50 = p90 = spread = float("nan")
                p50_late = np.mean([r["effective_lr_p50"] for r in ep50]) if ep50 else float("nan")
                short_group = group.replace("_encoder", "")
                W(f"| {cancer.upper()} | {method} | {short_group} | {fmt(p10)} | {fmt(p50)} | {fmt(p90)} | {fmt(spread, 1)}x | {fmt(p50_late)} |")

    W("")
    W("---")
    W("")

    # ── 5. Update Dynamics ───────────────────────────────────────────────────
    W("## 5. Update Dynamics")
    W("")
    W("### 5.1 Update-to-Weight Ratio (Step Size)")
    W("")
    W("Healthy range: 1e-4 to 1e-2. Tracks how much weights change per step.")
    W("")
    W("| Cancer | Method | Group | Early | Mid | Late |")
    W("|--------|--------|-------|:-----:|:---:|:----:|")

    for cancer in CANCERS:
        for method in ["learned_gating_aux", "pcrf_aux"]:
            step_rows = aggregate_step_data(cancer, method, ENCODER_GROUPS)
            if not step_rows:
                continue
            for group in ENCODER_GROUPS:
                by_epoch = defaultdict(list)
                for r in step_rows:
                    if r["group"] == group:
                        by_epoch[int(r["epoch"])].append(r["update_weight_ratio"])
                epoch_vals = {e: np.mean(v) for e, v in by_epoch.items()}
                early = np.mean([v for e, v in epoch_vals.items() if 1 <= e <= 20]) if epoch_vals else float("nan")
                mid = np.mean([v for e, v in epoch_vals.items() if 21 <= e <= 60]) if epoch_vals else float("nan")
                late = np.mean([v for e, v in epoch_vals.items() if 61 <= e <= 100]) if epoch_vals else float("nan")
                short_group = group.replace("_encoder", "")
                W(f"| {cancer.upper()} | {method} | {short_group} | {fmt(early)} | {fmt(mid)} | {fmt(late)} |")

    W("")

    # ── 5.2 Gradient-Update Alignment ────────────────────────────────────────
    W("### 5.2 Gradient–Update Alignment cos(g, Δθ)")
    W("")
    W("Measures how well Adam's update follows the raw gradient. cos ≈ −1 means")
    W("Adam descends along gradient; cos ≈ 0 means momentum/adaptivity dominate.")
    W("")
    W("| Cancer | Method | Group | Early | Mid | Late |")
    W("|--------|--------|-------|:-----:|:---:|:----:|")

    for cancer in CANCERS:
        for method in ["learned_gating_aux", "pcrf_aux"]:
            step_rows = aggregate_step_data(cancer, method, ENCODER_GROUPS)
            if not step_rows:
                continue
            for group in ENCODER_GROUPS:
                by_epoch = defaultdict(list)
                for r in step_rows:
                    if r["group"] == group and not np.isnan(r["grad_update_cos"]):
                        by_epoch[int(r["epoch"])].append(r["grad_update_cos"])
                epoch_vals = {e: np.mean(v) for e, v in by_epoch.items()}
                early = np.mean([v for e, v in epoch_vals.items() if 1 <= e <= 20]) if epoch_vals else float("nan")
                mid = np.mean([v for e, v in epoch_vals.items() if 21 <= e <= 60]) if epoch_vals else float("nan")
                late = np.mean([v for e, v in epoch_vals.items() if 61 <= e <= 100]) if epoch_vals else float("nan")
                short_group = group.replace("_encoder", "")
                W(f"| {cancer.upper()} | {method} | {short_group} | {fmt(early)} | {fmt(mid)} | {fmt(late)} |")

    W("")

    # ── 5.3 Update Oscillation ───────────────────────────────────────────────
    W("### 5.3 Update Oscillation cos(Δθ_t, Δθ_{t-1})")
    W("")
    W("Positive → consistent direction. Negative → oscillation (zig-zagging).")
    W("")
    W("| Cancer | Method | Group | Early | Mid | Late |")
    W("|--------|--------|-------|:-----:|:---:|:----:|")

    for cancer in CANCERS:
        for method in ["learned_gating_aux", "pcrf_aux"]:
            step_rows = aggregate_step_data(cancer, method, ENCODER_GROUPS)
            if not step_rows:
                continue
            for group in ENCODER_GROUPS:
                by_epoch = defaultdict(list)
                for r in step_rows:
                    if r["group"] == group and not np.isnan(r["update_prev_cos"]):
                        by_epoch[int(r["epoch"])].append(r["update_prev_cos"])
                epoch_vals = {e: np.mean(v) for e, v in by_epoch.items()}
                early = np.mean([v for e, v in epoch_vals.items() if 1 <= e <= 20]) if epoch_vals else float("nan")
                mid = np.mean([v for e, v in epoch_vals.items() if 21 <= e <= 60]) if epoch_vals else float("nan")
                late = np.mean([v for e, v in epoch_vals.items() if 61 <= e <= 100]) if epoch_vals else float("nan")
                short_group = group.replace("_encoder", "")
                W(f"| {cancer.upper()} | {method} | {short_group} | {fmt(early)} | {fmt(mid)} | {fmt(late)} |")

    W("")
    W("---")
    W("")

    # ── 6. Epoch-Level Gradient Noise ────────────────────────────────────────
    W("## 6. Gradient Noise Analysis")
    W("")
    W("### 6.1 Gradient Noise Ratio (σ/μ of batch gradient norms)")
    W("")
    W("Higher → noisier gradients → harder optimization.")
    W("")
    W("| Cancer | Method | Group | Early | Mid | Late |")
    W("|--------|--------|-------|:-----:|:---:|:----:|")

    for cancer in CANCERS:
        for method in ["learned_gating_aux", "pcrf_aux"]:
            epoch_rows = aggregate_epoch_data(cancer, method, ENCODER_GROUPS)
            if not epoch_rows:
                continue
            for group in ENCODER_GROUPS:
                by_epoch = {int(r["epoch"]): r["grad_noise_ratio"]
                            for r in epoch_rows if r["group"] == group
                            and not np.isnan(r.get("grad_noise_ratio", float("nan")))}
                early = np.mean([v for e, v in by_epoch.items() if 1 <= e <= 20]) if by_epoch else float("nan")
                mid = np.mean([v for e, v in by_epoch.items() if 21 <= e <= 60]) if by_epoch else float("nan")
                late = np.mean([v for e, v in by_epoch.items() if 61 <= e <= 100]) if by_epoch else float("nan")
                short_group = group.replace("_encoder", "")
                W(f"| {cancer.upper()} | {method} | {short_group} | {fmt(early)} | {fmt(mid)} | {fmt(late)} |")

    W("")

    W("### 6.2 Consecutive Gradient Cosine (batch-to-batch consistency)")
    W("")
    W("Higher → more consistent gradients between mini-batches.")
    W("")
    W("| Cancer | Method | Group | Early | Mid | Late |")
    W("|--------|--------|-------|:-----:|:---:|:----:|")

    for cancer in CANCERS:
        for method in ["learned_gating_aux", "pcrf_aux"]:
            epoch_rows = aggregate_epoch_data(cancer, method, ENCODER_GROUPS)
            if not epoch_rows:
                continue
            for group in ENCODER_GROUPS:
                by_epoch = {int(r["epoch"]): r["consecutive_grad_cos_mean"]
                            for r in epoch_rows if r["group"] == group
                            and not np.isnan(r.get("consecutive_grad_cos_mean", float("nan")))}
                early = np.mean([v for e, v in by_epoch.items() if 1 <= e <= 20]) if by_epoch else float("nan")
                mid = np.mean([v for e, v in by_epoch.items() if 21 <= e <= 60]) if by_epoch else float("nan")
                late = np.mean([v for e, v in by_epoch.items() if 61 <= e <= 100]) if by_epoch else float("nan")
                short_group = group.replace("_encoder", "")
                W(f"| {cancer.upper()} | {method} | {short_group} | {fmt(early)} | {fmt(mid)} | {fmt(late)} |")

    W("")
    W("---")
    W("")

    # ── 7. Moment Adaptation Speed ───────────────────────────────────────────
    W("## 7. Moment Adaptation Speed")
    W("")
    W("Epoch-over-epoch change in Adam moments. Faster adaptation → responding to distribution shifts.")
    W("")
    W("| Cancer | Method | Group | m_speed (early) | m_speed (late) | v_speed (early) | v_speed (late) |")
    W("|--------|--------|-------|:---------------:|:--------------:|:---------------:|:--------------:|")

    for cancer in CANCERS:
        for method in ["learned_gating_aux", "pcrf_aux"]:
            epoch_rows = aggregate_epoch_data(cancer, method, ENCODER_GROUPS)
            if not epoch_rows:
                continue
            for group in ENCODER_GROUPS:
                rows_g = [r for r in epoch_rows if r["group"] == group]
                m_early = np.mean([r["moment_m_speed"] for r in rows_g
                                   if 1 <= int(r["epoch"]) <= 20
                                   and not np.isnan(r.get("moment_m_speed", float("nan")))]) if rows_g else float("nan")
                m_late = np.mean([r["moment_m_speed"] for r in rows_g
                                  if 61 <= int(r["epoch"]) <= 100
                                  and not np.isnan(r.get("moment_m_speed", float("nan")))]) if rows_g else float("nan")
                v_early = np.mean([r["moment_v_speed"] for r in rows_g
                                   if 1 <= int(r["epoch"]) <= 20
                                   and not np.isnan(r.get("moment_v_speed", float("nan")))]) if rows_g else float("nan")
                v_late = np.mean([r["moment_v_speed"] for r in rows_g
                                  if 61 <= int(r["epoch"]) <= 100
                                  and not np.isnan(r.get("moment_v_speed", float("nan")))]) if rows_g else float("nan")
                short_group = group.replace("_encoder", "")
                W(f"| {cancer.upper()} | {method} | {short_group} | {fmt(m_early)} | {fmt(m_late)} | {fmt(v_early)} | {fmt(v_late)} |")

    W("")
    W("---")
    W("")

    # ── 8. Hessian Curvature (BRCA only) ────────────────────────────────────
    W("## 8. Hessian Curvature (BRCA)")
    W("")

    has_hessian = False
    for method in METHODS:
        hess_rows = aggregate_hessian_data("brca", method)
        if hess_rows:
            has_hessian = True
            break

    if has_hessian:
        W("| Method | Epoch | Top Eigenvalue | Trace | Update Curvature |")
        W("|--------|:-----:|:--------------:|:-----:|:----------------:|")
        for method in METHODS:
            hess_rows = aggregate_hessian_data("brca", method)
            for r in sorted(hess_rows, key=lambda x: int(x["epoch"])):
                eig = r.get("hessian_top_eigenvalue", float("nan"))
                tr = r.get("hessian_trace", float("nan"))
                curv = r.get("update_direction_curvature", float("nan"))
                W(f"| {method} | {int(r['epoch'])} | {fmt(eig)} | {fmt(tr)} | {fmt(curv)} |")
    else:
        W("No Hessian data available for BRCA.")

    W("")
    W("---")
    W("")

    # ── 9. Reliability Module Diagnostics ────────────────────────────────────
    W("## 9. Reliability Module Diagnostics")
    W("")
    W("Gradient norms and update ratios for the reliability networks (only in pcrf_aux).")
    W("")
    W("| Cancer | Group | Grad Norm (early) | Grad Norm (late) | Update/Weight (early) | Update/Weight (late) |")
    W("|--------|-------|:-----------------:|:----------------:|:---------------------:|:--------------------:|")

    rel_groups = ["reliability_img", "reliability_gene"]
    for cancer in CANCERS:
        step_rows = aggregate_step_data(cancer, "pcrf_aux", rel_groups)
        if not step_rows:
            continue
        for group in rel_groups:
            by_epoch_gn = defaultdict(list)
            by_epoch_uwr = defaultdict(list)
            for r in step_rows:
                if r["group"] == group:
                    by_epoch_gn[int(r["epoch"])].append(r["grad_norm"])
                    by_epoch_uwr[int(r["epoch"])].append(r["update_weight_ratio"])
            gn_early = np.mean([np.mean(v) for e, v in by_epoch_gn.items() if 1 <= e <= 20]) if by_epoch_gn else float("nan")
            gn_late = np.mean([np.mean(v) for e, v in by_epoch_gn.items() if 61 <= e <= 100]) if by_epoch_gn else float("nan")
            uwr_early = np.mean([np.mean(v) for e, v in by_epoch_uwr.items() if 1 <= e <= 20]) if by_epoch_uwr else float("nan")
            uwr_late = np.mean([np.mean(v) for e, v in by_epoch_uwr.items() if 61 <= e <= 100]) if by_epoch_uwr else float("nan")
            short = group.replace("reliability_", "rel_")
            W(f"| {cancer.upper()} | {short} | {fmt(gn_early)} | {fmt(gn_late)} | {fmt(uwr_early)} | {fmt(uwr_late)} |")

    W("")
    W("---")
    W("")

    # ── 10. Key Findings ─────────────────────────────────────────────────────
    W("## 10. Cross-Method Comparison Summary")
    W("")

    # Compute aggregate differences
    for cancer in CANCERS:
        W(f"### {cancer.upper()}")
        W("")

        for group in ENCODER_GROUPS:
            short_group = group.replace("_encoder", "")
            lg_steps = aggregate_step_data(cancer, "learned_gating_aux", [group])
            pcrf_steps = aggregate_step_data(cancer, "pcrf_aux", [group])

            if not lg_steps or not pcrf_steps:
                continue

            # Compare key metrics at epoch 50
            lg_50 = [r for r in lg_steps if int(r["epoch"]) == 50]
            pcrf_50 = [r for r in pcrf_steps if int(r["epoch"]) == 50]

            if lg_50 and pcrf_50:
                metrics = {
                    "grad_norm": ("Gradient norm", ""),
                    "update_weight_ratio": ("Update/weight ratio", ""),
                    "grad_update_cos": ("Grad-update alignment", ""),
                }
                W(f"**{short_group} encoder at epoch 50:**")
                W("")
                for key, (label, unit) in metrics.items():
                    lg_val = np.mean([r[key] for r in lg_50 if not np.isnan(r[key])])
                    pcrf_val = np.mean([r[key] for r in pcrf_50 if not np.isnan(r[key])])
                    diff = pcrf_val - lg_val
                    W(f"- {label}: gating={fmt(lg_val)}, pcrf={fmt(pcrf_val)} (Δ={fmt(diff)})")
                W("")

        # Epoch-level comparison
        lg_epoch = aggregate_epoch_data(cancer, "learned_gating_aux", ENCODER_GROUPS)
        pcrf_epoch = aggregate_epoch_data(cancer, "pcrf_aux", ENCODER_GROUPS)

        if lg_epoch and pcrf_epoch:
            W("**Gradient noise (epoch-level):**")
            W("")
            for group in ENCODER_GROUPS:
                short_group = group.replace("_encoder", "")
                lg_noise = [r["grad_noise_ratio"] for r in lg_epoch
                            if r["group"] == group and not np.isnan(r.get("grad_noise_ratio", float("nan")))]
                pcrf_noise = [r["grad_noise_ratio"] for r in pcrf_epoch
                              if r["group"] == group and not np.isnan(r.get("grad_noise_ratio", float("nan")))]
                if lg_noise and pcrf_noise:
                    W(f"- {short_group}: gating noise={fmt(np.mean(lg_noise))}, pcrf noise={fmt(np.mean(pcrf_noise))}")
            W("")

        W("---")
        W("")

    # ── 11. Conclusions ──────────────────────────────────────────────────────
    W("## 11. Diagnostic Conclusions")
    W("")
    W("### Data Collected")
    W("")

    # Count rows
    total_step = 0
    total_epoch = 0
    total_hessian = 0
    for cancer in CANCERS:
        for method in METHODS:
            for fold in FOLDS:
                sp = DIAG_DIR / cancer / f"fold_{fold}" / f"{method}_step_diagnostics.csv"
                ep = DIAG_DIR / cancer / f"fold_{fold}" / f"{method}_epoch_diagnostics.csv"
                hp = DIAG_DIR / cancer / f"fold_{fold}" / f"{method}_hessian_diagnostics.csv"
                if sp.exists():
                    with open(sp) as f:
                        total_step += sum(1 for _ in f) - 1
                if ep.exists():
                    with open(ep) as f:
                        total_epoch += sum(1 for _ in f) - 1
                if hp.exists():
                    with open(hp) as f:
                        total_hessian += sum(1 for _ in f) - 1

    W(f"- **Step-level diagnostics:** {total_step:,} rows (15 metrics per row)")
    W(f"- **Epoch-level diagnostics:** {total_epoch:,} rows (11 metrics per row)")
    W(f"- **Hessian diagnostics:** {total_hessian:,} rows (3 metrics per row)")
    W(f"- **Total data points:** ~{(total_step * 15 + total_epoch * 11 + total_hessian * 3):,}")
    W("")

    # Write report
    out_path = DIAG_DIR / "PHASE2_DIAGNOSTICS_REPORT.md"
    out_path.write_text("\n".join(report))
    print(f"Report written to: {out_path}")
    print(f"  Step rows: {total_step:,}")
    print(f"  Epoch rows: {total_epoch:,}")
    print(f"  Hessian rows: {total_hessian:,}")


if __name__ == "__main__":
    main()
