"""Noninvasive information acquisition diagnostics for multimodal survival models.

Computes per-epoch, per-encoder gradient sensitivity, approximate Fisher trace,
gradient norms, alignment, variability, and unimodal validation C-index.

All diagnostics are computed on a fixed, deterministic, training-only probe subset.
Model and optimizer state are preserved exactly: eval mode is set and restored,
RNG state is saved and restored, .grad fields are cleared after measurement.

Terminology (to avoid misattribution):
- "squared_cox_grad_norm": L2-norm-squared of batch Cox gradient w.r.t. encoder
  parameters. This is NOT the empirical Fisher trace because mini-batch Cox
  couples individuals via risk sets.
- "approx_fisher_trace": Sum of squared per-event score-gradient norms, normalized
  by number of events. Each event's score is d/d(theta) [risk_i - log sum_{j in R_i} exp(risk_j)].
  This approximates the diagonal Fisher under Breslow's tied-event handling,
  treating each mini-batch as an independent risk set. It is a biased estimator
  when batches don't contain the full risk set.
"""
import math
import hashlib
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from train_survival import concordance_index
from experiments.samo.objectives import cox_efron_free, ranking_loss, comparable_pairs


# ---------------------------------------------------------------------------
# Probe subset construction
# ---------------------------------------------------------------------------

def build_probe_subset(dataset, n_probe: int = 64, seed: int = 42) -> Tuple[List[int], dict]:
    """Build a fixed, deterministic, training-only probe subset.

    Stratified by event indicator. Returns (indices, metadata).
    The dataset must be the TRAINING split only.
    """
    rng = np.random.RandomState(seed)
    events, censored = [], []
    for i in range(len(dataset)):
        _, labels = dataset[i]
        if int(labels["vital_status"]) == 1:
            events.append(i)
        else:
            censored.append(i)

    n_events = max(1, int(round(n_probe * len(events) / max(len(dataset), 1))))
    n_censored = n_probe - n_events

    rng.shuffle(events)
    rng.shuffle(censored)
    selected = sorted(events[:n_events] + censored[:min(n_censored, len(censored))])

    # If we don't have enough censored, take more events
    if len(selected) < n_probe:
        remaining = [i for i in events[n_events:] if i not in selected]
        selected = sorted(selected + remaining[:n_probe - len(selected)])

    # Compute patient ID hashes for reproducibility verification
    patient_hashes = []
    event_count = 0
    for idx in selected:
        _, labels = dataset[idx]
        # Hash the index and survival time for deterministic ID
        h = hashlib.md5(f"{idx}_{float(labels['os_months']):.4f}".encode()).hexdigest()[:8]
        patient_hashes.append(h)
        if int(labels["vital_status"]) == 1:
            event_count += 1

    meta = {
        "n_probe": len(selected),
        "n_events": event_count,
        "n_censored": len(selected) - event_count,
        "patient_hashes": patient_hashes,
        "seed": seed,
        "total_train_size": len(dataset),
    }
    return selected, meta


# ---------------------------------------------------------------------------
# Per-event Fisher score computation
# ---------------------------------------------------------------------------

def _per_event_scores(risk, time, event, params):
    """Compute per-event Cox score gradients (Breslow approximation).

    For each observed event i, the individual log-likelihood contribution is:
        ell_i = risk_i - log(sum_{j in R_i} exp(risk_j))
    where R_i = {j : time_j >= time_i} is the risk set at time_i.

    The score for event i is grad_theta(ell_i).

    Under Breslow's approximation, tied events at the same time share the
    same risk set denominator.

    This operates on a MINI-BATCH, so R_i is the mini-batch risk set.
    This is a biased approximation of the full-data Fisher.

    Returns: list of per-event gradient vectors (flattened), one per event.
    """
    risk = risk.reshape(-1)
    time = time.reshape(-1)
    event = event.reshape(-1).bool()

    event_indices = torch.where(event)[0]
    if len(event_indices) == 0:
        return []

    scores = []
    for i in event_indices:
        # Risk set: all patients with time >= time[i]
        risk_set = time >= time[i]
        log_denom = torch.logsumexp(risk[risk_set], dim=0)
        ell_i = risk[i] - log_denom

        grads = torch.autograd.grad(ell_i, params, retain_graph=True,
                                     allow_unused=True)
        flat = torch.cat([
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for p, g in zip(params, grads)
        ])
        scores.append(flat.detach())

    return scores


def approx_fisher_trace(scores: List[torch.Tensor]) -> float:
    """Compute approximate Fisher trace from per-event score gradients.

    Fisher trace = (1/n_events) * sum_i ||score_i||^2

    This is the trace of the empirical Fisher information matrix under
    the assumption that each event contributes independently. For Cox
    models with coupled risk sets, this is an approximation.
    """
    if not scores:
        return 0.0
    sq_norms = [float(s.dot(s).item()) for s in scores]
    return float(np.mean(sq_norms))


def per_layer_fisher(scores: List[torch.Tensor], param_shapes: List[Tuple[int, ...]]) -> Dict[str, float]:
    """Break down Fisher trace by parameter layer.

    Returns dict mapping layer index to its Fisher trace contribution,
    normalized by parameter count for cross-encoder comparison.
    """
    if not scores:
        return {}
    offsets = [0]
    for shape in param_shapes:
        offsets.append(offsets[-1] + int(np.prod(shape)))

    result = {}
    for layer_idx, (start, end, shape) in enumerate(
            zip(offsets[:-1], offsets[1:], param_shapes)):
        n_params = int(np.prod(shape))
        layer_sq_norms = [float(s[start:end].dot(s[start:end]).item()) for s in scores]
        result[f"layer_{layer_idx}_fisher"] = float(np.mean(layer_sq_norms))
        result[f"layer_{layer_idx}_fisher_normalized"] = float(np.mean(layer_sq_norms)) / max(n_params, 1)
        result[f"layer_{layer_idx}_n_params"] = n_params
    return result


# ---------------------------------------------------------------------------
# Gradient diagnostics
# ---------------------------------------------------------------------------

@dataclass
class EncoderDiagnostics:
    """Per-encoder, per-epoch diagnostic measurements."""
    # Gradient norms
    cox_grad_norm: float = 0.0           # ||grad_Cox||_2
    squared_cox_grad_norm: float = 0.0   # ||grad_Cox||_2^2

    # Approximate Fisher trace (risk-set-aware, per-event)
    fisher_trace: float = 0.0
    fisher_trace_normalized: float = 0.0  # divided by parameter count
    n_events_fisher: int = 0

    # Gradient sensitivity (how much loss changes with small parameter perturbation)
    grad_sensitivity: float = 0.0  # mean |grad| per parameter

    # Per-layer Fisher breakdown
    layer_fisher: Dict[str, float] = field(default_factory=dict)


@dataclass
class AlignmentDiagnostics:
    """Cross-objective gradient alignment and variability."""
    # Cosine similarities
    cosine_cox_aux: float = 0.0      # cos(grad_cox, grad_aux)
    cosine_cox_rank: float = 0.0     # cos(grad_cox, grad_rank)
    cosine_aux_rank: float = 0.0     # cos(grad_aux, grad_rank)

    # Gradient variability across probe batches
    cox_grad_variability: float = 0.0   # std of batch grad norms / mean
    aux_grad_variability: float = 0.0
    rank_grad_variability: float = 0.0


@dataclass
class EpochDiagnostics:
    """Complete diagnostics for one epoch."""
    epoch: int = 0
    cancer: str = ""
    method: str = ""
    fold: int = 0

    # Per-encoder diagnostics
    image_encoder: EncoderDiagnostics = field(default_factory=EncoderDiagnostics)
    gene_encoder: EncoderDiagnostics = field(default_factory=EncoderDiagnostics)

    # Alignment diagnostics (computed over both encoders jointly)
    alignment: AlignmentDiagnostics = field(default_factory=AlignmentDiagnostics)

    # Validation C-indices
    val_ci_fused: float = 0.0
    val_ci_image_unimodal: float = 0.0
    val_ci_gene_unimodal: float = 0.0

    # Training statistics
    train_fusion_loss: float = 0.0
    train_img_loss: float = 0.0
    train_gene_loss: float = 0.0
    train_rank_loss: float = 0.0

    # Survival statistics
    n_events_train: int = 0
    n_comparable_pairs: int = 0

    # Resource usage
    wall_seconds: float = 0.0
    gpu_memory_mb: float = 0.0


# ---------------------------------------------------------------------------
# Core diagnostic computation
# ---------------------------------------------------------------------------

def _flat_grads(loss, params):
    """Compute gradients and return as flattened vector."""
    if not loss.requires_grad:
        return torch.zeros(sum(p.numel() for p in params), device=params[0].device)
    raw = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([
        (g if g is not None else torch.zeros_like(p)).reshape(-1)
        for p, g in zip(params, raw)
    ])


def _cosine_sim(a, b, eps=1e-12):
    """Cosine similarity between two flat gradient vectors."""
    na = a.norm()
    nb = b.norm()
    if na.item() < eps or nb.item() < eps:
        return 0.0
    return float((a.dot(b) / (na * nb + eps)).item())


def compute_epoch_diagnostics(
    model: nn.Module,
    probe_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    method: str,
    epoch: int,
    cancer: str,
    fold: int,
    temperature: float = 1.0,
    unimodal_probes: Optional[dict] = None,
) -> EpochDiagnostics:
    """Compute all diagnostics for one epoch. Does NOT modify model or optimizer.

    Protocol:
    1. Save model training mode and .grad fields
    2. Set eval mode (for BatchNorm/Dropout consistency)
    3. Enable gradients on probe forward passes
    4. Compute all measurements
    5. Restore model to exact prior state

    Args:
        model: The model (must have image_encoder and gene_encoder)
        probe_loader: DataLoader over the fixed training-only probe subset
        val_loader: Validation set DataLoader
        device: torch device
        method: Training method name
        epoch: Current epoch number
        cancer: Cancer type
        fold: Fold number
        temperature: Ranking loss temperature
        unimodal_probes: Optional dict with fitted unimodal linear probes
    """
    diag = EpochDiagnostics(epoch=epoch, cancer=cancer, method=method, fold=fold)

    # --- Save state ---
    was_training = model.training
    saved_grads = {}
    for name, p in model.named_parameters():
        saved_grads[name] = p.grad.clone() if p.grad is not None else None

    # Save RNG state
    rng_state = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    np_rng = np.random.get_state()

    try:
        model.eval()  # Deterministic BN/Dropout for consistent measurements

        # Collect encoder parameters
        img_params = [p for p in model.image_encoder.parameters() if p.requires_grad]
        gene_params = [p for p in model.gene_encoder.parameters() if p.requires_grad]
        img_param_shapes = [p.shape for p in img_params]
        gene_param_shapes = [p.shape for p in gene_params]
        n_img_params = sum(p.numel() for p in img_params)
        n_gene_params = sum(p.numel() for p in gene_params)

        # Per-batch gradient collection
        img_cox_norms = []
        gene_cox_norms = []
        img_aux_norms = []
        gene_aux_norms = []
        img_rank_norms = []
        gene_rank_norms = []

        # Cosine accumulators
        cox_aux_cosines = []
        cox_rank_cosines = []
        aux_rank_cosines = []

        # Fisher score accumulators
        img_fisher_scores = []
        gene_fisher_scores = []

        total_events = 0
        total_pairs = 0

        has_aux = hasattr(model, "aux_img")

        for inputs, labels in probe_loader:
            x = {k: v.to(device) for k, v in inputs.items()}
            t = labels["os_months"].float().to(device)
            e = labels["vital_status"].float().to(device)

            # Forward (with grad enabled even in eval mode)
            with torch.enable_grad():
                outputs = model(x)
                risk = outputs["survival"][:, 1]
                cox_loss = cox_efron_free(risk, t, e)

                # Aux losses (available only for models with aux heads)
                if has_aux:
                    aux_img_loss = cox_efron_free(outputs["aux_img_risk"], t, e)
                    aux_gene_loss = cox_efron_free(outputs["aux_gene_risk"], t, e)
                    aux_loss = aux_img_loss + aux_gene_loss
                else:
                    aux_loss = cox_loss * 0.0

                rank_loss_val = ranking_loss(risk, t, e, temperature) if e.bool().any() else cox_loss * 0.0

                # --- Image encoder gradients ---
                img_cox_grad = _flat_grads(cox_loss, img_params)
                img_aux_grad = _flat_grads(aux_loss, img_params)
                img_rank_grad = _flat_grads(rank_loss_val, img_params)

                img_cox_norms.append(float(img_cox_grad.norm().item()))
                img_aux_norms.append(float(img_aux_grad.norm().item()))
                img_rank_norms.append(float(img_rank_grad.norm().item()))

                # --- Gene encoder gradients ---
                gene_cox_grad = _flat_grads(cox_loss, gene_params)
                gene_aux_grad = _flat_grads(aux_loss, gene_params)
                gene_rank_grad = _flat_grads(rank_loss_val, gene_params)

                gene_cox_norms.append(float(gene_cox_grad.norm().item()))
                gene_aux_norms.append(float(gene_aux_grad.norm().item()))
                gene_rank_norms.append(float(gene_rank_grad.norm().item()))

                # --- Cosine similarities (joint over both encoders) ---
                all_cox = torch.cat([img_cox_grad, gene_cox_grad])
                all_aux = torch.cat([img_aux_grad, gene_aux_grad])
                all_rank = torch.cat([img_rank_grad, gene_rank_grad])

                cox_aux_cosines.append(_cosine_sim(all_cox, all_aux))
                cox_rank_cosines.append(_cosine_sim(all_cox, all_rank))
                aux_rank_cosines.append(_cosine_sim(all_aux, all_rank))

                # --- Per-event Fisher scores ---
                img_scores = _per_event_scores(risk, t, e, img_params)
                gene_scores = _per_event_scores(risk, t, e, gene_params)
                img_fisher_scores.extend(img_scores)
                gene_fisher_scores.extend(gene_scores)

            # Count events and pairs
            n_ev = int(e.bool().sum().item())
            total_events += n_ev
            mask = comparable_pairs(t, e)
            total_pairs += int(mask.sum().item())

        # --- Aggregate encoder diagnostics ---
        if img_cox_norms:
            mean_img_norm = float(np.mean(img_cox_norms))
            diag.image_encoder.cox_grad_norm = mean_img_norm
            diag.image_encoder.squared_cox_grad_norm = float(np.mean([x**2 for x in img_cox_norms]))
            diag.image_encoder.grad_sensitivity = mean_img_norm / max(n_img_params, 1)
            diag.image_encoder.fisher_trace = approx_fisher_trace(img_fisher_scores)
            diag.image_encoder.fisher_trace_normalized = diag.image_encoder.fisher_trace / max(n_img_params, 1)
            diag.image_encoder.n_events_fisher = len(img_fisher_scores)
            diag.image_encoder.layer_fisher = per_layer_fisher(img_fisher_scores, img_param_shapes)

        if gene_cox_norms:
            mean_gene_norm = float(np.mean(gene_cox_norms))
            diag.gene_encoder.cox_grad_norm = mean_gene_norm
            diag.gene_encoder.squared_cox_grad_norm = float(np.mean([x**2 for x in gene_cox_norms]))
            diag.gene_encoder.grad_sensitivity = mean_gene_norm / max(n_gene_params, 1)
            diag.gene_encoder.fisher_trace = approx_fisher_trace(gene_fisher_scores)
            diag.gene_encoder.fisher_trace_normalized = diag.gene_encoder.fisher_trace / max(n_gene_params, 1)
            diag.gene_encoder.n_events_fisher = len(gene_fisher_scores)
            diag.gene_encoder.layer_fisher = per_layer_fisher(gene_fisher_scores, gene_param_shapes)

        # --- Alignment diagnostics ---
        if cox_aux_cosines:
            diag.alignment.cosine_cox_aux = float(np.mean(cox_aux_cosines))
            diag.alignment.cosine_cox_rank = float(np.mean(cox_rank_cosines))
            diag.alignment.cosine_aux_rank = float(np.mean(aux_rank_cosines))

        if img_cox_norms:
            mean_cn = np.mean(img_cox_norms)
            diag.alignment.cox_grad_variability = float(np.std(img_cox_norms) / max(mean_cn, 1e-12))
        if img_aux_norms:
            mean_an = np.mean(img_aux_norms)
            diag.alignment.aux_grad_variability = float(np.std(img_aux_norms) / max(mean_an, 1e-12))
        if img_rank_norms:
            mean_rn = np.mean(img_rank_norms)
            diag.alignment.rank_grad_variability = float(np.std(img_rank_norms) / max(mean_rn, 1e-12))

        diag.n_events_train = total_events
        diag.n_comparable_pairs = total_pairs

        # --- Validation C-indices ---
        diag.val_ci_fused = _compute_val_ci(model, val_loader, device)

        # Unimodal C-indices
        if has_aux:
            diag.val_ci_image_unimodal = _compute_unimodal_ci(
                model, val_loader, device, modality="image")
            diag.val_ci_gene_unimodal = _compute_unimodal_ci(
                model, val_loader, device, modality="gene")
        elif unimodal_probes is not None:
            # For baseline/rank: use frozen-feature linear probes
            diag.val_ci_image_unimodal = _compute_probe_ci(
                model, val_loader, device, unimodal_probes.get("image"), "image")
            diag.val_ci_gene_unimodal = _compute_probe_ci(
                model, val_loader, device, unimodal_probes.get("gene"), "gene")

        # GPU memory
        if device.type == "cuda":
            diag.gpu_memory_mb = torch.cuda.max_memory_allocated(device) / 1e6

    finally:
        # --- Restore state exactly ---
        if was_training:
            model.train()
        else:
            model.eval()

        for name, p in model.named_parameters():
            if saved_grads[name] is not None:
                p.grad = saved_grads[name]
            else:
                p.grad = None

        # Restore RNG
        torch.random.set_rng_state(rng_state)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        np.random.set_state(np_rng)

    return diag


# ---------------------------------------------------------------------------
# Validation C-index helpers
# ---------------------------------------------------------------------------

def _compute_val_ci(model, loader, device):
    """Fused model validation C-index."""
    risks, times, events = [], [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            x = {k: v.to(device) for k, v in inputs.items()}
            out = model(x)
            risks.extend(out["survival"][:, 1].cpu().tolist())
            times.extend(labels["os_months"].float().tolist())
            events.extend(labels["vital_status"].float().tolist())
    return float(concordance_index(times, risks, events))


def _compute_unimodal_ci(model, loader, device, modality):
    """Unimodal C-index via auxiliary heads (for models with aux heads)."""
    risks, times, events = [], [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            x = {k: v.to(device) for k, v in inputs.items()}
            out = model(x)
            key = f"aux_{modality[:3]}_risk"  # aux_img_risk or aux_gen_risk
            if key not in out:
                return 0.5
            risks.extend(out[key].cpu().tolist())
            times.extend(labels["os_months"].float().tolist())
            events.extend(labels["vital_status"].float().tolist())
    return float(concordance_index(times, risks, events))


# ---------------------------------------------------------------------------
# Frozen-feature linear Cox probes for baseline/rank methods
# ---------------------------------------------------------------------------

class LinearCoxProbe(nn.Module):
    """Single linear layer mapping encoder features to a scalar risk score.
    Fitted on training data with frozen encoder features.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def fit_unimodal_probes(model, train_loader, device, n_epochs=10, lr=0.01):
    """Fit frozen-feature linear Cox probes on training data.

    The model encoders are NOT modified. Features are extracted with
    no_grad, then a small linear head is trained on those features.

    Returns dict with "image" and "gene" LinearCoxProbe instances.
    """
    model.eval()

    # Extract features
    img_feats, gene_feats, times, events = [], [], [], []
    with torch.no_grad():
        for inputs, labels in train_loader:
            x = {k: v.to(device) for k, v in inputs.items()}
            img_f = model.image_encoder(x["image"])
            gene_f = model.gene_encoder(x["gene"])
            img_feats.append(img_f.cpu())
            gene_feats.append(gene_f.cpu())
            times.extend(labels["os_months"].float().tolist())
            events.extend(labels["vital_status"].float().tolist())

    img_feats = torch.cat(img_feats)
    gene_feats = torch.cat(gene_feats)
    t = torch.tensor(times, dtype=torch.float32)
    e = torch.tensor(events, dtype=torch.float32)

    probes = {}
    for name, feats in [("image", img_feats), ("gene", gene_feats)]:
        probe = LinearCoxProbe(feats.shape[1]).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=lr)
        feats_d = feats.to(device)
        t_d, e_d = t.to(device), e.to(device)

        for _ in range(n_epochs):
            opt.zero_grad()
            risk = probe(feats_d)
            loss = cox_efron_free(risk, t_d, e_d)
            if torch.isfinite(loss):
                loss.backward()
                opt.step()

        probe.eval()
        probes[name] = probe

    return probes


def _compute_probe_ci(model, val_loader, device, probe, modality):
    """Compute C-index using a frozen-feature linear probe on validation data."""
    if probe is None:
        return 0.5

    model.eval()
    probe.eval()
    risks, times, events = [], [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            x = {k: v.to(device) for k, v in inputs.items()}
            encoder = model.image_encoder if modality == "image" else model.gene_encoder
            feat = encoder(x[modality] if modality == "image" else x["gene"])
            risk = probe(feat)
            risks.extend(risk.cpu().tolist())
            times.extend(labels["os_months"].float().tolist())
            events.extend(labels["vital_status"].float().tolist())
    return float(concordance_index(times, risks, events))


# ---------------------------------------------------------------------------
# Flatten diagnostics to CSV-friendly dict
# ---------------------------------------------------------------------------

def diagnostics_to_row(diag: EpochDiagnostics) -> dict:
    """Flatten EpochDiagnostics to a flat dict for CSV output."""
    row = {
        "epoch": diag.epoch,
        "cancer": diag.cancer,
        "method": diag.method,
        "fold": diag.fold,
        # Image encoder
        "img_cox_grad_norm": diag.image_encoder.cox_grad_norm,
        "img_squared_cox_grad_norm": diag.image_encoder.squared_cox_grad_norm,
        "img_fisher_trace": diag.image_encoder.fisher_trace,
        "img_fisher_trace_normalized": diag.image_encoder.fisher_trace_normalized,
        "img_n_events_fisher": diag.image_encoder.n_events_fisher,
        "img_grad_sensitivity": diag.image_encoder.grad_sensitivity,
        # Gene encoder
        "gene_cox_grad_norm": diag.gene_encoder.cox_grad_norm,
        "gene_squared_cox_grad_norm": diag.gene_encoder.squared_cox_grad_norm,
        "gene_fisher_trace": diag.gene_encoder.fisher_trace,
        "gene_fisher_trace_normalized": diag.gene_encoder.fisher_trace_normalized,
        "gene_n_events_fisher": diag.gene_encoder.n_events_fisher,
        "gene_grad_sensitivity": diag.gene_encoder.grad_sensitivity,
        # Ratios
        "fisher_ratio_gene_over_img": (
            diag.gene_encoder.fisher_trace_normalized /
            max(diag.image_encoder.fisher_trace_normalized, 1e-15)
        ),
        "grad_norm_ratio_gene_over_img": (
            diag.gene_encoder.cox_grad_norm /
            max(diag.image_encoder.cox_grad_norm, 1e-15)
        ),
        # Alignment
        "cosine_cox_aux": diag.alignment.cosine_cox_aux,
        "cosine_cox_rank": diag.alignment.cosine_cox_rank,
        "cosine_aux_rank": diag.alignment.cosine_aux_rank,
        "cox_grad_variability": diag.alignment.cox_grad_variability,
        "aux_grad_variability": diag.alignment.aux_grad_variability,
        "rank_grad_variability": diag.alignment.rank_grad_variability,
        # Validation C-indices
        "val_ci_fused": diag.val_ci_fused,
        "val_ci_image_unimodal": diag.val_ci_image_unimodal,
        "val_ci_gene_unimodal": diag.val_ci_gene_unimodal,
        "val_ci_gap_gene_minus_img": diag.val_ci_gene_unimodal - diag.val_ci_image_unimodal,
        # Training losses
        "train_fusion_loss": diag.train_fusion_loss,
        "train_img_loss": diag.train_img_loss,
        "train_gene_loss": diag.train_gene_loss,
        "train_rank_loss": diag.train_rank_loss,
        # Survival stats
        "n_events_train": diag.n_events_train,
        "n_comparable_pairs": diag.n_comparable_pairs,
        # Resources
        "wall_seconds": diag.wall_seconds,
        "gpu_memory_mb": diag.gpu_memory_mb,
    }
    return row
