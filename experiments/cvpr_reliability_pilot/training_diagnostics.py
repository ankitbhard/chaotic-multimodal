"""
Training dynamics diagnostics for PCRF Phase 2.

Captures 18 categories of optimizer/training metrics:

Per-step (every step):
  1. Gradient norm ||g_t||
  2. Gradient-to-weight ratio
  3. First-moment norm ||m_hat_t||  (bias-corrected)
  4. Second-moment mean v_hat_t     (bias-corrected)
  5. Effective LR distribution (p10, p50, p90)
  6. Update norm ||Δθ_t||
  7. Update-to-weight ratio
  8. Gradient–update alignment cos(g, Δθ)
  9. Update oscillation cos(Δθ_t, Δθ_{t-1})
 10. Predicted delta g^T Δθ

Per-epoch (aggregated from steps):
 11. Gradient variance across mini-batches
 12. Consecutive gradient cosine similarity (batch-to-batch)
 13. Moment adaptation speed (epoch-over-epoch change)
 14. Train–validation loss gap
 15. C-index trajectory

Periodic (every N epochs):
 16. Top Hessian eigenvalue (power iteration)
 17. Hessian trace (Hutchinson estimator)
 18. Update-direction curvature u^T H u / ||u||^2
"""

import csv, math
from pathlib import Path
import torch
import torch.nn.functional as F


# ── Parameter grouping ───────────────────────────────────────────────────────

def _param_group(name):
    """Map parameter name to a semantic group."""
    for prefix in ("image_encoder", "gene_encoder", "proj_img", "proj_gene",
                   "reliability_img", "reliability_gene", "shared",
                   "head_survival", "aux_img", "aux_gene", "fusion"):
        if name.startswith(prefix):
            return prefix
    return "other"


# ── Per-step diagnostics ─────────────────────────────────────────────────────

def compute_step_diagnostics(model, optimizer, before_params, raw_grads, lr, prev_update):
    """
    Compute per-step optimizer diagnostics, grouped by parameter group.

    Args:
        model: the model after optimizer.step()
        optimizer: AdamW optimizer (has .state with exp_avg, exp_avg_sq)
        before_params: {name: param_tensor_clone} before optimizer.step()
        raw_grads: {name: grad_clone} captured after backward, before step
        lr: current learning rate
        prev_update: {group: update_vector} from previous step (mutated in-place)

    Returns:
        list of dicts, one per parameter group
    """
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad or name not in before_params or name not in raw_grads:
            continue
        group = _param_group(name)
        d = groups.setdefault(group, {"g": [], "m": [], "v": [], "scale": [],
                                      "u": [], "w": []})
        state = optimizer.state.get(p, {})
        if not state or "exp_avg" not in state:
            continue

        t = state["step"]
        t = float(t.item()) if torch.is_tensor(t) else float(t)
        if t == 0:
            continue

        beta1 = optimizer.defaults["betas"][0]
        beta2 = optimizer.defaults["betas"][1]
        eps = optimizer.defaults["eps"]

        # Bias-corrected moments
        m_hat = state["exp_avg"].detach() / (1 - beta1 ** t)
        v_hat = state["exp_avg_sq"].detach() / (1 - beta2 ** t)

        flat = lambda x: x.reshape(-1).float().cpu()
        d["g"].append(flat(raw_grads[name]))
        d["m"].append(flat(m_hat))
        d["v"].append(flat(v_hat))
        d["scale"].append(flat(lr / (v_hat.sqrt() + eps)))
        d["u"].append(flat(p.detach() - before_params[name]))
        d["w"].append(flat(before_params[name]))

    rows = []
    for group, d in groups.items():
        if not d["g"]:
            continue
        z = {k: torch.cat(v) for k, v in d.items() if v}
        g, u, w = z["g"], z["u"], z["w"]
        scale = z["scale"]

        norm = lambda x: float(torch.linalg.vector_norm(x))
        def cos(a, b):
            na, nb = norm(a), norm(b)
            if na > 0 and nb > 0:
                return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))
            return float("nan")

        w_norm = norm(w)
        row = {
            "group": group,
            "n_params": int(g.numel()),
            # Gradient metrics
            "grad_norm": norm(g),
            "grad_weight_ratio": norm(g) / (w_norm + 1e-12),
            # Adam moment metrics
            "m_norm": norm(z["m"]),
            "v_mean": float(z["v"].mean()),
            "v_max": float(z["v"].max()),
            # Effective LR distribution
            "effective_lr_p10": float(torch.quantile(scale, 0.1)),
            "effective_lr_p50": float(torch.quantile(scale, 0.5)),
            "effective_lr_p90": float(torch.quantile(scale, 0.9)),
            # Update metrics
            "update_norm": norm(u),
            "update_weight_ratio": norm(u) / (w_norm + 1e-12),
            # Alignment metrics
            "grad_update_cos": cos(g, u),
            "predicted_delta": float(torch.dot(g, u)),
            # Oscillation: cosine with previous step's update
            "update_prev_cos": (
                cos(u, prev_update[group])
                if group in prev_update and prev_update[group].numel() == u.numel()
                else float("nan")
            ),
        }
        prev_update[group] = u
        rows.append(row)

    return rows


# ── Batch-level gradient tracking ────────────────────────────────────────────

class EpochGradientTracker:
    """Track gradient statistics across batches within an epoch."""

    def __init__(self):
        self.batch_grad_norms = {}       # group -> [norms]
        self.prev_batch_grads = {}       # group -> flat gradient vector
        self.batch_grad_cosines = {}     # group -> [cosine values]

    def record_batch(self, model):
        """Call after backward, before optimizer.step()."""
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            group = _param_group(name)
            flat_g = p.grad.detach().reshape(-1).float()

            # Accumulate into group
            if group not in self.batch_grad_norms:
                self.batch_grad_norms[group] = []
                self.batch_grad_cosines[group] = []

            # For norms: aggregate per-param then combine
            # Simpler: track total gradient norm per group per batch
            # We'll aggregate across params at end

        # Compute per-group aggregate
        group_grads = {}
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            group = _param_group(name)
            if group not in group_grads:
                group_grads[group] = []
            group_grads[group].append(p.grad.detach().reshape(-1).float().cpu())

        for group, parts in group_grads.items():
            cat = torch.cat(parts)
            gnorm = float(torch.linalg.vector_norm(cat))

            if group not in self.batch_grad_norms:
                self.batch_grad_norms[group] = []
                self.batch_grad_cosines[group] = []

            self.batch_grad_norms[group].append(gnorm)

            # Consecutive gradient cosine
            if group in self.prev_batch_grads:
                prev = self.prev_batch_grads[group]
                if prev.numel() == cat.numel():
                    n1 = torch.linalg.vector_norm(prev)
                    n2 = torch.linalg.vector_norm(cat)
                    if n1 > 0 and n2 > 0:
                        cos_val = float(F.cosine_similarity(
                            prev.unsqueeze(0), cat.unsqueeze(0)))
                        self.batch_grad_cosines[group].append(cos_val)

            self.prev_batch_grads[group] = cat

    def get_epoch_stats(self):
        """Return per-group gradient statistics for the epoch."""
        stats = {}
        for group in self.batch_grad_norms:
            norms = self.batch_grad_norms[group]
            cosines = self.batch_grad_cosines.get(group, [])
            stats[group] = {
                "grad_norm_mean": float(torch.tensor(norms).mean()) if norms else 0,
                "grad_norm_std": float(torch.tensor(norms).std()) if len(norms) > 1 else 0,
                "grad_noise_ratio": (
                    float(torch.tensor(norms).std() / (torch.tensor(norms).mean() + 1e-12))
                    if len(norms) > 1 else 0
                ),
                "consecutive_grad_cos_mean": (
                    float(torch.tensor(cosines).mean()) if cosines else float("nan")
                ),
                "consecutive_grad_cos_std": (
                    float(torch.tensor(cosines).std()) if len(cosines) > 1 else 0
                ),
                "n_batches": len(norms),
            }
        return stats


# ── Hessian diagnostics ──────────────────────────────────────────────────────

def _hessian_vector_product(loss_fn, model, batch, device, v):
    """
    Compute Hv (Hessian-vector product) via two backward passes.

    Args:
        loss_fn: callable(model, batch, device) -> scalar loss
        model: the model
        batch: (inputs, labels) tuple
        device: torch device
        v: dict {name: tensor} with same shapes as model parameters

    Returns:
        dict {name: Hv tensor}
    """
    # Forward + backward to get gradients
    model.zero_grad()
    loss = loss_fn(model, batch, device)
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)

    # Compute g^T v — skip params with None gradients
    gv = sum(
        (g * v_i).sum()
        for g, (n, p), v_i in zip(grads, model.named_parameters(), v.values())
        if p.requires_grad and g is not None
    )

    # Second backward: d(g^T v)/d(params) = Hv
    hvp = torch.autograd.grad(gv, params, allow_unused=True)

    result = {}
    for (name, p), hv in zip(model.named_parameters(), hvp):
        if p.requires_grad and hv is not None:
            result[name] = hv.detach()

    return result


def compute_hessian_diagnostics(model, loss_fn, batch, device,
                                n_power_iter=20, n_hutchinson=10,
                                last_update=None):
    """
    Compute Hessian-based diagnostics:
      - Top eigenvalue via power iteration
      - Trace via Hutchinson estimator
      - Update-direction curvature (if last_update provided)

    Args:
        model: the model (must be in train mode)
        loss_fn: callable(model, batch, device) -> scalar loss
        batch: a single training batch
        device: torch device
        n_power_iter: iterations for power method
        n_hutchinson: samples for Hutchinson trace
        last_update: dict {name: Δθ tensor} for update-direction curvature

    Returns:
        dict with hessian metrics
    """
    model.train()
    params_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}

    # --- Top eigenvalue via power iteration ---
    # Initialize random vector
    v = {n: torch.randn_like(p) for n, p in params_dict.items()}
    v_norm = math.sqrt(sum((vi ** 2).sum().item() for vi in v.values()))
    v = {n: vi / (v_norm + 1e-12) for n, vi in v.items()}

    top_eigenvalue = 0.0
    for _ in range(n_power_iter):
        hv = _hessian_vector_product(loss_fn, model, batch, device, v)
        # Rayleigh quotient: λ ≈ v^T Hv
        eigenvalue = sum((vi * hv[n]).sum().item() for n, vi in v.items() if n in hv)
        # Normalize: v = Hv / ||Hv||
        hv_norm = math.sqrt(sum((hvi ** 2).sum().item() for hvi in hv.values()))
        if hv_norm > 1e-12:
            v = {n: hvi / hv_norm for n, hvi in hv.items()}
        top_eigenvalue = eigenvalue

    # --- Trace via Hutchinson estimator ---
    trace_estimates = []
    for _ in range(n_hutchinson):
        # Rademacher random vector
        z = {n: torch.randint(0, 2, p.shape, device=device).float() * 2 - 1
             for n, p in params_dict.items()}
        hz = _hessian_vector_product(loss_fn, model, batch, device, z)
        # tr(H) ≈ z^T Hz
        tr_est = sum((zi * hz[n]).sum().item() for n, zi in z.items() if n in hz)
        trace_estimates.append(tr_est)

    trace = float(torch.tensor(trace_estimates).mean())

    # --- Update-direction curvature ---
    update_curvature = float("nan")
    if last_update is not None:
        u_norm_sq = sum((ui ** 2).sum().item() for ui in last_update.values())
        if u_norm_sq > 1e-12:
            hu = _hessian_vector_product(loss_fn, model, batch, device, last_update)
            uhu = sum((ui * hu[n]).sum().item()
                      for n, ui in last_update.items() if n in hu)
            update_curvature = uhu / u_norm_sq

    return {
        "hessian_top_eigenvalue": top_eigenvalue,
        "hessian_trace": trace,
        "update_direction_curvature": update_curvature,
    }


# ── Moment adaptation speed ─────────────────────────────────────────────────

class MomentTracker:
    """Track epoch-over-epoch changes in Adam moments."""

    def __init__(self):
        self.prev_m_norms = {}
        self.prev_v_means = {}

    def compute_speed(self, optimizer, model):
        """Compute moment adaptation speed from current optimizer state."""
        group_moments = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            state = optimizer.state.get(p, {})
            if not state or "exp_avg" not in state:
                continue
            group = _param_group(name)
            if group not in group_moments:
                group_moments[group] = {"m": [], "v": []}
            group_moments[group]["m"].append(
                state["exp_avg"].detach().reshape(-1).float().cpu())
            group_moments[group]["v"].append(
                state["exp_avg_sq"].detach().reshape(-1).float().cpu())

        results = {}
        for group, mv in group_moments.items():
            m_cat = torch.cat(mv["m"])
            v_cat = torch.cat(mv["v"])
            m_norm = float(torch.linalg.vector_norm(m_cat))
            v_mean = float(v_cat.mean())

            m_speed = abs(m_norm - self.prev_m_norms.get(group, m_norm))
            v_speed = abs(v_mean - self.prev_v_means.get(group, v_mean))

            self.prev_m_norms[group] = m_norm
            self.prev_v_means[group] = v_mean

            results[group] = {
                "moment_m_norm": m_norm,
                "moment_v_mean": v_mean,
                "moment_m_speed": m_speed,
                "moment_v_speed": v_speed,
            }
        return results


# ── CSV I/O ──────────────────────────────────────────────────────────────────

def write_diagnostics_csv(path, rows):
    """Write list of dicts to CSV."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_epoch_diagnostics_csv(path, rows):
    """Write epoch-level diagnostics to CSV."""
    write_diagnostics_csv(path, rows)
