"""Stability-Gated Curvature AdamW (SGC-AdamW).

Three ablation modes beyond a disabled baseline:
  - drift:                   adaptive beta2 from squared-gradient drift
  - drift_alignment:         + gradient-momentum alignment signal
  - drift_alignment_curvature: + same-minibatch curvature gating

Disabled mode reproduces torch.optim.AdamW exactly (decoupled WD, standard
bias correction). Variable-beta2 modes use cumulative-product bias correction.
"""
import math
import time
import torch
from torch.optim import Optimizer

MODES = ("disabled", "drift", "drift_alignment", "drift_alignment_curvature")


class SGCAdamW(Optimizer):
    r"""Stability-Gated Curvature AdamW.

    Args:
        params: iterable of parameters or param groups
        lr: learning rate (default: 1e-3)
        betas: (beta1, beta2_max) coefficients (default: (0.9, 0.999))
        eps: term added to denominator for numerical stability (default: 1e-8)
        weight_decay: decoupled weight decay coefficient (default: 1e-4)
        mode: one of "disabled", "drift", "drift_alignment",
              "drift_alignment_curvature" (default: "drift_alignment")
        beta2_min: lower bound for adaptive beta2 (default: 0.95)
        drift_sensitivity: scaling factor for drift signal (default: 3.0)
        drift_ema_decay: EMA decay for drift signal (default: 0.9)
        alignment_sensitivity: scaling factor for alignment signal (default: 1.0)
        alignment_ema_decay: EMA decay for alignment signal (default: 0.9)
        curvature_sensitivity: scaling factor for curvature signal (default: 1.0)
        curvature_ema_decay: EMA decay for curvature estimate (default: 0.9)
        eps_change: epsilon for drift denominator (default: 1e-6)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-4, mode="drift_alignment",
                 beta2_min=0.95, drift_sensitivity=3.0, drift_ema_decay=0.9,
                 alignment_sensitivity=1.0, alignment_ema_decay=0.9,
                 curvature_sensitivity=1.0, curvature_ema_decay=0.9,
                 eps_change=1e-6):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        beta1, beta2_max = betas
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"invalid beta1: {beta1}")
        if not (0.0 < beta2_min <= beta2_max < 1.0):
            raise ValueError(f"invalid beta2 bounds: beta2_min={beta2_min}, "
                             f"beta2_max={beta2_max}")
        if lr <= 0:
            raise ValueError(f"invalid lr: {lr}")
        if eps <= 0:
            raise ValueError(f"invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            mode=mode, beta2_min=beta2_min,
            drift_sensitivity=drift_sensitivity,
            drift_ema_decay=drift_ema_decay,
            alignment_sensitivity=alignment_sensitivity,
            alignment_ema_decay=alignment_ema_decay,
            curvature_sensitivity=curvature_sensitivity,
            curvature_ema_decay=curvature_ema_decay,
            eps_change=eps_change,
        )
        super().__init__(params, defaults)
        self._global_step = 0
        self._start_time = time.monotonic()

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2_max = group["betas"]
            mode = group.get("mode", self.defaults["mode"])
            use_curvature = (mode == "drift_alignment_curvature")
            use_alignment = mode in ("drift_alignment",
                                     "drift_alignment_curvature")
            use_drift = mode in ("drift", "drift_alignment",
                                 "drift_alignment_curvature")

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("SGCAdamW does not support sparse "
                                       "gradients")
                if not torch.isfinite(g).all():
                    continue  # skip non-finite gradients

                state = self.state[p]

                # --- Lazy init ---
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["beta2_product"] = 1.0
                    state["drift_ema"] = 0.0
                    state["alignment_ema"] = 0.0
                    state["curvature_ema"] = 0.0
                    state["last_beta2"] = beta2_max
                    if use_curvature:
                        state["prev_param"] = p.detach().clone()

                state["step"] += 1
                t = state["step"]
                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                g2 = g.square()

                # --- Adaptive beta2 ---
                if mode == "disabled":
                    beta2 = beta2_max
                else:
                    # Drift signal
                    if use_drift and t > 1:
                        disc = ((g2 - v).abs() /
                                (g2 + v + group["eps_change"])).mean().item()
                        ema_d = group["drift_ema_decay"]
                        state["drift_ema"] = (ema_d * state["drift_ema"]
                                              + (1.0 - ema_d) * disc)
                        drift_sig = (group["drift_sensitivity"]
                                     * state["drift_ema"])
                    else:
                        drift_sig = 0.0

                    # Alignment signal
                    if use_alignment and t > 1:
                        m_norm = torch.linalg.vector_norm(m)
                        g_norm = torch.linalg.vector_norm(g)
                        if m_norm > 0 and g_norm > 0:
                            cos = (torch.dot(m.reshape(-1),
                                             g.reshape(-1))
                                   / (m_norm * g_norm)).item()
                            misalign = (1.0 - cos) / 2.0  # [0, 1]
                        else:
                            misalign = 0.0
                        ema_a = group["alignment_ema_decay"]
                        state["alignment_ema"] = (
                            ema_a * state["alignment_ema"]
                            + (1.0 - ema_a) * misalign)
                        align_sig = (group["alignment_sensitivity"]
                                     * state["alignment_ema"])
                    else:
                        align_sig = 0.0

                    # Curvature signal (updated by curvature_probe)
                    if use_curvature:
                        curv_sig = (group["curvature_sensitivity"]
                                    * state["curvature_ema"])
                    else:
                        curv_sig = 0.0

                    combined = drift_sig + align_sig + curv_sig
                    beta2 = (group["beta2_min"]
                             + (beta2_max - group["beta2_min"])
                             * math.exp(-combined))

                state["last_beta2"] = beta2

                # --- Save prev_param BEFORE update for curvature ---
                if use_curvature and "prev_param" in state:
                    state["prev_param"].copy_(p.detach())

                # --- Bias-correction bookkeeping ---
                if mode == "disabled":
                    bc2 = 1.0 - beta2_max ** t
                else:
                    state["beta2_product"] *= beta2
                    bc2 = 1.0 - state["beta2_product"]
                bc1 = 1.0 - beta1 ** t

                # --- Moment updates ---
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).add_(g2, alpha=1.0 - beta2)

                # --- Bias-corrected estimates ---
                m_hat = m / bc1
                v_hat = v / bc2

                # --- Decoupled weight decay then Adam update ---
                p.mul_(1.0 - lr * group["weight_decay"])
                p.addcdiv_(m_hat, v_hat.sqrt().add_(group["eps"]),
                           value=-lr)

        return loss

    # ------------------------------------------------------------------
    # Curvature probe — call between backward() and step()
    # ------------------------------------------------------------------
    def curvature_probe(self, closure):
        """Same-minibatch curvature estimate.

        Call after loss.backward() but BEFORE step(). ``closure`` should
        return a scalar loss (no need to call backward — this method does it).

        Returns a dict with probe metadata for logging.
        """
        probe_start = time.monotonic()
        saved_params = {}
        saved_grads = {}
        current_grads = {}
        prev_params = {}

        # Collect curvature-mode params
        for group in self.param_groups:
            mode = group.get("mode", self.defaults["mode"])
            if mode != "drift_alignment_curvature":
                continue
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state.get(p, {})
                if "prev_param" not in state:
                    continue
                # Skip first step — prev_param == current param
                if state.get("step", 0) < 1:
                    continue
                saved_params[p] = p.data.clone()
                saved_grads[p] = p.grad.clone()
                current_grads[p] = p.grad.clone()
                prev_params[p] = state["prev_param"]

        if not saved_params:
            return {"elapsed_s": 0.0, "n_params": 0}

        # Save RNG state for reproducibility
        rng_state = torch.random.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state()
                    if torch.cuda.is_available() and
                    next(iter(saved_params)).is_cuda else None)

        # Load prev_param into model
        for p in saved_params:
            p.data.copy_(prev_params[p])

        # Zero grads and recompute on same batch with prev params
        for p in saved_params:
            p.grad.zero_()

        with torch.enable_grad():
            loss = closure()
            loss.backward()

        # Compute per-parameter curvature
        n_updated = 0
        for group in self.param_groups:
            mode = group.get("mode", self.defaults["mode"])
            if mode != "drift_alignment_curvature":
                continue
            ema_c = group.get("curvature_ema_decay", 0.9)
            for p in group["params"]:
                if p not in saved_params or p.grad is None:
                    continue
                state = self.state[p]
                g_curr = current_grads[p]
                g_prev = p.grad
                param_diff = torch.linalg.vector_norm(
                    saved_params[p] - prev_params[p]).item()
                grad_diff = torch.linalg.vector_norm(
                    g_curr - g_prev).item()
                curv = (grad_diff / param_diff
                        if param_diff > 1e-12 else 0.0)
                state["curvature_ema"] = (
                    ema_c * state["curvature_ema"]
                    + (1.0 - ema_c) * curv)
                n_updated += 1

        # Restore params, grads, and RNG
        for p, saved in saved_params.items():
            p.data.copy_(saved)
        for p, grad in saved_grads.items():
            p.grad.copy_(grad)
        torch.random.set_rng_state(rng_state)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng)

        elapsed = time.monotonic() - probe_start
        return {"elapsed_s": elapsed, "n_params": n_updated}

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------
    def get_telemetry(self):
        """Per-group aggregated telemetry for logging."""
        elapsed = time.monotonic() - self._start_time
        out = {}
        for gi, group in enumerate(self.param_groups):
            name = group.get("name", f"group_{gi}")
            beta2s, drifts, aligns, curvs = [], [], [], []
            for p in group["params"]:
                s = self.state.get(p, {})
                if not s:
                    continue
                beta2s.append(s.get("last_beta2", group["betas"][1]))
                drifts.append(s.get("drift_ema", 0.0))
                aligns.append(s.get("alignment_ema", 0.0))
                curvs.append(s.get("curvature_ema", 0.0))
            if not beta2s:
                continue
            n = len(beta2s)
            out[name] = {
                "beta2_mean": sum(beta2s) / n,
                "beta2_min": min(beta2s),
                "beta2_max": max(beta2s),
                "drift_ema_mean": sum(drifts) / n,
                "alignment_ema_mean": sum(aligns) / n,
                "curvature_ema_mean": sum(curvs) / n,
                "global_step": self._global_step,
                "elapsed_s": elapsed,
            }
        return out


# ------------------------------------------------------------------
# Factory helpers
# ------------------------------------------------------------------
def make_sgc_adamw(params, mode, lr=1e-3, weight_decay=1e-4,
                   beta2_max=0.999, **kwargs):
    """Convenience factory for the experiment matrix."""
    return SGCAdamW(params, lr=lr, betas=(0.9, beta2_max), eps=1e-8,
                    weight_decay=weight_decay, mode=mode, **kwargs)


def make_fixed_adamw(params, beta2, lr=1e-3, weight_decay=1e-4):
    """Standard AdamW with a specific fixed beta2."""
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, beta2),
                             eps=1e-8, weight_decay=weight_decay)
