"""
Multimodal Chaotic Optimizer with M-SAM (Modality-Aware SAM) extension.

Combines three complementary mechanisms:
  1. ChaoticLRScheduler  — logistic-map LR scheduling (non-monotone exploration)
  2. Modality-Aware SAM  — SAM perturbation applied only to dominant modality
                           encoder, finding flatter minima for that modality
  3. OGM-GE             — On-the-fly Gradient Modulation + Generalization
                           Enhancement; suppresses dominant modality gradients

Contribution estimation uses Shapley values (more principled than unimodal
accuracy used in original OGM-GE):
    φ_image = 0.5 * acc_image + 0.5 * (acc_full - acc_gene)
    φ_gene  = 0.5 * acc_gene  + 0.5 * (acc_full - acc_image)
Requires 3 forward passes per OGM-GE step (image-only, gene-only, full).

SAM perturbation (dominant modality encoder only):
    ε = ρ * grad / ||grad||         (unit-normalised, scaled by ρ)
    θ_dom += ε                      (perturb)
    second forward + backward       (get gradients at perturbed point)
    θ_dom -= ε                      (restore)
Then OGM-GE modulation and base optimizer step follow.

Reference: M-SAM (NeurIPS 2025) — modality-aware sharpness-aware minimisation.

Usage:
    from optimizers.chaotic_optimizer_msam import MultimodalChaoticOptimizerMSAM

    base_opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9,
                               weight_decay=1e-4)
    msam_opt = MultimodalChaoticOptimizerMSAM(
        base_opt, model,
        modality_names=["image", "gene"],
        base_lr=1e-3,
        rho=0.05,       # SAM perturbation radius
        alpha=0.5,      # OGM-GE suppression strength
    )

    # training loop — requires closure
    for inputs, labels in loader:
        def closure():
            out  = model(inputs)
            loss = criterion(out, labels)
            return loss, out

        loss, out = msam_opt.step(closure, inputs, labels)
"""

import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple

try:
    from optimizers.chaotic_lr_scheduler import ChaoticLRScheduler
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from optimizers.chaotic_lr_scheduler import ChaoticLRScheduler


__all__ = ["MultimodalChaoticOptimizerMSAM"]


class MultimodalChaoticOptimizerMSAM:
    """
    Chaotic LR + Modality-Aware SAM + OGM-GE optimizer.

    Args:
        optimizer      : base PyTorch optimizer (SGD recommended)
        model          : JointModel instance
        modality_names : list of modality name strings, e.g. ["image", "gene"]
        base_lr        : base LR for ChaoticLRScheduler
        r              : logistic map parameter (default 3.99, fully chaotic)
        rho            : SAM perturbation radius (default 0.05)
        alpha          : OGM-GE suppression strength (default 0.5)
        use_ge         : inject Gaussian noise on dominant modality (default True)
        compute_every  : apply OGM-GE + SAM every N steps (default 1)
    """

    def __init__(self,
                 optimizer,
                 model:           nn.Module,
                 modality_names:  List[str],
                 base_lr:         float = 1e-3,
                 r:               float = 3.99,
                 rho:             float = 0.05,
                 alpha:           float = 0.5,
                 use_ge:          bool  = True,
                 compute_every:   int   = 1):
        self.optimizer      = optimizer
        self.model          = model
        self.modality_names = modality_names
        self.rho            = rho
        self.alpha          = alpha
        self.use_ge         = use_ge
        self.compute_every  = compute_every
        self._step_count    = 0

        self.scheduler = ChaoticLRScheduler(optimizer, base_lr=base_lr, r=r)

        # Contribution history for analysis
        self.contribution_history: Dict[str, List[float]] = {
            m: [] for m in modality_names
        }

    # ── Shapley contribution estimator ────────────────────────────────────────

    @torch.no_grad()
    def _compute_shapley(self,
                         inputs: Dict[str, torch.Tensor],
                         labels: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Estimate per-modality contribution via Shapley values.

        For 2 modalities (image, gene):
            φ_image = 0.5 * acc_image + 0.5 * (acc_full - acc_gene)
            φ_gene  = 0.5 * acc_gene  + 0.5 * (acc_full - acc_image)

        Requires 3 forward passes. Returns normalised values in [0, 1].
        """
        self.model.eval()
        surv_labels = labels.get("survival") if isinstance(labels, dict) else labels

        acc = {}

        # Unimodal passes
        for modality in self.modality_names:
            uni_input = {
                k: v if k == modality else torch.zeros_like(v)
                for k, v in inputs.items()
            }
            out   = self.model(uni_input)
            logits = out["survival"] if isinstance(out, dict) else out
            preds  = logits.argmax(dim=1)
            acc[modality] = (preds == surv_labels).float().mean().item()

        # Full model pass
        out_full   = self.model(inputs)
        logits_full = out_full["survival"] if isinstance(out_full, dict) else out_full
        preds_full  = logits_full.argmax(dim=1)
        acc["full"] = (preds_full == surv_labels).float().mean().item()

        self.model.train()

        # Shapley values for 2 modalities
        m0, m1 = self.modality_names[0], self.modality_names[1]
        phi = {
            m0: 0.5 * acc[m0] + 0.5 * (acc["full"] - acc[m1]),
            m1: 0.5 * acc[m1] + 0.5 * (acc["full"] - acc[m0]),
        }

        # Normalise to sum to 1 (keep non-negative)
        phi = {k: max(v, 0.0) for k, v in phi.items()}
        total = sum(phi.values())
        if total > 0:
            phi = {k: v / total for k, v in phi.items()}
        else:
            phi = {k: 1.0 / len(self.modality_names) for k in self.modality_names}

        return phi

    # ── Dominant modality encoder parameters ─────────────────────────────────

    def _encoder_params(self, modality: str):
        """Yield (name, param) for the given modality's encoder."""
        key = f"{modality}_encoder"
        for name, param in self.model.named_parameters():
            if key in name and param.requires_grad:
                yield name, param

    # ── SAM perturbation ──────────────────────────────────────────────────────

    def _sam_perturb(self, dominant: str) -> Dict[str, torch.Tensor]:
        """
        Compute and apply SAM perturbation to dominant modality encoder.
        Returns saved perturbations {param_name: epsilon} for restore step.
        """
        grad_norm = torch.norm(
            torch.stack([
                param.grad.norm()
                for _, param in self._encoder_params(dominant)
                if param.grad is not None
            ])
        )

        eps_store = {}
        if grad_norm > 1e-8:
            scale = self.rho / (grad_norm + 1e-12)
            for name, param in self._encoder_params(dominant):
                if param.grad is not None:
                    eps = param.grad * scale
                    eps_store[name] = eps.clone()
                    param.data.add_(eps)

        return eps_store

    def _sam_restore(self, dominant: str, eps_store: Dict[str, torch.Tensor]):
        """Restore dominant modality encoder parameters after SAM second pass."""
        for name, param in self._encoder_params(dominant):
            if name in eps_store:
                param.data.sub_(eps_store[name])

    # ── OGM-GE gradient modulation ────────────────────────────────────────────

    @torch.no_grad()
    def _modulate_gradients(self,
                            contributions: Dict[str, float],
                            dominant: str) -> None:
        """
        Suppress dominant modality gradients and optionally add GE noise.
        Non-dominant modalities are left unchanged (k = 1.0).
        """
        for modality, ratio in contributions.items():
            if modality == dominant and ratio > 0.5:
                k = 1.0 - float(torch.tanh(
                    torch.tensor(self.alpha * ratio)
                ))
            else:
                k = 1.0

            key = f"{modality}_encoder"
            for name, param in self.model.named_parameters():
                if key not in name or param.grad is None:
                    continue
                param.grad.mul_(k)
                if self.use_ge and modality == dominant and k < 1.0:
                    noise_std = param.grad.std() * 0.1
                    param.grad.add_(torch.randn_like(param.grad) * noise_std)

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self,
             closure:   Callable[[], Tuple[torch.Tensor, dict]],
             inputs:    Optional[Dict[str, torch.Tensor]] = None,
             labels:    Optional[Dict[str, torch.Tensor]] = None
             ) -> Tuple[torch.Tensor, dict]:
        """
        Full M-SAM step:
            1. First forward+backward  → get gradients
            2. Compute Shapley contributions
            3. SAM perturb dominant encoder
            4. Second forward+backward → get gradients at perturbed point
            5. Restore dominant encoder
            6. OGM-GE modulation on restored gradients
            7. Base optimizer step + chaotic LR step

        Args:
            closure : callable () → (loss, outputs_dict)
                      Should NOT call zero_grad internally.
            inputs  : modality input dict (for Shapley; can be None to skip)
            labels  : label dict with key "survival" (for Shapley; can be None)

        Returns:
            (loss, outputs) from the second forward pass.
        """
        self._step_count += 1
        apply_msam = (
            inputs is not None and
            labels is not None and
            self._step_count % self.compute_every == 0
        )

        # ── Pass 1: standard forward + backward ──────────────────────────────
        self.optimizer.zero_grad()
        loss, out = closure()
        loss.backward()

        if apply_msam:
            # ── Shapley contributions ─────────────────────────────────────────
            contributions = self._compute_shapley(inputs, labels)
            dominant      = max(contributions, key=contributions.get)

            for m in self.modality_names:
                self.contribution_history[m].append(contributions.get(m, 0.0))

            # ── SAM: perturb dominant encoder ────────────────────────────────
            eps_store = self._sam_perturb(dominant)

            # ── Pass 2: forward + backward at perturbed point ────────────────
            self.optimizer.zero_grad()
            loss, out = closure()
            loss.backward()

            # ── Restore dominant encoder ─────────────────────────────────────
            self._sam_restore(dominant, eps_store)

            # ── OGM-GE modulation ────────────────────────────────────────────
            self._modulate_gradients(contributions, dominant)

        # ── Base optimizer + chaotic LR step ─────────────────────────────────
        self.optimizer.step()
        self.scheduler.step()

        return loss, out

    def get_current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def get_contribution_history(self) -> Dict[str, List[float]]:
        """Returns per-modality Shapley contribution history."""
        return self.contribution_history


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.joint_model import build_model

    B, N_GENES, IMG_DIM = 4, 1000, 256

    model    = build_model(n_genes=N_GENES, image_dim=IMG_DIM,
                           n_subtypes=4, verbose=False)
    base_opt = torch.optim.SGD(model.parameters(), lr=1e-3,
                                momentum=0.9, weight_decay=1e-4)
    msam_opt = MultimodalChaoticOptimizerMSAM(
        base_opt, model,
        modality_names=["image", "gene"],
        base_lr=1e-3,
        rho=0.05, alpha=0.5,
        compute_every=1,
    )

    criterion_surv = torch.nn.CrossEntropyLoss()
    criterion_sub  = torch.nn.CrossEntropyLoss()

    lrs = []
    for step in range(5):
        inputs = {
            "image": torch.randn(B, IMG_DIM),
            "gene":  torch.randn(B, N_GENES),
        }
        labels = {
            "survival": torch.randint(0, 2, (B,)),
            "subtype":  torch.randint(0, 4, (B,)),
        }

        model.train()

        def closure():
            out  = model(inputs)
            loss = (criterion_surv(out["survival"], labels["survival"]) +
                    1.5 * criterion_sub(out["subtype"], labels["subtype"]))
            return loss, out

        loss, out = msam_opt.step(closure, inputs, labels)
        lrs.append(msam_opt.get_current_lr())
        print(f"  step {step+1}  loss={loss.item():.4f}  lr={lrs[-1]:.6f}")

    hist = msam_opt.get_contribution_history()
    assert "image" in hist and "gene" in hist
    assert len(hist["image"]) == 5
    print(f"\nShapley image: {[f'{v:.3f}' for v in hist['image']]}")
    print(f"Shapley gene : {[f'{v:.3f}' for v in hist['gene']]}")
    print("All MultimodalChaoticOptimizerMSAM self-tests PASSED.")
