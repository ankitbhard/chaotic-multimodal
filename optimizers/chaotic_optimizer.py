"""
Multimodal Chaotic Optimizer with OGM-GE gradient modulation.

Combines:
  1. ChaoticLRScheduler — logistic-map LR scheduling
  2. OGM-GE (On-the-fly Gradient Modulation + Generalization Enhancement)
     — estimates per-modality informativeness from unimodal accuracy
     — suppresses dominant modality's gradients via k = 1 − tanh(α · ratio)
     — injects Gaussian noise on dominant modality for generalization

Designed for JointModel where encoders are named:
    model.image_encoder.*  →  modality "image"
    model.gene_encoder.*   →  modality "gene"

Usage:
    from optimizers.chaotic_optimizer import MultimodalChaoticOptimizer

    base_opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9,
                               weight_decay=1e-4)
    chaotic  = MultimodalChaoticOptimizer(
        base_opt, model,
        modality_names=["image", "gene"],
        base_lr=1e-3,
    )

    # training loop
    for batch in loader:
        loss = compute_loss(model, batch)
        chaotic.step(loss, batch["inputs"], batch["labels"])
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Optional

try:
    from optimizers.chaotic_lr_scheduler import ChaoticLRScheduler
except ModuleNotFoundError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from optimizers.chaotic_lr_scheduler import ChaoticLRScheduler


# ── Core scheduler (re-exported for import convenience) ──────────────────────

__all__ = ["MultimodalChaoticOptimizer"]


# ── Optimizer wrapper ─────────────────────────────────────────────────────────

class MultimodalChaoticOptimizer:
    """
    Wraps a base PyTorch optimizer with:
      - chaotic LR scheduling (logistic map)
      - OGM-GE gradient modulation per modality

    Args:
        optimizer      : base optimizer (e.g. SGD or Adam)
        model          : JointModel instance
        modality_names : list of modality name strings that appear in
                         parameter names, e.g. ["image", "gene"]
        base_lr        : base LR passed to ChaoticLRScheduler
        r              : logistic map parameter (default 3.99)
        T_max          : total training epochs for cosine decay envelope (None = no decay)
        alpha          : gradient modulation strength (default 0.5)
        use_ge         : inject Gaussian noise for generalization (default True)
        compute_every  : compute OGM-GE every N steps (default 1)
    """

    def __init__(self,
                 optimizer,
                 model:           nn.Module,
                 modality_names:  List[str],
                 base_lr:         float = 1e-3,
                 r:               float = 3.99,
                 T_max:           int   = None,
                 alpha:           float = 0.5,
                 use_ge:          bool  = True,
                 compute_every:   int   = 1):
        self.optimizer      = optimizer
        self.model          = model
        self.modality_names = modality_names
        self.alpha          = alpha
        self.use_ge         = use_ge
        self.compute_every  = compute_every
        self._step_count    = 0

        self.scheduler = ChaoticLRScheduler(optimizer, base_lr=base_lr, r=r, T_max=T_max)

        # Contribution history for analysis
        self.contribution_history: Dict[str, List[float]] = {
            m: [] for m in modality_names
        }

    # ── Unimodal accuracy estimator ───────────────────────────────────────────

    @torch.no_grad()
    def _compute_contributions(self,
                               inputs: Dict[str, torch.Tensor],
                               labels: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Approximate per-modality contribution via unimodal survival accuracy.

        Zero out every modality except the one being evaluated, run a forward
        pass, and measure classification accuracy on the survival head.
        Returns normalized contributions that sum to 1.
        """
        self.model.eval()
        contributions: Dict[str, float] = {}

        surv_labels = labels.get("survival") if isinstance(labels, dict) else labels

        for modality in self.modality_names:
            uni_input = {
                k: v if k == modality else torch.zeros_like(v)
                for k, v in inputs.items()
            }
            out    = self.model(uni_input)
            logits = out["survival"] if isinstance(out, dict) else out
            preds  = logits.argmax(dim=1)
            acc    = (preds == surv_labels).float().mean().item()
            contributions[modality] = acc

        self.model.train()

        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}
        return contributions

    # ── Gradient modulation ───────────────────────────────────────────────────

    @torch.no_grad()
    def _modulate_gradients(self, contributions: Dict[str, float]) -> None:
        """
        Suppress the dominant modality's gradients and optionally add GE noise.

        For the dominant modality d with normalized ratio ρ_d > 0.5:
            k = 1 − tanh(α · ρ_d)   (k < 1 → suppress)
        Other modalities: k = 1 (unchanged).
        """
        dominant = max(contributions, key=contributions.get)

        for modality, ratio in contributions.items():
            if modality == dominant and ratio > 0.5:
                k = 1.0 - float(torch.tanh(
                    torch.tensor(self.alpha * ratio)
                ))
            else:
                k = 1.0

            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                # Match by encoder name prefix (image_encoder / gene_encoder)
                encoder_key = f"{modality}_encoder"
                if encoder_key in name:
                    param.grad.mul_(k)
                    if self.use_ge and modality == dominant and k < 1.0:
                        noise_std = param.grad.std() * 0.1
                        param.grad.add_(torch.randn_like(param.grad) * noise_std)

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self,
             loss:   torch.Tensor,
             inputs: Optional[Dict[str, torch.Tensor]] = None,
             labels: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """
        Backward + gradient modulation + optimizer + scheduler step.

        Args:
            loss   : scalar loss tensor (already computed from forward pass)
            inputs : modality input dict (needed for OGM-GE; skip if None)
            labels : label dict with key "survival" (needed for OGM-GE; skip if None)
        """
        self.optimizer.zero_grad()
        loss.backward()

        self._step_count += 1
        apply_ogm = (
            inputs is not None and
            labels is not None and
            self._step_count % self.compute_every == 0
        )

        if apply_ogm:
            contributions = self._compute_contributions(inputs, labels)
            self._modulate_gradients(contributions)
            for m in self.modality_names:
                self.contribution_history[m].append(contributions.get(m, 0.0))

        self.optimizer.step()
        self.scheduler.step()

    def get_current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def get_contribution_history(self) -> Dict[str, List[float]]:
        """Returns per-modality contribution history (normalized, [0,1])."""
        return self.contribution_history


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.joint_model import build_model

    B       = 4
    N_GENES = 1000   # small for speed
    IMG_DIM = 256

    model   = build_model(n_genes=N_GENES, image_dim=IMG_DIM, verbose=False)
    base_opt = torch.optim.SGD(model.parameters(), lr=1e-3,
                               momentum=0.9, weight_decay=1e-4)
    chaotic  = MultimodalChaoticOptimizer(
        base_opt, model,
        modality_names=["image", "gene"],
        base_lr=1e-3,
        compute_every=1,
    )

    criterion = torch.nn.CrossEntropyLoss()

    lrs = []
    for step in range(10):
        inputs = {
            "image": torch.randn(B, IMG_DIM),
            "gene":  torch.randn(B, N_GENES),
        }
        labels = {"survival": torch.randint(0, 2, (B,))}

        model.train()
        out  = model(inputs)
        loss = criterion(out["survival"], labels["survival"])

        chaotic.step(loss, inputs, labels)
        lrs.append(chaotic.get_current_lr())

    assert all(lr > 0 for lr in lrs), "LR should be positive"
    hist = chaotic.get_contribution_history()
    assert "image" in hist and "gene" in hist, "Missing contribution keys"
    assert len(hist["image"]) == 10, "Contribution history length mismatch"

    print(f"LR range : [{min(lrs):.6f}, {max(lrs):.6f}]")
    print(f"image contributions (first 5): {[f'{v:.3f}' for v in hist['image'][:5]]}")
    print(f"gene  contributions (first 5): {[f'{v:.3f}' for v in hist['gene'][:5]]}")
    print("All MultimodalChaoticOptimizer self-tests PASSED.")
