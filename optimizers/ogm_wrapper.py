"""
OGM-GE wrapper for standard PyTorch optimizers.

Wraps any base optimizer with OGM-GE gradient modulation but NO chaotic LR.
Fixed/standard LR only — this isolates the OGM-GE contribution from chaotic LR.

Usage:
    from optimizers.ogm_wrapper import OGMWrapper

    base_opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    wrapped  = OGMWrapper(base_opt, model, modality_names=["image", "gene"])

    # training loop
    for batch in loader:
        out  = model(inputs)
        loss = criterion(out, labels)
        wrapped.step(loss, inputs, labels)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class OGMWrapper:
    """
    Wraps any standard PyTorch optimizer with OGM-GE gradient modulation.
    No chaotic LR — fixed/standard LR only.

    Reuses the same _compute_contributions and _modulate_gradients logic as
    MultimodalChaoticOptimizer but without the ChaoticLRScheduler.

    Args:
        optimizer      : base optimizer (e.g. SGD, Adam, Adadelta)
        model          : JointModel instance
        modality_names : list of modality name strings, e.g. ["image", "gene"]
        alpha          : gradient modulation strength (default 0.5)
        use_ge         : inject Gaussian noise for generalization (default True)
        compute_every  : apply OGM-GE every N steps (default 1)
    """

    def __init__(self,
                 optimizer,
                 model:          nn.Module,
                 modality_names: List[str],
                 alpha:          float = 0.5,
                 use_ge:         bool  = True,
                 compute_every:  int   = 1):
        self.optimizer      = optimizer
        self.model          = model
        self.modality_names = modality_names
        self.alpha          = alpha
        self.use_ge         = use_ge
        self.compute_every  = compute_every
        self._step_count    = 0

        self.contribution_history: Dict[str, List[float]] = {
            m: [] for m in modality_names
        }

    @torch.no_grad()
    def _compute_contributions(self,
                               inputs: Dict[str, torch.Tensor],
                               labels: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Approximate per-modality contribution via unimodal survival accuracy.
        Zeros out all modalities except the one being evaluated.
        Returns normalized contributions summing to 1.
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
                encoder_key = f"{modality}_encoder"
                if encoder_key in name:
                    param.grad.mul_(k)
                    if self.use_ge and modality == dominant and k < 1.0:
                        noise_std = param.grad.std() * 0.1
                        param.grad.add_(torch.randn_like(param.grad) * noise_std)

    def step(self,
             loss:   torch.Tensor,
             inputs: Optional[Dict[str, torch.Tensor]] = None,
             labels: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """
        Backward + gradient modulation + optimizer step.
        No scheduler step (fixed LR).

        Args:
            loss   : scalar loss tensor
            inputs : modality input dict (needed for OGM-GE)
            labels : label dict with key "survival" (needed for OGM-GE)
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
        # No scheduler.step() — fixed LR is the point of this wrapper

    def get_current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def get_contribution_history(self) -> Dict[str, List[float]]:
        """Returns per-modality contribution history (normalized, [0,1])."""
        return self.contribution_history
