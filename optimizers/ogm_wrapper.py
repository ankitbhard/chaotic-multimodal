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
    Wraps any standard PyTorch optimizer with adaptive OGM-GE gradient modulation.
    No chaotic LR — fixed/standard LR only.

    Adaptive gating (new): suppression only fires when BOTH conditions hold:
      1. Modality imbalance > imbalance_threshold (default 0.0)
      2. Weaker modality's contribution is NOT already trending upward

    When imbalance_threshold == 0.0, both gates are disabled and OGM-GE is
    always-on (replicating the original Pan et al. behaviour).

    This prevents OGM-GE from interfering when modalities are naturally balanced
    or when the weaker modality is self-correcting.

    Args:
        optimizer           : base optimizer (e.g. SGD, Adam, Adadelta)
        model               : JointModel instance
        modality_names      : list of modality name strings, e.g. ["image", "gene"]
        alpha               : gradient modulation strength (default 0.5)
        use_ge              : inject Gaussian noise for generalization (default True)
        compute_every       : apply OGM-GE every N steps (default 1)
        imbalance_threshold : minimum contribution gap to trigger suppression (default 0.0)
                              Set to 0.0 to replicate original always-on behaviour.
        trend_window        : look-back window for weaker-modality improvement gate (default 10)
                              Only active when imbalance_threshold > 0.
    """

    def __init__(self,
                 optimizer,
                 model:               nn.Module,
                 modality_names:      List[str],
                 alpha:               float = 0.5,
                 use_ge:              bool  = True,
                 compute_every:       int   = 1,
                 imbalance_threshold: float = 0.0,
                 trend_window:        int   = 10):
        self.optimizer            = optimizer
        self.model                = model
        self.modality_names       = modality_names
        self.alpha                = alpha
        self.use_ge               = use_ge
        self.compute_every        = compute_every
        self.imbalance_threshold  = imbalance_threshold
        self.trend_window         = trend_window
        self._step_count          = 0
        self._ogm_applied         = 0
        self._ogm_skipped         = 0

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
        Adaptive OGM-GE gradient suppression.

        Gate 1 — imbalance threshold:
            If dominant − weaker ≤ imbalance_threshold → skip (modalities balanced).

        Gate 2 — trend gate:
            If the weaker modality's contribution has been trending upward over the
            last trend_window steps → skip (it's self-correcting, no need to intervene).

        When both gates pass, apply proportional suppression scaled to excess imbalance:
            effective_ratio = (imbalance − θ) / (1 − θ)
            k = 1 − tanh(α · effective_ratio)
        """
        dominant = max(contributions, key=contributions.get)
        weaker   = min(contributions, key=contributions.get)
        imbalance = contributions[dominant] - contributions[weaker]

        # Gate 1: imbalance threshold
        if imbalance <= self.imbalance_threshold:
            self._ogm_skipped += 1
            return

        # Gate 2: trend gate — is the weaker modality already improving?
        # Only active when imbalance_threshold > 0 (adaptive mode).
        # When threshold == 0 this is always-on OGM-GE, no trend gating.
        if self.imbalance_threshold > 0:
            weaker_history = self.contribution_history.get(weaker, [])
            if len(weaker_history) >= 2 * self.trend_window:
                recent = sum(weaker_history[-self.trend_window:]) / self.trend_window
                older  = sum(weaker_history[-2 * self.trend_window:-self.trend_window]) / self.trend_window
                if recent > older:
                    self._ogm_skipped += 1
                    return

        # Both gates passed — apply proportional suppression
        effective_ratio = (imbalance - self.imbalance_threshold) / (1.0 - self.imbalance_threshold)
        self._ogm_applied += 1

        for modality in contributions:
            if modality == dominant:
                k = 1.0 - float(torch.tanh(torch.tensor(self.alpha * effective_ratio)))
            else:
                k = 1.0

            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                if f"{modality}_encoder" in name:
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

    def get_ogm_stats(self) -> Dict[str, int]:
        """Returns count of steps where OGM-GE was applied vs skipped."""
        return {"applied": self._ogm_applied, "skipped": self._ogm_skipped}
