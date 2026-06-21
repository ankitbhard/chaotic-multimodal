"""
Chaotic Learning Rate Scheduler.

Implements logistic-map-based LR modulation with optional cosine decay envelope:

  Standard (T_max=None):
      x_{t+1} = r · x_t · (1 − x_t),  r = 3.99
      lr_t    = base_lr · x_t

  With cosine decay envelope (T_max set):
      lr_t    = base_lr · decay(t) · x_t
      decay(t) = 0.5 · (1 + cos(π · t / T_max))   ∈ [0, 1]

  The envelope preserves chaotic exploration early in training but forces LR
  toward zero by epoch T_max, preventing the late-stage loss divergence
  observed without decay (val_loss grows 2–3× after convergence epoch ~30).

Also provides CosineAnnealingScheduler as a direct baseline comparison.

Usage:
    from optimizers.chaotic_lr_scheduler import ChaoticLRScheduler

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler = ChaoticLRScheduler(optimizer, base_lr=1e-3, T_max=200)

    for epoch in range(n_epochs):
        train(...)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
"""

import math
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler


class ChaoticLRScheduler(_LRScheduler):
    """
    Logistic-map chaotic LR scheduler with optional cosine decay envelope.

    Args:
        optimizer : wrapped optimizer
        base_lr   : peak learning rate (default: 1e-3)
        r         : logistic map parameter — 3.99 = fully chaotic regime
        x0        : initial chaotic state in (0, 1) (default: 0.5)
        T_max     : total epochs for cosine decay envelope; None = no decay
        last_epoch: index of last epoch (default: -1)
    """

    def __init__(self,
                 optimizer,
                 base_lr:    float = 1e-3,
                 r:          float = 3.99,
                 x0:         float = 0.5,
                 T_max:      int   = None,
                 last_epoch: int   = -1):
        self.base_lr = base_lr
        self.r       = r
        self.x_t     = x0
        self.T_max   = T_max
        super().__init__(optimizer, last_epoch)

    def _cosine_decay(self) -> float:
        """Cosine envelope ∈ [0, 1], 1.0 at epoch 0 → 0.0 at epoch T_max."""
        if self.T_max is None:
            return 1.0
        t = max(0, self.last_epoch)
        return 0.5 * (1.0 + math.cos(math.pi * t / self.T_max))

    def get_lr(self):
        self.x_t = self.r * self.x_t * (1.0 - self.x_t)
        decay = self._cosine_decay()
        # Use each param group's own base_lr so per-modality LR ratios are preserved
        return [base_lr * decay * self.x_t for base_lr in self.base_lrs]

    def get_chaotic_state(self) -> float:
        """Return current chaotic state x_t ∈ (0, 1)."""
        return self.x_t


class CosineAnnealingScheduler(_LRScheduler):
    """
    Standard cosine annealing scheduler (baseline comparison).

    Args:
        optimizer : wrapped optimizer
        T_max     : total number of epochs (period)
        eta_min   : minimum learning rate (default: 0)
        last_epoch: index of last epoch (default: -1)
    """

    def __init__(self,
                 optimizer,
                 T_max:      int,
                 eta_min:    float = 0.0,
                 last_epoch: int   = -1):
        self.T_max   = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1.0 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2.0
            for base_lr in self.base_lrs
        ]


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    dummy = torch.nn.Linear(4, 2)
    opt   = torch.optim.SGD(dummy.parameters(), lr=1e-3)

    # Chaotic (no decay): check LR stays in (0, base_lr)
    sched = ChaoticLRScheduler(opt, base_lr=1e-3)
    lrs   = []
    for _ in range(200):
        sched.step()
        lrs.append(sched.get_last_lr()[0])
    assert all(0 < lr <= 1e-3 for lr in lrs), "Chaotic LR out of (0, base_lr]"
    print(f"ChaoticLRScheduler (no decay) : min={min(lrs):.6f}  max={max(lrs):.6f}  ✓")

    # Chaotic + cosine decay: LR at end must be near 0
    opt3  = torch.optim.SGD(dummy.parameters(), lr=1e-3)
    sched3 = ChaoticLRScheduler(opt3, base_lr=1e-3, T_max=200)
    lrs3  = []
    for _ in range(200):
        sched3.step()
        lrs3.append(sched3.get_last_lr()[0])
    assert max(lrs3[:10]) > max(lrs3[-10:]), "Decay envelope not reducing LR over time"
    assert lrs3[-1] < 1e-5, f"Final LR should be ~0, got {lrs3[-1]:.6f}"
    print(f"ChaoticLRScheduler (T_max=200): ep1={lrs3[0]:.6f}  ep100={lrs3[99]:.6f}  ep200={lrs3[-1]:.6f}  ✓")

    # Cosine: check monotone decrease for first half
    opt2   = torch.optim.SGD(dummy.parameters(), lr=1e-3)
    sched2 = CosineAnnealingScheduler(opt2, T_max=200, eta_min=0)
    lrs2   = []
    for _ in range(200):
        sched2.step()
        lrs2.append(sched2.get_last_lr()[0])

    assert lrs2[0] > lrs2[99] > lrs2[-1], "Cosine LR not decreasing"
    print(f"CosineAnnealingScheduler : start={lrs2[0]:.6f}  end={lrs2[-1]:.6f}  ✓")

    print("All scheduler self-tests PASSED.")
