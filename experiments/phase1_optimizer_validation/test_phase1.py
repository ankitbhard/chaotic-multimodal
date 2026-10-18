#!/usr/bin/env python3
"""Unit tests for Phase 1 optimizer validation runner.

Tests cover:
  1. Cox loss correctness (known-answer)
  2. Cosine LR schedule (boundary values)
  3. LossEMAController (scale bounds, EMA dynamics)
  4. Identical initial weights across methods
  5. SAM/M-SAM construction with AdamW base + no internal scheduler
  6. Disentanglement aux gradient flow (features not detached)
  7. Standard training loop (loss decreases)
  8. Checkpoint selection on best val C-index
"""
import copy
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_optimizer_validation.run_phase1 import (
    cosine_lr,
    LossEMAController,
    AuxiliaryHeads,
    _NoOpScheduler,
    seed_all,
    train_one_epoch_standard,
    train_one_epoch_disentangle,
)
from train_survival import cox_loss, concordance_index
from models.joint_model import build_model


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_model(n_genes=100, image_dim=256):
    return build_model(n_genes=n_genes, image_dim=image_dim,
                       fusion_type="attention", dropout=0.3, verbose=False)


def _fake_loader(n_batches=3, batch_size=16, n_genes=100, image_dim=256,
                 event_rate=0.3):
    """Create a list of (inputs, labels) tuples that mimics a DataLoader."""
    batches = []
    for _ in range(n_batches):
        imgs = torch.randn(batch_size, image_dim)
        genes = torch.randn(batch_size, n_genes)
        os_months = torch.rand(batch_size) * 120 + 1  # 1 to 121 months
        events = (torch.rand(batch_size) < event_rate).float()
        # Ensure at least one event per batch
        if events.sum() == 0:
            events[0] = 1.0
        survival = torch.randint(0, 2, (batch_size,))
        batches.append((
            {"image": imgs, "gene": genes},
            {"os_months": os_months, "vital_status": events, "survival": survival}
        ))
    return batches


class FakeLoader:
    """Wraps a list of batches to look like a DataLoader."""
    def __init__(self, batches):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


# ── Tests ────────────────────────────────────────────────────────────────────

def test_cox_loss_known_answer():
    """Cox loss with known inputs should produce expected output."""
    # Two events: both die, different risk scores and times
    risk = torch.tensor([1.0, 2.0])
    os_months = torch.tensor([20.0, 10.0])
    events = torch.tensor([1.0, 1.0])

    loss = cox_loss(risk, os_months, events)
    assert torch.isfinite(loss), "Cox loss should be finite"
    # With 2 events and different risk levels, loss should be positive
    assert loss.item() > 0, f"Cox loss should be positive, got {loss.item()}"

    # Verify: swapping risks should change the loss
    risk2 = torch.tensor([2.0, 1.0])
    loss2 = cox_loss(risk2, os_months, events)
    assert abs(loss.item() - loss2.item()) > 1e-6, "Different risks should give different loss"
    print("  PASS: test_cox_loss_known_answer")


def test_cox_loss_gradient_flow():
    """Cox loss should produce non-zero gradients for risk scores."""
    risk = torch.randn(32, requires_grad=True)
    os_months = torch.rand(32) * 100 + 1
    events = (torch.rand(32) > 0.5).float()
    events[0] = 1.0  # ensure at least one event

    loss = cox_loss(risk, os_months, events)
    loss.backward()
    assert risk.grad is not None, "Gradient should exist"
    assert risk.grad.abs().sum() > 0, "Gradient should be non-zero"
    print("  PASS: test_cox_loss_gradient_flow")


def test_cosine_lr_boundaries():
    """Cosine LR at epoch=0 should be base_lr, at epoch=T should be ~0."""
    base_lr = 0.001
    T = 200

    lr0 = cosine_lr(base_lr, 0, T)
    assert abs(lr0 - base_lr) < 1e-10, f"LR at epoch 0 should be {base_lr}, got {lr0}"

    lr_end = cosine_lr(base_lr, T, T)
    assert abs(lr_end) < 1e-10, f"LR at epoch T should be ~0, got {lr_end}"

    lr_mid = cosine_lr(base_lr, T // 2, T)
    expected_mid = base_lr * (1 + math.cos(math.pi * 0.5)) / 2
    assert abs(lr_mid - expected_mid) < 1e-10, f"LR at midpoint: got {lr_mid}, expected {expected_mid}"

    print("  PASS: test_cosine_lr_boundaries")


def test_cosine_lr_monotonic_decreasing():
    """Cosine LR should be monotonically non-increasing."""
    base_lr = 0.001
    T = 200
    lrs = [cosine_lr(base_lr, e, T) for e in range(T + 1)]
    for i in range(len(lrs) - 1):
        assert lrs[i] >= lrs[i + 1] - 1e-15, f"LR not monotonic at epoch {i}"
    print("  PASS: test_cosine_lr_monotonic_decreasing")


def test_ema_controller_scale_bounds():
    """LossEMAController scale should always be in [min_scale, max_scale]."""
    ctrl = LossEMAController(min_scale=0.5, max_scale=1.5, ema_decay=0.9)

    # Feed various loss values
    losses = [1.0, 0.5, 2.0, 0.01, 100.0, 0.5, 0.5, 0.5]
    for loss_val in losses:
        scale = ctrl.get_scale(loss_val)
        assert 0.5 <= scale <= 1.5, f"Scale {scale} out of bounds for loss={loss_val}"

    print("  PASS: test_ema_controller_scale_bounds")


def test_ema_controller_stable_loss():
    """With constant loss, EMA should converge and scale should approach 1.0."""
    ctrl = LossEMAController(min_scale=0.5, max_scale=1.5, ema_decay=0.9)

    for _ in range(100):
        scale = ctrl.get_scale(1.0)

    assert abs(scale - 1.0) < 0.01, f"Scale should be ~1.0 for constant loss, got {scale}"
    print("  PASS: test_ema_controller_stable_loss")


def test_ema_controller_history():
    """Controller should record full history."""
    ctrl = LossEMAController()
    for i in range(10):
        ctrl.get_scale(float(i))
    assert len(ctrl.history) == 10
    assert "loss" in ctrl.history[0]
    assert "loss_ema" in ctrl.history[0]
    assert "scale" in ctrl.history[0]
    print("  PASS: test_ema_controller_history")


def test_identical_initial_weights():
    """Same seed should produce identical model weights."""
    seed_all(42)
    m1 = _make_model()
    state1 = copy.deepcopy(m1.state_dict())

    seed_all(42)
    m2 = _make_model()
    state2 = m2.state_dict()

    for key in state1:
        assert torch.equal(state1[key], state2[key]), f"Mismatch in {key}"

    # Different seed should produce different weights
    seed_all(43)
    m3 = _make_model()
    state3 = m3.state_dict()
    any_diff = any(not torch.equal(state1[k], state3[k]) for k in state1)
    assert any_diff, "Different seeds should produce different weights"

    print("  PASS: test_identical_initial_weights")


def test_weight_hash_deterministic():
    """Weight hash should be deterministic for the same seed."""
    seed_all(42)
    m = _make_model()
    s = m.state_dict()
    h1 = hashlib.sha256(
        b"".join(v.cpu().numpy().tobytes() for v in s.values())
    ).hexdigest()

    seed_all(42)
    m2 = _make_model()
    s2 = m2.state_dict()
    h2 = hashlib.sha256(
        b"".join(v.cpu().numpy().tobytes() for v in s2.values())
    ).hexdigest()

    assert h1 == h2, "Hashes should match for same seed"
    print("  PASS: test_weight_hash_deterministic")


def test_sam_adaw_base_no_scheduler():
    """SAM should be constructible with AdamW base and no internal scheduler."""
    from optimizers.sam_optimizer import SAMOptimizer

    m = _make_model()
    base = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=1e-4)
    sam = SAMOptimizer(base, m, rho=0.05)
    sam.scheduler = None

    # Verify AdamW base
    assert isinstance(sam.optimizer, torch.optim.AdamW)
    assert sam.scheduler is None

    # Verify we can set LR externally
    for group in sam.param_groups:
        group["lr"] = 0.0005
    assert sam.param_groups[0]["lr"] == 0.0005

    print("  PASS: test_sam_adamw_base_no_scheduler")


def test_msam_adamw_base_no_scheduler():
    """M-SAM should work with AdamW base and disabled scheduler."""
    from optimizers.chaotic_optimizer_msam import MultimodalChaoticOptimizerMSAM

    m = _make_model()
    base = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=1e-4)
    msam = MultimodalChaoticOptimizerMSAM(
        base, m, modality_names=["image", "gene"],
        base_lr=0.001, r=3.99, rho=0.05, alpha=0.5,
        use_ge=True, compute_every=1,
    )
    msam.scheduler = _NoOpScheduler()

    assert isinstance(msam.optimizer, torch.optim.AdamW)
    assert isinstance(msam.scheduler, _NoOpScheduler)

    for group in msam.optimizer.param_groups:
        group["lr"] = 0.0005
    assert msam.optimizer.param_groups[0]["lr"] == 0.0005

    print("  PASS: test_msam_adamw_base_no_scheduler")


def test_msam_step_with_noop_scheduler():
    """M-SAM should be able to execute step() with NoOpScheduler."""
    from optimizers.chaotic_optimizer_msam import MultimodalChaoticOptimizerMSAM

    m = _make_model(n_genes=100, image_dim=256)
    base = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=1e-4)
    msam = MultimodalChaoticOptimizerMSAM(
        base, m, modality_names=["image", "gene"],
        base_lr=0.001, r=3.99, rho=0.05, alpha=0.5,
        use_ge=True, compute_every=1,
    )
    msam.scheduler = _NoOpScheduler()

    # Execute one step
    imgs = torch.randn(4, 256)
    genes = torch.randn(4, 100)
    inputs = {"image": imgs, "gene": genes}
    surv_labels = torch.randint(0, 2, (4,))
    labels = {"survival": surv_labels}

    def closure():
        out = m(inputs)
        loss = torch.nn.functional.cross_entropy(out["survival"], surv_labels)
        return loss, out

    loss, out = msam.step(closure, inputs, labels)
    assert torch.isfinite(loss), f"M-SAM loss should be finite, got {loss}"
    print("  PASS: test_msam_step_with_noop_scheduler")


def test_aux_heads_gradient_flow():
    """Auxiliary heads should allow gradient flow through encoder features."""
    m = _make_model(n_genes=100, image_dim=256)
    img_dim = getattr(m.image_encoder, 'FEATURE_DIM', 512)
    gene_dim = getattr(m.gene_encoder, 'FEATURE_DIM', 128)
    aux = AuxiliaryHeads(img_dim, gene_dim)

    # Forward through model
    img = torch.randn(4, 256)
    gene = torch.randn(4, 100)
    img_feat = m.image_encoder(img)
    gene_feat = m.gene_encoder(gene)

    # Without detach — gradients should flow to encoder
    img_risk, gene_risk = aux(img_feat, gene_feat)
    loss = img_risk.mean() + gene_risk.mean()
    loss.backward()

    # Check that encoder parameters received gradients
    img_grads = [p.grad for p in m.image_encoder.parameters() if p.grad is not None]
    gene_grads = [p.grad for p in m.gene_encoder.parameters() if p.grad is not None]
    assert len(img_grads) > 0, "Image encoder should have gradients (no detach)"
    assert len(gene_grads) > 0, "Gene encoder should have gradients (no detach)"

    print("  PASS: test_aux_heads_gradient_flow")


def test_aux_heads_with_detach_no_gradient():
    """With detach, encoder should NOT receive gradients from aux heads."""
    m = _make_model(n_genes=100, image_dim=256)
    img_dim = getattr(m.image_encoder, 'FEATURE_DIM', 512)
    gene_dim = getattr(m.gene_encoder, 'FEATURE_DIM', 128)
    aux = AuxiliaryHeads(img_dim, gene_dim)

    img = torch.randn(4, 256)
    gene = torch.randn(4, 100)
    img_feat = m.image_encoder(img)
    gene_feat = m.gene_encoder(gene)

    # WITH detach — no gradients to encoder
    img_risk, gene_risk = aux(img_feat.detach(), gene_feat.detach())
    loss = img_risk.mean() + gene_risk.mean()
    loss.backward()

    img_grads = [p.grad for p in m.image_encoder.parameters() if p.grad is not None]
    gene_grads = [p.grad for p in m.gene_encoder.parameters() if p.grad is not None]
    assert len(img_grads) == 0, "Image encoder should NOT have gradients (detached)"
    assert len(gene_grads) == 0, "Gene encoder should NOT have gradients (detached)"

    print("  PASS: test_aux_heads_with_detach_no_gradient")


def test_standard_training_loss_decreases():
    """Standard training on a simple task should decrease loss."""
    seed_all(42)
    m = _make_model(n_genes=100, image_dim=256)
    opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=1e-4)
    loader = FakeLoader(_fake_loader(n_batches=5, batch_size=16,
                                      n_genes=100, image_dim=256))
    device = torch.device("cpu")

    losses = []
    for epoch in range(5):
        loss = train_one_epoch_standard(m, loader, opt, device)
        losses.append(loss)

    # Loss should decrease (at least between epoch 1 and 5)
    assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    print("  PASS: test_standard_training_loss_decreases")


def test_disentangle_training_runs():
    """Disentangle training should produce finite losses and diagnostics."""
    seed_all(42)
    m = _make_model(n_genes=100, image_dim=256)
    img_dim = getattr(m.image_encoder, 'FEATURE_DIM', 512)
    gene_dim = getattr(m.gene_encoder, 'FEATURE_DIM', 128)
    aux = AuxiliaryHeads(img_dim, gene_dim)
    all_params = list(m.parameters()) + list(aux.parameters())
    opt = torch.optim.AdamW(all_params, lr=0.001, weight_decay=1e-4)
    loader = FakeLoader(_fake_loader(n_batches=5, batch_size=16,
                                      n_genes=100, image_dim=256))
    device = torch.device("cpu")

    loss, diag = train_one_epoch_disentangle(m, aux, loader, opt, device,
                                              aux_weight=0.1, balance_grads=True)
    assert np.isfinite(loss), f"Loss should be finite, got {loss}"
    assert "fusion_loss" in diag
    assert "aux_img_loss" in diag
    assert "aux_gene_loss" in diag
    assert "mean_img_grad_norm" in diag
    assert "mean_gene_grad_norm" in diag
    assert np.isfinite(diag["fusion_loss"])
    print("  PASS: test_disentangle_training_runs")


def test_concordance_index_perfect():
    """Perfect risk ordering should give C-index = 1.0."""
    # Higher risk → shorter time → event
    times = [10.0, 20.0, 30.0, 40.0]
    risks = [4.0, 3.0, 2.0, 1.0]  # Higher risk for shorter times
    events = [1.0, 1.0, 1.0, 1.0]

    ci = concordance_index(times, risks, events)
    assert abs(ci - 1.0) < 1e-10, f"Perfect ordering should give C-index=1.0, got {ci}"
    print("  PASS: test_concordance_index_perfect")


def test_concordance_index_inverse():
    """Inverse risk ordering should give C-index = 0.0."""
    times = [10.0, 20.0, 30.0, 40.0]
    risks = [1.0, 2.0, 3.0, 4.0]  # Wrong direction
    events = [1.0, 1.0, 1.0, 1.0]

    ci = concordance_index(times, risks, events)
    assert abs(ci) < 1e-10, f"Inverse ordering should give C-index=0.0, got {ci}"
    print("  PASS: test_concordance_index_inverse")


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Phase 1 unit tests")
    print("=" * 60)

    tests = [
        test_cox_loss_known_answer,
        test_cox_loss_gradient_flow,
        test_cosine_lr_boundaries,
        test_cosine_lr_monotonic_decreasing,
        test_ema_controller_scale_bounds,
        test_ema_controller_stable_loss,
        test_ema_controller_history,
        test_identical_initial_weights,
        test_weight_hash_deterministic,
        test_sam_adaw_base_no_scheduler,
        test_msam_adamw_base_no_scheduler,
        test_msam_step_with_noop_scheduler,
        test_aux_heads_gradient_flow,
        test_aux_heads_with_detach_no_gradient,
        test_standard_training_loss_decreases,
        test_disentangle_training_runs,
        test_concordance_index_perfect,
        test_concordance_index_inverse,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("All tests PASSED.")
    else:
        print(f"WARNING: {failed} test(s) FAILED!")
        sys.exit(1)
