"""Unit tests for SGC-AdamW optimizer."""
import copy
import math
import pytest
import torch
import torch.nn as nn
from experiments.sgc_adamw.sgc_optimizer import SGCAdamW, MODES


# ── Helpers ──────────────────────────────────────────────────────────────────

def _simple_model(in_dim=4, out_dim=2):
    torch.manual_seed(42)
    return nn.Linear(in_dim, out_dim, bias=True)


def _det_grads(model, seed=0):
    """Deterministic forward/backward on fixed data."""
    torch.manual_seed(seed)
    x = torch.randn(8, model.in_features)
    y = torch.randint(0, model.out_features, (8,))
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    return loss


def _det_forward(model, seed=0):
    """Deterministic forward only (no backward) — for curvature probe closures."""
    torch.manual_seed(seed)
    x = torch.randn(8, model.in_features)
    y = torch.randint(0, model.out_features, (8,))
    return nn.functional.cross_entropy(model(x), y)


# ── 1. Disabled mode matches torch.optim.AdamW exactly ──────────────────────

class TestDisabledMatchesAdamW:
    """Disabled SGCAdamW must produce identical updates to torch.optim.AdamW."""

    @pytest.mark.parametrize("steps", [1, 5, 20])
    def test_exact_match(self, steps):
        torch.manual_seed(0)
        m1 = _simple_model(); m2 = copy.deepcopy(m1)
        opt1 = torch.optim.AdamW(m1.parameters(), lr=1e-3, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=1e-4)
        opt2 = SGCAdamW(m2.parameters(), lr=1e-3, betas=(0.9, 0.999),
                        eps=1e-8, weight_decay=1e-4, mode="disabled")
        for s in range(steps):
            opt1.zero_grad(); opt2.zero_grad()
            _det_grads(m1, seed=s); _det_grads(m2, seed=s)
            opt1.step(); opt2.step()
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            torch.testing.assert_close(p1, p2, atol=1e-6, rtol=1e-5)

    def test_lr_schedule(self):
        """Disabled mode with varying LR matches AdamW."""
        torch.manual_seed(0)
        m1 = _simple_model(); m2 = copy.deepcopy(m1)
        opt1 = torch.optim.AdamW(m1.parameters(), lr=1e-3, weight_decay=1e-4)
        opt2 = SGCAdamW(m2.parameters(), lr=1e-3, weight_decay=1e-4,
                        mode="disabled")
        for s in range(10):
            lr = 1e-3 * 0.5 * (1 + math.cos(math.pi * s / 10))
            for pg in opt1.param_groups: pg["lr"] = lr
            for pg in opt2.param_groups: pg["lr"] = lr
            opt1.zero_grad(); opt2.zero_grad()
            _det_grads(m1, seed=s); _det_grads(m2, seed=s)
            opt1.step(); opt2.step()
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            torch.testing.assert_close(p1, p2, atol=1e-6, rtol=1e-5)


# ── 2. Variable-beta2 bias correction ───────────────────────────────────────

class TestBiasCorrection:
    """Verify cumulative-product beta2 correction by hand computation."""

    def test_two_step_hand_computed(self):
        """Hand-compute v_hat after 2 steps with known beta2 values."""
        torch.manual_seed(0)
        p = nn.Parameter(torch.tensor([1.0, 2.0]))
        opt = SGCAdamW([p], lr=0.01, betas=(0.9, 0.999), mode="drift",
                       beta2_min=0.95, drift_sensitivity=0.0)
        # With sensitivity=0, drift signal is 0, so beta2 = beta2_min +
        # (beta2_max - beta2_min) * exp(0) = beta2_max = 0.999 always
        g1 = torch.tensor([0.5, -0.3])
        p.grad = g1.clone()
        opt.step()
        s = opt.state[p]
        assert s["step"] == 1
        # First step: beta2 = 0.999 (no drift on step 1)
        assert abs(s["last_beta2"] - 0.999) < 1e-6
        assert abs(s["beta2_product"] - 0.999) < 1e-6

        g2 = torch.tensor([0.1, 0.4])
        p.grad = g2.clone()
        opt.step()
        # Second step: drift sensitivity=0 → combined=0 → beta2=0.999
        assert s["step"] == 2
        assert abs(s["beta2_product"] - 0.999 ** 2) < 1e-6

    def test_variable_beta2_product(self):
        """Force different beta2 values and verify product accumulation."""
        torch.manual_seed(0)
        p = nn.Parameter(torch.tensor([1.0]))
        # Use high drift_sensitivity to force beta2 toward beta2_min
        opt = SGCAdamW([p], lr=0.01, betas=(0.9, 0.999), mode="drift",
                       beta2_min=0.95, drift_sensitivity=100.0,
                       drift_ema_decay=0.0)  # no EMA smoothing → raw drift
        # Step 1: beta2 = 0.999 (first step always max)
        p.grad = torch.tensor([1.0])
        opt.step()
        bp1 = opt.state[p]["beta2_product"]
        assert abs(bp1 - 0.999) < 1e-6

        # Step 2: high drift sensitivity → beta2 ≈ beta2_min
        p.grad = torch.tensor([10.0])  # very different from v
        opt.step()
        b2_step2 = opt.state[p]["last_beta2"]
        bp2 = opt.state[p]["beta2_product"]
        assert b2_step2 < 0.97  # should be near beta2_min
        assert abs(bp2 - bp1 * b2_step2) < 1e-6


# ── 3. Mode validation ──────────────────────────────────────────────────────

class TestModes:
    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            SGCAdamW([nn.Parameter(torch.zeros(1))], mode="bad")

    @pytest.mark.parametrize("mode", MODES)
    def test_all_modes_run(self, mode):
        m = _simple_model()
        opt = SGCAdamW(m.parameters(), mode=mode)
        for s in range(5):
            opt.zero_grad()
            _det_grads(m, seed=s)
            opt.step()
        # No crash = pass; check state exists
        for p in m.parameters():
            assert opt.state[p]["step"] == 5


# ── 4. Parameter groups ─────────────────────────────────────────────────────

class TestParamGroups:
    def test_independent_groups(self):
        m = _simple_model()
        opt = SGCAdamW([
            {"params": [m.weight], "lr": 1e-2, "name": "weight"},
            {"params": [m.bias], "lr": 1e-4, "name": "bias"},
        ], mode="drift_alignment")
        for s in range(5):
            opt.zero_grad()
            _det_grads(m, seed=s)
            opt.step()
        # Both groups should have state
        assert opt.state[m.weight]["step"] == 5
        assert opt.state[m.bias]["step"] == 5
        tel = opt.get_telemetry()
        assert "weight" in tel
        assert "bias" in tel


# ── 5. Missing / None gradients ─────────────────────────────────────────────

class TestMissingGradients:
    def test_none_grad_skipped(self):
        m = _simple_model()
        opt = SGCAdamW(m.parameters(), mode="drift")
        # Only set grad on weight, not bias
        m.weight.grad = torch.randn_like(m.weight)
        m.bias.grad = None
        opt.step()
        assert m.weight in opt.state and opt.state[m.weight]["step"] == 1
        assert m.bias not in opt.state or opt.state[m.bias].get("step", 0) == 0


# ── 6. Non-finite gradient handling ─────────────────────────────────────────

class TestNonFiniteGradients:
    def test_nan_grad_skipped(self):
        p = nn.Parameter(torch.tensor([1.0, 2.0]))
        opt = SGCAdamW([p], mode="drift")
        p_before = p.data.clone()
        p.grad = torch.tensor([float("nan"), 1.0])
        opt.step()
        # Parameter should not change (NaN grad skipped)
        torch.testing.assert_close(p.data, p_before)

    def test_inf_grad_skipped(self):
        p = nn.Parameter(torch.tensor([1.0, 2.0]))
        opt = SGCAdamW([p], mode="drift")
        p_before = p.data.clone()
        p.grad = torch.tensor([float("inf"), 1.0])
        opt.step()
        torch.testing.assert_close(p.data, p_before)


# ── 7. Sparse gradient rejection ────────────────────────────────────────────

class TestSparseGradients:
    def test_sparse_raises(self):
        p = nn.Parameter(torch.zeros(4, 4))
        opt = SGCAdamW([p], mode="drift")
        idx = torch.tensor([[0, 1], [2, 3]])
        vals = torch.tensor([1.0, 2.0])
        p.grad = torch.sparse_coo_tensor(idx, vals, (4, 4))
        with pytest.raises(RuntimeError, match="sparse"):
            opt.step()


# ── 8. State dict save/load ─────────────────────────────────────────────────

class TestStateDict:
    @pytest.mark.parametrize("mode", ["drift", "drift_alignment",
                                       "drift_alignment_curvature"])
    def test_save_load_reproduces_next_step(self, mode):
        torch.manual_seed(0)
        m1 = _simple_model()
        opt1 = SGCAdamW(m1.parameters(), mode=mode, lr=1e-3)
        # Run 5 steps
        for s in range(5):
            opt1.zero_grad()
            _det_grads(m1, seed=s)
            opt1.step()
        # Save state
        sd = copy.deepcopy(opt1.state_dict())
        # One more step on original
        opt1.zero_grad()
        _det_grads(m1, seed=5)
        opt1.step()
        params_after = [p.data.clone() for p in m1.parameters()]

        # Reload into fresh model
        torch.manual_seed(0)
        m2 = _simple_model()
        opt2 = SGCAdamW(m2.parameters(), mode=mode, lr=1e-3)
        # Run same 5 steps to populate param_groups mapping
        for s in range(5):
            opt2.zero_grad()
            _det_grads(m2, seed=s)
            opt2.step()
        opt2.load_state_dict(sd)
        # Same 6th step
        opt2.zero_grad()
        _det_grads(m2, seed=5)
        opt2.step()
        for p1, p2 in zip(params_after, m2.parameters()):
            torch.testing.assert_close(p1, p2.data, atol=1e-6, rtol=1e-5)


# ── 9. Curvature probe ──────────────────────────────────────────────────────

class TestCurvatureProbe:
    def test_restores_params_and_grads(self):
        m = _simple_model()
        opt = SGCAdamW(m.parameters(), mode="drift_alignment_curvature")
        # Run 2 steps so prev_param diverges from current
        for s in range(2):
            opt.zero_grad()
            _det_grads(m, seed=s)
            opt.step()
        # Now do forward/backward on a batch
        opt.zero_grad()
        _det_grads(m, seed=99)
        params_before = [p.data.clone() for p in m.parameters()]
        grads_before = [p.grad.clone() for p in m.parameters()]
        # Probe
        info = opt.curvature_probe(lambda: _det_forward(m, seed=99))
        # Check restoration
        for p, pb in zip(m.parameters(), params_before):
            torch.testing.assert_close(p.data, pb)
        for p, gb in zip(m.parameters(), grads_before):
            torch.testing.assert_close(p.grad, gb)
        assert info["n_params"] > 0
        assert info["elapsed_s"] >= 0

    def test_curvature_ema_updated(self):
        m = _simple_model()
        opt = SGCAdamW(m.parameters(), mode="drift_alignment_curvature")
        for s in range(3):
            opt.zero_grad()
            _det_grads(m, seed=s)
            opt.step()
        # Check curvature_ema is still 0 (no probe called)
        for p in m.parameters():
            assert opt.state[p]["curvature_ema"] == 0.0
        # Now probe
        opt.zero_grad()
        _det_grads(m, seed=10)
        opt.curvature_probe(lambda: _det_forward(m, seed=10))
        # curvature_ema should be nonzero now
        has_nonzero = any(opt.state[p]["curvature_ema"] > 0
                         for p in m.parameters())
        assert has_nonzero

    def test_no_hidden_optimizer_state_change(self):
        m = _simple_model()
        opt = SGCAdamW(m.parameters(), mode="drift_alignment_curvature")
        for s in range(3):
            opt.zero_grad(); _det_grads(m, seed=s); opt.step()
        # Save optimizer state (except curvature_ema which probe updates)
        states_before = {}
        for p in m.parameters():
            s = opt.state[p]
            states_before[p] = {
                "step": s["step"],
                "exp_avg": s["exp_avg"].clone(),
                "exp_avg_sq": s["exp_avg_sq"].clone(),
                "beta2_product": s["beta2_product"],
            }
        opt.zero_grad(); _det_grads(m, seed=10)
        opt.curvature_probe(lambda: _det_forward(m, seed=10))
        # Check that probe didn't change m, v, step, beta2_product
        for p in m.parameters():
            s = opt.state[p]
            b = states_before[p]
            assert s["step"] == b["step"]
            assert s["beta2_product"] == b["beta2_product"]
            torch.testing.assert_close(s["exp_avg"], b["exp_avg"])
            torch.testing.assert_close(s["exp_avg_sq"], b["exp_avg_sq"])

    def test_noop_for_non_curvature_modes(self):
        m = _simple_model()
        opt = SGCAdamW(m.parameters(), mode="drift_alignment")
        opt.zero_grad(); _det_grads(m, seed=0); opt.step()
        opt.zero_grad(); _det_grads(m, seed=1)
        info = opt.curvature_probe(lambda: _det_forward(m, seed=1))
        assert info["n_params"] == 0


# ── 10. Telemetry ───────────────────────────────────────────────────────────

class TestTelemetry:
    def test_telemetry_keys(self):
        m = _simple_model()
        opt = SGCAdamW(m.parameters(), mode="drift_alignment")
        for s in range(3):
            opt.zero_grad(); _det_grads(m, seed=s); opt.step()
        tel = opt.get_telemetry()
        assert len(tel) >= 1
        for name, info in tel.items():
            assert "beta2_mean" in info
            assert "drift_ema_mean" in info
            assert "alignment_ema_mean" in info
            assert "global_step" in info
            assert info["global_step"] == 3

    def test_empty_before_step(self):
        m = _simple_model()
        opt = SGCAdamW(m.parameters(), mode="drift")
        tel = opt.get_telemetry()
        assert len(tel) == 0  # no state populated yet


# ── 11. Hyperparameter validation ────────────────────────────────────────────

class TestValidation:
    def test_invalid_beta1(self):
        with pytest.raises(ValueError, match="beta1"):
            SGCAdamW([nn.Parameter(torch.zeros(1))], betas=(1.0, 0.999))

    def test_invalid_beta2_bounds(self):
        with pytest.raises(ValueError, match="beta2"):
            SGCAdamW([nn.Parameter(torch.zeros(1))], betas=(0.9, 0.999),
                     beta2_min=0.9999)

    def test_invalid_lr(self):
        with pytest.raises(ValueError, match="lr"):
            SGCAdamW([nn.Parameter(torch.zeros(1))], lr=-1)

    def test_invalid_wd(self):
        with pytest.raises(ValueError, match="weight_decay"):
            SGCAdamW([nn.Parameter(torch.zeros(1))], weight_decay=-0.1)


# ── 12. Smoke test: 10 steps ────────────────────────────────────────────────

class TestSmoke:
    @pytest.mark.parametrize("mode", MODES)
    def test_10_steps_converge(self, mode):
        """10-step training on a tiny problem should decrease loss."""
        torch.manual_seed(42)
        m = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        opt = SGCAdamW(m.parameters(), lr=1e-2, mode=mode)
        x = torch.randn(32, 4); y = torch.randint(0, 2, (32,))
        losses = []
        for _ in range(10):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(m(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses}"


# ── 13. CUDA test (if available) ────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
class TestCUDA:
    def test_cuda_step(self):
        m = _simple_model().cuda()
        opt = SGCAdamW(m.parameters(), mode="drift_alignment")
        x = torch.randn(8, 4, device="cuda")
        y = torch.randint(0, 2, (8,), device="cuda")
        for _ in range(3):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(m(x), y)
            loss.backward()
            opt.step()
        assert all(p.is_cuda for p in m.parameters())


# ── 14. Drift-only mode matches DynamicsAwareAdamW ──────────────────────────

class TestDriftOnlyMatchesDynamicsAware:
    def test_numerical_match(self):
        """Drift-only SGCAdamW should give same beta2 trajectory as
        DynamicsAwareAdamW with matching hyperparameters."""
        from experiments.dynamics_aware_ec2.dynamics_optimizer import (
            DynamicsAwareAdamW)
        torch.manual_seed(0)
        m1 = _simple_model(); m2 = copy.deepcopy(m1)
        opt1 = DynamicsAwareAdamW(m1.parameters(), lr=1e-3,
                                  betas=(0.9, 0.999), weight_decay=1e-4,
                                  beta2_min=0.95, sensitivity=3.0,
                                  change_ema=0.9)
        opt2 = SGCAdamW(m2.parameters(), lr=1e-3, betas=(0.9, 0.999),
                        weight_decay=1e-4, mode="drift", beta2_min=0.95,
                        drift_sensitivity=3.0, drift_ema_decay=0.9)
        for s in range(10):
            opt1.zero_grad(); opt2.zero_grad()
            _det_grads(m1, seed=s); _det_grads(m2, seed=s)
            opt1.step(); opt2.step()
        # Check params are close
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            torch.testing.assert_close(p1, p2, atol=1e-5, rtol=1e-4)
        # Check beta2 values match
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            b1 = opt1.state[p1]["last_beta2"]
            b2 = opt2.state[p2]["last_beta2"]
            assert abs(b1 - b2) < 1e-6, f"beta2 mismatch: {b1} vs {b2}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
