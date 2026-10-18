"""
Unit tests for the CVPR reliability pilot.

Tests:
  1. Cox loss correctness and gradient flow
  2. Cosine LR schedule boundaries
  3. PCRF model forward pass shapes
  4. Reliability network output shapes
  5. Aux Cox head output shapes
  6. Modality dropout behavior
  7. Modality mask handling
  8. Perturbation sensitivity computation
  9. Reliability loss computation
  10. Combined loss for each method
  11. Identical initial weights across methods
  12. JointModelWithAux forward
  13. JointModelWithDropout forward
  14. Evaluate function
  15. Learned gating vs PCRF capacity match
  16. Perturbation calibrator running stats
  17. Reliability collapse detection
  18. Missing modality evaluation

Usage:
    python -m pytest experiments/cvpr_reliability_pilot/test_pilot.py -v
    # or directly:
    python experiments/cvpr_reliability_pilot/test_pilot.py
"""

import sys, os
import unittest
import torch
import torch.nn as nn
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)


class TestCoxLoss(unittest.TestCase):
    def test_positive_loss(self):
        from experiments.cvpr_reliability_pilot.losses import cox_loss
        risk = torch.tensor([2.0, 1.0, 0.5], requires_grad=True)
        times = torch.tensor([30.0, 20.0, 10.0])
        events = torch.tensor([1.0, 1.0, 0.0])
        loss = cox_loss(risk, times, events)
        self.assertFalse(torch.isnan(loss))
        loss.backward()
        self.assertIsNotNone(risk.grad)

    def test_zero_events(self):
        from experiments.cvpr_reliability_pilot.losses import cox_loss
        risk = torch.tensor([1.0, 2.0], requires_grad=True)
        times = torch.tensor([10.0, 20.0])
        events = torch.tensor([0.0, 0.0])
        loss = cox_loss(risk, times, events)
        self.assertEqual(loss.item(), 0.0)


class TestCosineLR(unittest.TestCase):
    def test_boundaries(self):
        from experiments.cvpr_reliability_pilot.train import cosine_lr
        base = 0.001
        T = 100
        self.assertAlmostEqual(cosine_lr(base, 0, T), base, places=6)
        self.assertAlmostEqual(cosine_lr(base, T, T), 0.0, places=6)
        mid = cosine_lr(base, T // 2, T)
        self.assertGreater(mid, 0)
        self.assertLess(mid, base)


class TestPCRFModel(unittest.TestCase):
    def setUp(self):
        from experiments.cvpr_reliability_pilot.models import PCRFModel
        self.model = PCRFModel(
            n_genes=100, image_dim=64, dropout=0.0,
            use_aux_loss=True, use_reliability_loss=True,
        )
        self.B = 4
        self.inputs = {
            "image": torch.randn(self.B, 64),
            "gene": torch.randn(self.B, 100),
        }

    def test_forward_shapes(self):
        outputs = self.model(self.inputs)
        self.assertEqual(outputs["survival"].shape, (self.B, 2))
        self.assertEqual(outputs["reliability_weights"].shape, (self.B, 2))
        self.assertEqual(outputs["reliability_logits"].shape, (self.B, 2))
        self.assertEqual(outputs["aux_img_risk"].shape, (self.B,))
        self.assertEqual(outputs["aux_gene_risk"].shape, (self.B,))

    def test_reliability_weights_sum_to_one(self):
        outputs = self.model(self.inputs)
        w = outputs["reliability_weights"]
        sums = w.sum(dim=1)
        for s in sums:
            self.assertAlmostEqual(s.item(), 1.0, places=5)

    def test_modality_mask(self):
        mask = torch.tensor([[1, 0]] * self.B, dtype=torch.float32)
        outputs = self.model(self.inputs, modality_mask=mask)
        w = outputs["reliability_weights"]
        # With gene masked out, all weight should go to image
        for i in range(self.B):
            self.assertGreater(w[i, 0].item(), 0.99)

    def test_gradient_flow(self):
        outputs = self.model(self.inputs)
        loss = outputs["survival"][:, 1].mean()
        loss.backward()
        # Check gradients exist for encoders
        for p in self.model.image_encoder.parameters():
            self.assertIsNotNone(p.grad)
        for p in self.model.gene_encoder.parameters():
            self.assertIsNotNone(p.grad)

    def test_encoder_features_cached(self):
        self.model(self.inputs)
        img_f, gene_f = self.model.get_encoder_features()
        self.assertEqual(img_f.shape[0], self.B)
        self.assertEqual(gene_f.shape[0], self.B)


class TestJointModelWithAux(unittest.TestCase):
    def test_forward(self):
        from experiments.cvpr_reliability_pilot.models import JointModelWithAux
        model = JointModelWithAux(n_genes=100, image_dim=64)
        B = 4
        inputs = {"image": torch.randn(B, 64), "gene": torch.randn(B, 100)}
        outputs = model(inputs)
        self.assertEqual(outputs["survival"].shape, (B, 2))
        self.assertEqual(outputs["aux_img_risk"].shape, (B,))
        self.assertEqual(outputs["aux_gene_risk"].shape, (B,))


class TestJointModelWithDropout(unittest.TestCase):
    def test_forward_train(self):
        from experiments.cvpr_reliability_pilot.models import JointModelWithDropout
        model = JointModelWithDropout(n_genes=100, image_dim=64, p_drop=0.5)
        model.train()
        B = 8
        inputs = {"image": torch.randn(B, 64), "gene": torch.randn(B, 100)}
        outputs = model(inputs)
        self.assertEqual(outputs["survival"].shape, (B, 2))

    def test_forward_eval_no_dropout(self):
        from experiments.cvpr_reliability_pilot.models import JointModelWithDropout
        model = JointModelWithDropout(n_genes=100, image_dim=64, p_drop=0.5)
        model.eval()
        B = 4
        inputs = {"image": torch.randn(B, 64), "gene": torch.randn(B, 100)}
        # Run twice — should be deterministic in eval
        with torch.no_grad():
            o1 = model(inputs)["survival"]
            o2 = model(inputs)["survival"]
        self.assertTrue(torch.allclose(o1, o2))


class TestPerturbationSensitivity(unittest.TestCase):
    def test_output_shape(self):
        from experiments.cvpr_reliability_pilot.losses import perturbation_sensitivity
        from experiments.cvpr_reliability_pilot.models import AuxCoxHead
        head = AuxCoxHead(64)
        features = torch.randn(8, 64)
        sens = perturbation_sensitivity(head, features, sigma=0.1, n_perturbations=3)
        self.assertEqual(sens.shape, (8,))
        self.assertTrue((sens >= 0).all())

    def test_zero_sigma_zero_sensitivity(self):
        from experiments.cvpr_reliability_pilot.losses import perturbation_sensitivity
        from experiments.cvpr_reliability_pilot.models import AuxCoxHead
        head = AuxCoxHead(64)
        features = torch.randn(8, 64)
        sens = perturbation_sensitivity(head, features, sigma=0.0, n_perturbations=3)
        self.assertTrue(torch.allclose(sens, torch.zeros(8)))


class TestReliabilityLoss(unittest.TestCase):
    def test_finite_loss(self):
        from experiments.cvpr_reliability_pilot.losses import reliability_loss
        from experiments.cvpr_reliability_pilot.models import PCRFModel
        model = PCRFModel(n_genes=100, image_dim=64, dropout=0.0)
        B = 8
        inputs = {"image": torch.randn(B, 64), "gene": torch.randn(B, 100)}
        outputs = model(inputs)
        loss = reliability_loss(
            model, outputs["reliability_logits"], sigma=0.1, n_perturbations=3,
        )
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))


class TestCombinedLoss(unittest.TestCase):
    def test_adamw_method(self):
        from experiments.cvpr_reliability_pilot.losses import combined_loss
        from models.joint_model import build_model
        model = build_model(100, 64, "attention", 0.3, verbose=False)
        B = 4
        inputs = {"image": torch.randn(B, 64), "gene": torch.randn(B, 100)}
        outputs = model(inputs)
        os_m = torch.tensor([30.0, 20.0, 10.0, 5.0])
        vs = torch.tensor([1.0, 0.0, 1.0, 0.0])
        total, loss_dict = combined_loss(
            outputs, os_m, vs, model=model,
            use_aux=False, use_reliability=False,
        )
        self.assertIn("joint", loss_dict)
        self.assertIn("total", loss_dict)

    def test_pcrf_method(self):
        from experiments.cvpr_reliability_pilot.losses import combined_loss
        from experiments.cvpr_reliability_pilot.models import PCRFModel
        model = PCRFModel(n_genes=100, image_dim=64, dropout=0.0,
                         use_aux_loss=True, use_reliability_loss=True)
        B = 8
        inputs = {"image": torch.randn(B, 64), "gene": torch.randn(B, 100)}
        outputs = model(inputs)
        os_m = torch.rand(B) * 100
        vs = (torch.rand(B) > 0.5).float()
        vs[0] = 1.0  # ensure at least one event
        total, loss_dict = combined_loss(
            outputs, os_m, vs, model=model,
            use_aux=True, use_reliability=True,
            reliability_weight=0.01,
        )
        self.assertIn("reliability", loss_dict)
        self.assertIn("aux_img", loss_dict)
        total.backward()


class TestCapacityMatch(unittest.TestCase):
    def test_learned_gating_same_params_as_pcrf(self):
        """Learned gating must have identical capacity to PCRF."""
        from experiments.cvpr_reliability_pilot.models import build_pilot_model
        model_lg, _ = build_pilot_model("learned_gating", 100, 64)
        model_pcrf, _ = build_pilot_model("pcrf", 100, 64)
        params_lg = sum(p.numel() for p in model_lg.parameters())
        params_pcrf = sum(p.numel() for p in model_pcrf.parameters())
        self.assertEqual(params_lg, params_pcrf)


class TestIdenticalInitialization(unittest.TestCase):
    def test_shared_encoder_weights(self):
        """All methods must start from the same encoder weights."""
        from experiments.cvpr_reliability_pilot.train import _build_initial_state, _load_compatible_state
        from experiments.cvpr_reliability_pilot.models import build_pilot_model
        import hashlib

        state = _build_initial_state(100, 64, seed=42)

        # Build two different models and load compatible weights
        model_a, _ = build_pilot_model("adamw", 100, 64)
        model_b, _ = build_pilot_model("pcrf_aux", 100, 64)

        _load_compatible_state(model_a, state)
        _load_compatible_state(model_b, state)

        # Check that shared encoder weights are identical
        for name_a, param_a in model_a.named_parameters():
            if "image_encoder" in name_a or "gene_encoder" in name_a:
                found = False
                for name_b, param_b in model_b.named_parameters():
                    if name_a == name_b:
                        self.assertTrue(
                            torch.equal(param_a, param_b),
                            f"Mismatch in {name_a}",
                        )
                        found = True
                        break
                self.assertTrue(found, f"Key {name_a} not found in model_b")


class TestPerturbationCalibrator(unittest.TestCase):
    def test_running_stats(self):
        from experiments.cvpr_reliability_pilot.reliability import PerturbationCalibrator
        cal = PerturbationCalibrator(decay=0.9)
        for _ in range(10):
            cal.update(torch.randn(8).abs(), torch.randn(8).abs())
        stats = cal.get_stats()
        self.assertEqual(stats["n_updates"], 10)
        self.assertGreater(stats["img_mean"], 0)


class TestReliabilityCollapse(unittest.TestCase):
    def test_detect_collapse(self):
        from experiments.cvpr_reliability_pilot.reliability import check_reliability_collapse
        diag = {"alpha_img": [0.99] * 100}
        result = check_reliability_collapse(diag, threshold=0.05)
        self.assertTrue(result["collapsed"])

    def test_no_collapse(self):
        from experiments.cvpr_reliability_pilot.reliability import check_reliability_collapse
        rng = np.random.RandomState(42)
        diag = {"alpha_img": rng.uniform(0.3, 0.7, 100).tolist()}
        result = check_reliability_collapse(diag, threshold=0.05)
        self.assertFalse(result["collapsed"])


class TestRobustnessEvaluation(unittest.TestCase):
    def test_clean_evaluation(self):
        from experiments.cvpr_reliability_pilot.evaluate import evaluate_robustness
        from models.joint_model import build_model
        from torch.utils.data import DataLoader, TensorDataset

        model = build_model(50, 32, "attention", 0.0, verbose=False)
        model.eval()

        B = 16
        images = torch.randn(B, 32)
        genes = torch.randn(B, 50)
        os_m = torch.rand(B) * 100
        vs = (torch.rand(B) > 0.5).float()
        vs[0] = 1.0

        # Create a simple loader-like structure
        class FakeLoader:
            def __init__(self):
                self.dataset = None
            def __iter__(self):
                yield (
                    {"image": images, "gene": genes},
                    {"os_months": os_m, "vital_status": vs, "survival": torch.zeros(B).long()},
                )
            def __len__(self):
                return 1

        loader = FakeLoader()
        loss, ci, n = evaluate_robustness(model, loader, "cpu")
        self.assertEqual(n, B)
        self.assertGreater(ci, 0)
        self.assertLessEqual(ci, 1)


class TestBuildPilotModel(unittest.TestCase):
    def test_all_methods_build(self):
        from experiments.cvpr_reliability_pilot.models import build_pilot_model, ALL_METHODS
        for method in ALL_METHODS:
            model, cfg = build_pilot_model(method, 100, 64)
            self.assertIsNotNone(model)
            B = 4
            inputs = {"image": torch.randn(B, 64), "gene": torch.randn(B, 100)}
            outputs = model(inputs)
            self.assertEqual(outputs["survival"].shape, (B, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
