"""Unit tests for CLAM attention MIL experiment.

Tests: tensor shapes, attention normalization, variable bag lengths,
gradient propagation, RA-SAMO compatibility, collate, and capping.

Run:  python -m pytest experiments/clam_brca/test_clam.py -v
"""

import sys, os
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.clam_brca.models import (
    GatedAttentionAggregator, CLAMImageEncoder,
    JointModelCLAM, JointModelCLAMWithAux)
from experiments.clam_brca.dataset import (
    clam_collate_fn, load_patches, load_meanpooled)
from experiments.samo.objectives import cox_efron_free, ranking_loss
from experiments.samo.representation_gradients import (
    prepare_branch_gradients, apply_branch_gradients)


class TestGatedAttentionAggregator(unittest.TestCase):

    def test_output_shape(self):
        agg = GatedAttentionAggregator(input_dim=1024, hidden_dim=256)
        patches = torch.randn(100, 1024)
        z, att = agg(patches)
        self.assertEqual(z.shape, (1024,))
        self.assertEqual(att.shape, (100,))

    def test_attention_sums_to_one(self):
        agg = GatedAttentionAggregator(1024, 256)
        agg.eval()
        with torch.no_grad():
            _, att = agg(torch.randn(50, 1024))
        self.assertAlmostEqual(att.sum().item(), 1.0, places=5)

    def test_single_patch(self):
        agg = GatedAttentionAggregator(1024, 256)
        agg.eval()
        with torch.no_grad():
            z, att = agg(torch.randn(1, 1024))
        self.assertEqual(z.shape, (1024,))
        self.assertAlmostEqual(att.item(), 1.0, places=5)

    def test_gradient_flow_to_params(self):
        agg = GatedAttentionAggregator(64, 32, dropout=0.0)
        patches = torch.randn(20, 64)
        z, _ = agg(patches)
        z.sum().backward()
        for name, p in agg.named_parameters():
            # attention_v.bias gradient is mathematically zero:
            # adding a constant to all softmax inputs doesn't change output
            if name == "attention_v.bias":
                continue
            self.assertIsNotNone(p.grad, f"{name} has no gradient")
            self.assertGreater(p.grad.abs().sum().item(), 0,
                               f"{name} gradient is zero")

    def test_different_input_dims(self):
        for dim in [512, 1024, 1536]:
            agg = GatedAttentionAggregator(input_dim=dim, hidden_dim=128)
            z, att = agg(torch.randn(30, dim))
            self.assertEqual(z.shape, (dim,))


class TestCLAMImageEncoder(unittest.TestCase):

    def test_output_shape(self):
        enc = CLAMImageEncoder(1024, 256, dropout=0.0)
        patches = torch.randn(150, 1024)
        counts = torch.tensor([50, 30, 70])
        out = enc(patches, counts)
        self.assertEqual(out.shape, (3, 512))

    def test_single_slide_batch(self):
        enc = CLAMImageEncoder(1024, 256, dropout=0.0)
        enc.eval()  # BatchNorm1d requires B>1 in train mode
        with torch.no_grad():
            out = enc(torch.randn(40, 1024), torch.tensor([40]))
        self.assertEqual(out.shape, (1, 512))

    def test_variable_lengths(self):
        """Verify different patch counts produce different embeddings."""
        enc = CLAMImageEncoder(1024, 256, dropout=0.0)
        enc.eval()
        torch.manual_seed(0)
        p1 = torch.randn(10, 1024)
        p2 = torch.randn(200, 1024)
        with torch.no_grad():
            out1 = enc(p1, torch.tensor([10]))
            out2 = enc(p2, torch.tensor([200]))
        self.assertFalse(torch.allclose(out1, out2))

    def test_gradient_flow_through_attention(self):
        """All attention params receive nonzero gradients."""
        enc = CLAMImageEncoder(64, 32, dropout=0.0)
        patches = torch.randn(30, 64)
        counts = torch.tensor([15, 15])
        out = enc(patches, counts)
        out.sum().backward()
        attention_params = [
            (n, p) for n, p in enc.named_parameters()
            if "attention" in n or "aggregator" in n]
        self.assertGreater(len(attention_params), 0,
                           "No attention parameters found")
        for name, p in attention_params:
            self.assertIsNotNone(p.grad, f"{name} has no gradient")
            self.assertGreater(p.grad.abs().sum().item(), 0,
                               f"{name} gradient is zero")

    def test_attention_weights_cached(self):
        enc = CLAMImageEncoder(1024, 256, dropout=0.0)
        enc.eval()
        with torch.no_grad():
            enc(torch.randn(50, 1024), torch.tensor([20, 30]))
        attns = enc.get_attention_weights()
        self.assertIsNotNone(attns)
        self.assertEqual(len(attns), 2)
        self.assertEqual(attns[0].shape, (20,))
        self.assertEqual(attns[1].shape, (30,))


class TestJointModels(unittest.TestCase):

    def _make_inputs(self, batch=2, patches_per=25, n_genes=100, dim=64):
        return {
            "patches": torch.randn(batch * patches_per, dim),
            "patch_counts": torch.tensor([patches_per] * batch),
            "gene": torch.randn(batch, n_genes),
        }

    def test_clam_forward(self):
        model = JointModelCLAM(n_genes=100, input_dim=64, hidden_dim=32)
        inp = self._make_inputs()
        out = model(inp)
        self.assertIn("survival", out)
        self.assertEqual(out["survival"].shape, (2, 2))
        self.assertNotIn("aux_img_risk", out)

    def test_clam_with_aux_forward(self):
        model = JointModelCLAMWithAux(n_genes=100, input_dim=64,
                                       hidden_dim=32)
        inp = self._make_inputs()
        out = model(inp)
        self.assertEqual(out["survival"].shape, (2, 2))
        self.assertEqual(out["aux_img_risk"].shape, (2,))
        self.assertEqual(out["aux_gene_risk"].shape, (2,))


class TestRASAMOCompatibility(unittest.TestCase):
    """Verify prepare_branch_gradients works with CLAM models."""

    def test_prepare_and_apply(self):
        model = JointModelCLAMWithAux(
            n_genes=100, input_dim=64, hidden_dim=32, dropout=0.0)
        inp = {
            "patches": torch.randn(40, 64),
            "patch_counts": torch.tensor([20, 20]),
            "gene": torch.randn(2, 100),
        }
        outputs = model(inp)

        t = torch.tensor([12.0, 36.0])
        e = torch.tensor([1.0, 0.0])

        risk = outputs["survival"][:, 1]
        fusion = cox_efron_free(risk, t, e)
        img = cox_efron_free(outputs["aux_img_risk"], t, e)
        gene = cox_efron_free(outputs["aux_gene_risk"], t, e)
        rank = ranking_loss(risk, t, e)

        # This must not raise
        prepared = prepare_branch_gradients(
            model, fusion, img, gene, rank,
            [0.1, 0.1, 0.1], project=True)

        self.assertIn("image", prepared)
        self.assertIn("gene", prepared)

        # Image branch must include CLAM attention params
        img_params = [p for p in model.image_encoder.parameters()
                      if p.requires_grad]
        self.assertEqual(len(prepared["image"][0]), len(img_params))
        self.assertGreater(len(img_params), 0)

        # Apply after backward
        loss = fusion + 0.1 * img + 0.1 * gene + 0.1 * rank
        loss.backward()
        diagnostics = apply_branch_gradients(prepared)

        # Diagnostics should have cosine values
        for branch in ("image", "gene"):
            diag = diagnostics[branch]
            self.assertTrue(hasattr(diag, "cosine_aux"))
            self.assertTrue(hasattr(diag, "projected_aux"))

    def test_attention_params_receive_projected_gradients(self):
        """After apply_branch_gradients, attention params have nonzero .grad."""
        model = JointModelCLAMWithAux(
            n_genes=50, input_dim=32, hidden_dim=16, dropout=0.0)
        inp = {
            "patches": torch.randn(20, 32),
            "patch_counts": torch.tensor([10, 10]),
            "gene": torch.randn(2, 50),
        }
        outputs = model(inp)
        t = torch.tensor([10.0, 40.0])
        e = torch.tensor([1.0, 1.0])

        risk = outputs["survival"][:, 1]
        fusion = cox_efron_free(risk, t, e)
        img = cox_efron_free(outputs["aux_img_risk"], t, e)
        gene = cox_efron_free(outputs["aux_gene_risk"], t, e)
        rank = ranking_loss(risk, t, e)

        prepared = prepare_branch_gradients(
            model, fusion, img, gene, rank, [0.1, 0.1, 0.1], project=True)

        loss = fusion + 0.1 * img + 0.1 * gene + 0.1 * rank
        loss.backward()
        apply_branch_gradients(prepared)

        # Check that attention parameters in image_encoder have grads
        attn_params_with_grad = 0
        for name, p in model.image_encoder.named_parameters():
            if "attention" in name or "aggregator" in name:
                if p.grad is not None and p.grad.abs().sum() > 0:
                    attn_params_with_grad += 1
        self.assertGreater(attn_params_with_grad, 0,
                           "No attention params received projected gradients")

    def test_adamw_gradient_flow_without_rasamo(self):
        """Plain AdamW also flows gradients to attention params."""
        model = JointModelCLAM(
            n_genes=50, input_dim=32, hidden_dim=16, dropout=0.0)
        inp = {
            "patches": torch.randn(20, 32),
            "patch_counts": torch.tensor([10, 10]),
            "gene": torch.randn(2, 50),
        }
        outputs = model(inp)
        risk = outputs["survival"][:, 1]
        loss = cox_efron_free(risk,
                              torch.tensor([10.0, 40.0]),
                              torch.tensor([1.0, 1.0]))
        loss.backward()

        for name, p in model.image_encoder.named_parameters():
            if "aggregator" in name:
                self.assertIsNotNone(p.grad, f"{name} missing grad under AdamW")
                self.assertGreater(p.grad.abs().sum().item(), 0,
                                   f"{name} zero grad under AdamW")


class TestCollateFunction(unittest.TestCase):

    def test_basic_collate(self):
        batch = [
            ({"patches": torch.randn(50, 1024), "gene": torch.randn(100)},
             {"os_months": 12.0, "vital_status": 1, "survival": 1}),
            ({"patches": torch.randn(30, 1024), "gene": torch.randn(100)},
             {"os_months": 24.0, "vital_status": 0, "survival": 0}),
        ]
        inputs, labels = clam_collate_fn(batch)
        self.assertEqual(inputs["patches"].shape, (80, 1024))
        self.assertEqual(inputs["patch_counts"].tolist(), [50, 30])
        self.assertEqual(inputs["gene"].shape, (2, 100))
        self.assertEqual(labels["os_months"].tolist(), [12.0, 24.0])
        self.assertEqual(labels["vital_status"].tolist(), [1, 0])

    def test_single_item_batch(self):
        batch = [
            ({"patches": torch.randn(10, 512), "gene": torch.randn(50)},
             {"os_months": 5.0, "vital_status": 1, "survival": 1}),
        ]
        inputs, labels = clam_collate_fn(batch)
        self.assertEqual(inputs["patches"].shape, (10, 512))
        self.assertEqual(inputs["patch_counts"].tolist(), [10])


class TestPatchLoading(unittest.TestCase):

    def test_subsample_deterministic(self):
        """Same barcode → same subsample."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.pt")
            torch.save(torch.randn(200, 64), path)

            p1 = load_patches([path], 64, max_patches=50,
                              patient_barcode="TCGA-XX-0001")
            p2 = load_patches([path], 64, max_patches=50,
                              patient_barcode="TCGA-XX-0001")
            self.assertTrue(torch.equal(p1, p2))

    def test_subsample_different_barcodes(self):
        """Different barcodes → different subsamples."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.pt")
            torch.save(torch.randn(200, 64), path)

            p1 = load_patches([path], 64, max_patches=50,
                              patient_barcode="TCGA-XX-0001")
            p2 = load_patches([path], 64, max_patches=50,
                              patient_barcode="TCGA-XX-0002")
            self.assertFalse(torch.equal(p1, p2))

    def test_no_subsample_when_under_cap(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.pt")
            orig = torch.randn(30, 64)
            torch.save(orig, path)
            loaded = load_patches([path], 64, max_patches=100)
            self.assertEqual(loaded.shape[0], 30)

    def test_multi_slide_concatenation(self):
        """Multiple files → concatenated patches."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "slide1.pt")
            p2 = os.path.join(tmp, "slide2.pt")
            torch.save(torch.randn(20, 64), p1)
            torch.save(torch.randn(30, 64), p2)
            loaded = load_patches([p1, p2], 64, max_patches=100)
            self.assertEqual(loaded.shape, (50, 64))

    def test_meanpool(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.pt")
            data = torch.ones(100, 64) * 3.0
            torch.save(data, path)
            pooled = load_meanpooled([path], 64)
            self.assertEqual(pooled.shape, (64,))
            self.assertAlmostEqual(pooled.mean().item(), 3.0, places=4)

    def test_fallback_on_missing(self):
        result = load_patches(["/nonexistent/file.pt"], 1024, 4096)
        self.assertEqual(result.shape, (1, 1024))
        self.assertTrue((result == 0).all())


class TestModelInitialization(unittest.TestCase):
    """Verify matched initialization across methods."""

    def test_gene_encoder_matches(self):
        """Gene encoder weights identical between baseline and CLAM."""
        from models.joint_model import build_model
        from experiments.clam_brca.train import _copy_matching

        torch.manual_seed(42)
        baseline = build_model(100, 64, "attention", 0.3, verbose=False)
        torch.manual_seed(42)
        baseline2 = build_model(100, 64, "attention", 0.3, verbose=False)

        clam = JointModelCLAMWithAux(100, 64, 32, 0.3)
        _copy_matching(baseline, clam)

        # Gene encoder should match
        for (n1, p1), (n2, p2) in zip(
                baseline2.gene_encoder.named_parameters(),
                clam.gene_encoder.named_parameters()):
            self.assertTrue(torch.equal(p1, p2),
                            f"gene_encoder.{n1} mismatch")

        # Fusion should match
        for (n1, p1), (n2, p2) in zip(
                baseline2.fusion.named_parameters(),
                clam.fusion.named_parameters()):
            self.assertTrue(torch.equal(p1, p2),
                            f"fusion.{n1} mismatch")


if __name__ == "__main__":
    unittest.main()
