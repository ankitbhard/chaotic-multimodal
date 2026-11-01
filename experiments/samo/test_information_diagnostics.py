"""Unit tests for information acquisition diagnostics.

Tests probe invariance, edge cases, Fisher trace correctness,
state preservation, and deterministic probing.
"""
import math
import unittest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset

from experiments.samo.information_diagnostics import (
    build_probe_subset,
    _per_event_scores,
    approx_fisher_trace,
    per_layer_fisher,
    compute_epoch_diagnostics,
    diagnostics_to_row,
    fit_unimodal_probes,
    _cosine_sim,
    _flat_grads,
    LinearCoxProbe,
)
from experiments.samo.objectives import cox_efron_free


# ---------------------------------------------------------------------------
# Minimal model for testing
# ---------------------------------------------------------------------------

class MinimalEncoder(nn.Module):
    FEATURE_DIM = 4

    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.ReLU(),
        )
        self.feature_dim = 4

    def forward(self, x):
        return self.encoder(x)


class MinimalModel(nn.Module):
    """Minimal model matching JointModelWithAux interface."""

    def __init__(self, image_dim=8, n_genes=16):
        super().__init__()
        self.image_encoder = MinimalEncoder(image_dim)
        self.gene_encoder = MinimalEncoder(n_genes)
        self.fusion = nn.Linear(8, 4)
        self.shared = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.head_survival = nn.Linear(4, 2)
        self.aux_img = nn.Sequential(nn.Linear(4, 1))
        self.aux_gene = nn.Sequential(nn.Linear(4, 1))
        self._attn_weights = None

    def forward(self, inputs, modality_mask=None):
        img = self.image_encoder(inputs["image"])
        gene = self.gene_encoder(inputs["gene"])
        if modality_mask is not None:
            img = img * modality_mask[:, 0:1]
            gene = gene * modality_mask[:, 1:2]
        fused = self.fusion(torch.cat([img, gene], dim=1))
        self._attn_weights = torch.ones(img.shape[0], 2) * 0.5
        shared = self.shared(fused)
        return {
            "survival": self.head_survival(shared),
            "aux_img_risk": self.aux_img(img).squeeze(-1),
            "aux_gene_risk": self.aux_gene(gene).squeeze(-1),
        }


class MinimalModelNoAux(nn.Module):
    """Model without aux heads (for baseline/rank)."""

    def __init__(self, image_dim=8, n_genes=16):
        super().__init__()
        self.image_encoder = MinimalEncoder(image_dim)
        self.gene_encoder = MinimalEncoder(n_genes)
        self.fusion = nn.Linear(8, 4)
        self.shared = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.head_survival = nn.Linear(4, 2)
        self._attn_weights = None

    def forward(self, inputs, modality_mask=None):
        img = self.image_encoder(inputs["image"])
        gene = self.gene_encoder(inputs["gene"])
        fused = self.fusion(torch.cat([img, gene], dim=1))
        self._attn_weights = torch.ones(img.shape[0], 2) * 0.5
        shared = self.shared(fused)
        return {"survival": self.head_survival(shared)}


def _make_dataset(n=32, image_dim=8, n_genes=16, event_rate=0.5, seed=42):
    """Create a synthetic survival dataset."""
    rng = np.random.RandomState(seed)
    images = torch.randn(n, image_dim)
    genes = torch.randn(n, n_genes)
    times = torch.tensor(rng.exponential(12, n), dtype=torch.float32)
    events = torch.tensor(rng.binomial(1, event_rate, n), dtype=torch.float32)
    # Ensure at least one event
    events[0] = 1.0
    times[0] = 5.0

    class SurvDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.images = images
            self.genes = genes
            self.times = times
            self.events = events

        def __len__(self):
            return n

        def __getitem__(self, idx):
            inputs = {"image": self.images[idx], "gene": self.genes[idx]}
            labels = {
                "os_months": self.times[idx],
                "vital_status": self.events[idx],
            }
            return inputs, labels

    return SurvDataset()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProbeSubset(unittest.TestCase):
    def test_deterministic(self):
        ds = _make_dataset(n=100, seed=42)
        idx1, meta1 = build_probe_subset(ds, n_probe=20, seed=99)
        idx2, meta2 = build_probe_subset(ds, n_probe=20, seed=99)
        self.assertEqual(idx1, idx2)
        self.assertEqual(meta1["patient_hashes"], meta2["patient_hashes"])

    def test_different_seeds_give_different_subsets(self):
        ds = _make_dataset(n=100, seed=42)
        idx1, _ = build_probe_subset(ds, n_probe=20, seed=1)
        idx2, _ = build_probe_subset(ds, n_probe=20, seed=2)
        self.assertNotEqual(idx1, idx2)

    def test_stratified(self):
        ds = _make_dataset(n=100, event_rate=0.3, seed=42)
        idx, meta = build_probe_subset(ds, n_probe=20, seed=42)
        self.assertGreater(meta["n_events"], 0)
        self.assertGreater(meta["n_censored"], 0)
        self.assertEqual(meta["n_events"] + meta["n_censored"], len(idx))

    def test_small_dataset(self):
        ds = _make_dataset(n=5, seed=42)
        idx, meta = build_probe_subset(ds, n_probe=10, seed=42)
        self.assertLessEqual(len(idx), 5)  # Can't exceed dataset size


class TestPerEventScores(unittest.TestCase):
    def test_correct_number_of_scores(self):
        torch.manual_seed(42)
        model = MinimalModel()
        ds = _make_dataset(n=16, seed=42)
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        inputs, labels = next(iter(loader))
        outputs = model(inputs)
        risk = outputs["survival"][:, 1]
        t = labels["os_months"].float()
        e = labels["vital_status"].float()
        n_events = int(e.bool().sum().item())
        params = list(model.image_encoder.parameters())
        scores = _per_event_scores(risk, t, e, params)
        self.assertEqual(len(scores), n_events)

    def test_all_censored_returns_empty(self):
        torch.manual_seed(42)
        model = MinimalModel()
        risk = torch.randn(8, requires_grad=True)
        time = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.])
        event = torch.zeros(8)
        params = list(model.image_encoder.parameters())
        scores = _per_event_scores(risk, time, event, params)
        self.assertEqual(len(scores), 0)

    def test_single_event(self):
        torch.manual_seed(42)
        model = MinimalModel()
        ds = _make_dataset(n=8, event_rate=0.0, seed=42)
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        inputs, labels = next(iter(loader))
        # Force exactly one event
        labels["vital_status"][:] = 0
        labels["vital_status"][0] = 1
        outputs = model(inputs)
        risk = outputs["survival"][:, 1]
        params = list(model.image_encoder.parameters())
        scores = _per_event_scores(risk, labels["os_months"], labels["vital_status"], params)
        self.assertEqual(len(scores), 1)
        # Score should be finite
        self.assertTrue(torch.isfinite(scores[0]).all())

    def test_tied_event_times(self):
        torch.manual_seed(42)
        model = MinimalModel()
        risk = model({"image": torch.randn(4, 8), "gene": torch.randn(4, 16)})["survival"][:, 1]
        time = torch.tensor([5., 5., 10., 15.])
        event = torch.tensor([1., 1., 0., 1.])
        params = list(model.image_encoder.parameters())
        scores = _per_event_scores(risk, time, event, params)
        self.assertEqual(len(scores), 3)  # 3 events
        for s in scores:
            self.assertTrue(torch.isfinite(s).all())


class TestFisherTrace(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(approx_fisher_trace([]), 0.0)

    def test_positive(self):
        scores = [torch.randn(10) for _ in range(5)]
        ft = approx_fisher_trace(scores)
        self.assertGreater(ft, 0.0)

    def test_per_layer_breakdown(self):
        scores = [torch.randn(15) for _ in range(3)]
        shapes = [(5,), (10,)]  # two layers: 5 + 10 = 15
        result = per_layer_fisher(scores, shapes)
        self.assertIn("layer_0_fisher", result)
        self.assertIn("layer_1_fisher", result)
        self.assertIn("layer_0_fisher_normalized", result)

    def test_layer_fisher_sums_to_total(self):
        scores = [torch.randn(15) for _ in range(5)]
        shapes = [(5,), (10,)]
        result = per_layer_fisher(scores, shapes)
        total = approx_fisher_trace(scores)
        layer_sum = result["layer_0_fisher"] + result["layer_1_fisher"]
        self.assertAlmostEqual(total, layer_sum, places=5)


class TestStatePreservation(unittest.TestCase):
    """Verify diagnostics don't modify model or optimizer state."""

    def test_model_params_unchanged(self):
        torch.manual_seed(42)
        model = MinimalModel()
        ds = _make_dataset(n=32, seed=42)
        probe_ds = Subset(ds, list(range(16)))
        probe_loader = DataLoader(probe_ds, batch_size=8, shuffle=False)
        val_loader = DataLoader(ds, batch_size=16, shuffle=False)

        # Record parameter values before
        params_before = {n: p.data.clone() for n, p in model.named_parameters()}
        grads_before = {n: (p.grad.clone() if p.grad is not None else None)
                        for n, p in model.named_parameters()}

        # Run diagnostics
        model.train()
        diag = compute_epoch_diagnostics(
            model, probe_loader, val_loader, torch.device("cpu"),
            "adaptive_aux", 1, "test", 0,
        )

        # Verify parameters unchanged
        for name, p in model.named_parameters():
            self.assertTrue(
                torch.equal(p.data, params_before[name]),
                f"Parameter {name} changed after diagnostics",
            )

        # Verify grad fields restored
        for name, p in model.named_parameters():
            if grads_before[name] is None:
                self.assertIsNone(p.grad, f"Grad appeared for {name}")
            else:
                self.assertTrue(
                    torch.equal(p.grad, grads_before[name]),
                    f"Grad changed for {name}",
                )

    def test_training_mode_restored(self):
        torch.manual_seed(42)
        model = MinimalModel()
        ds = _make_dataset(n=16, seed=42)
        probe_loader = DataLoader(Subset(ds, [0, 1, 2, 3]), batch_size=4)
        val_loader = DataLoader(ds, batch_size=8)

        model.train()
        self.assertTrue(model.training)
        compute_epoch_diagnostics(
            model, probe_loader, val_loader, torch.device("cpu"),
            "adaptive_aux", 1, "test", 0,
        )
        self.assertTrue(model.training)

    def test_rng_state_preserved(self):
        torch.manual_seed(42)
        model = MinimalModel()
        ds = _make_dataset(n=16, seed=42)
        probe_loader = DataLoader(Subset(ds, [0, 1, 2, 3]), batch_size=4)
        val_loader = DataLoader(ds, batch_size=8)

        rng_before = torch.random.get_rng_state().clone()
        np_state_before = np.random.get_state()[1].copy()

        compute_epoch_diagnostics(
            model, probe_loader, val_loader, torch.device("cpu"),
            "adaptive_aux", 1, "test", 0,
        )

        rng_after = torch.random.get_rng_state()
        np_state_after = np.random.get_state()[1]

        self.assertTrue(torch.equal(rng_before, rng_after))
        np.testing.assert_array_equal(np_state_before, np_state_after)


class TestDiagnosticsOutput(unittest.TestCase):
    def test_row_format(self):
        torch.manual_seed(42)
        model = MinimalModel()
        ds = _make_dataset(n=32, seed=42)
        probe_loader = DataLoader(Subset(ds, list(range(16))), batch_size=8)
        val_loader = DataLoader(ds, batch_size=16)

        diag = compute_epoch_diagnostics(
            model, probe_loader, val_loader, torch.device("cpu"),
            "adaptive_aux", 5, "brca", 0,
        )
        row = diagnostics_to_row(diag)

        # Check required columns
        required = [
            "epoch", "cancer", "method", "fold",
            "img_cox_grad_norm", "img_fisher_trace", "img_fisher_trace_normalized",
            "gene_cox_grad_norm", "gene_fisher_trace", "gene_fisher_trace_normalized",
            "cosine_cox_aux", "cosine_cox_rank",
            "val_ci_fused", "val_ci_image_unimodal", "val_ci_gene_unimodal",
            "fisher_ratio_gene_over_img", "grad_norm_ratio_gene_over_img",
        ]
        for col in required:
            self.assertIn(col, row, f"Missing column: {col}")

        # Values should be finite
        for k, v in row.items():
            if isinstance(v, float):
                self.assertTrue(math.isfinite(v), f"{k} is not finite: {v}")

    def test_diagnostics_with_no_aux_heads(self):
        """Baseline model uses frozen-feature probes."""
        torch.manual_seed(42)
        model = MinimalModelNoAux()
        ds = _make_dataset(n=32, seed=42)
        probe_loader = DataLoader(Subset(ds, list(range(16))), batch_size=8)
        val_loader = DataLoader(ds, batch_size=16)
        train_loader = DataLoader(ds, batch_size=16)

        # Fit probes
        probes = fit_unimodal_probes(model, train_loader, torch.device("cpu"))

        diag = compute_epoch_diagnostics(
            model, probe_loader, val_loader, torch.device("cpu"),
            "baseline", 1, "test", 0,
            unimodal_probes=probes,
        )
        row = diagnostics_to_row(diag)
        # Unimodal CI should be available via probes
        self.assertIsInstance(row["val_ci_image_unimodal"], float)
        self.assertIsInstance(row["val_ci_gene_unimodal"], float)


class TestLinearCoxProbe(unittest.TestCase):
    def test_fit_and_predict(self):
        torch.manual_seed(42)
        model = MinimalModelNoAux()
        ds = _make_dataset(n=32, seed=42)
        train_loader = DataLoader(ds, batch_size=16)

        probes = fit_unimodal_probes(model, train_loader, torch.device("cpu"))
        self.assertIn("image", probes)
        self.assertIn("gene", probes)

        # Probes should produce finite outputs
        with torch.no_grad():
            features = model.image_encoder(torch.randn(4, 8))
            risk = probes["image"](features)
            self.assertEqual(risk.shape, (4,))
            self.assertTrue(torch.isfinite(risk).all())


class TestEdgeCases(unittest.TestCase):
    def test_all_censored_batch(self):
        """Diagnostics should handle all-censored batches gracefully."""
        torch.manual_seed(42)
        model = MinimalModel()
        # Create dataset with NO events
        ds = _make_dataset(n=16, event_rate=0.0, seed=42)
        # Force all censored
        for i in range(len(ds)):
            ds.events[i] = 0.0

        probe_loader = DataLoader(Subset(ds, list(range(8))), batch_size=8)
        val_loader = DataLoader(ds, batch_size=16)

        diag = compute_epoch_diagnostics(
            model, probe_loader, val_loader, torch.device("cpu"),
            "baseline", 1, "test", 0,
        )
        row = diagnostics_to_row(diag)
        # Fisher trace should be 0 with no events
        self.assertEqual(diag.image_encoder.n_events_fisher, 0)
        self.assertEqual(diag.image_encoder.fisher_trace, 0.0)

    def test_single_sample(self):
        """Single-sample probe should not crash."""
        torch.manual_seed(42)
        model = MinimalModel()
        ds = _make_dataset(n=16, seed=42)
        probe_loader = DataLoader(Subset(ds, [0]), batch_size=1)
        val_loader = DataLoader(ds, batch_size=16)

        diag = compute_epoch_diagnostics(
            model, probe_loader, val_loader, torch.device("cpu"),
            "adaptive_aux", 1, "test", 0,
        )
        row = diagnostics_to_row(diag)
        for k, v in row.items():
            if isinstance(v, float):
                self.assertTrue(math.isfinite(v), f"{k} is not finite: {v}")


if __name__ == "__main__":
    unittest.main()
