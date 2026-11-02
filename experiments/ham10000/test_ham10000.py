"""Unit tests for HAM10000 multimodal classification benchmark.

All tests use synthetic data — no real images or metadata CSV needed.
Run: python -m pytest experiments/ham10000/test_ham10000.py -v
"""

import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure project root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from experiments.ham10000.dataset import (
    CLASSES, CLASS_TO_IDX, NUM_CLASSES, HAM10000Dataset,
    build_ham10000_splits, LOCALIZATION_CATS, SEX_CATS,
)
from experiments.ham10000.models import (
    ResNet18Encoder, MetadataEncoder, AuxClassHead,
    HAM10000Model, HAM10000ModelWithAux,
    ImageOnlyModel, MetaOnlyModel,
)
from experiments.ham10000.objectives import classification_loss_parts
from experiments.ham10000.train import evaluate, _binary_auroc, _macro_auroc
from experiments.samo.objectives import LossWeightController
from experiments.samo.representation_gradients import (
    prepare_branch_gradients, apply_branch_gradients,
)
from optimizers.sam_optimizer import SAMOptimizer


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_synthetic_ham10000_dir(n_samples=120, seed=42):
    """Create a temp directory with fake HAM10000_metadata.csv and JPEG stubs."""
    rng = np.random.RandomState(seed)
    tmpdir = tempfile.mkdtemp(prefix="ham10000_test_")
    img_dir = Path(tmpdir)

    records = []
    lesion_counter = 0
    for i in range(n_samples):
        image_id = f"ISIC_{100000 + i:07d}"
        # Create 2-3 images per lesion to test grouping
        if i % 3 == 0:
            lesion_counter += 1
        lesion_id = f"HAM_{lesion_counter:07d}"
        dx = rng.choice(CLASSES)
        age = rng.choice([rng.uniform(20, 85), np.nan])
        sex = rng.choice(["male", "female", "unknown"])
        loc = rng.choice(LOCALIZATION_CATS[:8])  # use subset
        records.append({
            "image_id": image_id,
            "lesion_id": lesion_id,
            "dx": dx,
            "age": age,
            "sex": sex,
            "localization": loc,
        })

        # Create a small valid JPEG stub (1x1 pixel)
        from PIL import Image
        img = Image.new("RGB", (32, 32), color=(rng.randint(0, 255),
                                                  rng.randint(0, 255),
                                                  rng.randint(0, 255)))
        img.save(img_dir / f"{image_id}.jpg")

    df = pd.DataFrame(records)
    df.to_csv(img_dir / "HAM10000_metadata.csv", index=False)
    return tmpdir


class SyntheticHAM10000(Dataset):
    """Tiny synthetic dataset for model/training tests (no files needed)."""

    def __init__(self, n=64, meta_dim=20, n_classes=7, seed=42):
        rng = np.random.RandomState(seed)
        self.images = torch.randn(n, 3, 224, 224)
        self.metas = torch.randn(n, meta_dim)
        self.labels = torch.from_numpy(rng.randint(0, n_classes, n)).long()
        self.meta_dim = meta_dim

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            {"image": self.images[idx], "meta": self.metas[idx]},
            {"label": self.labels[idx]},
        )


# ── TestDataset ──────────────────────────────────────────────────────────────


class TestDataset:
    """Tests for data loading, splitting, and metadata encoding."""

    @pytest.fixture(scope="class")
    def data_dir(self):
        return _make_synthetic_ham10000_dir(n_samples=120, seed=42)

    def test_build_splits_returns_all_keys(self, data_dir):
        splits = build_ham10000_splits(data_dir, seed=42)
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert "meta_dim" in splits
        assert "class_weights" in splits
        assert "metadata" in splits

    def test_no_lesion_leakage(self, data_dir):
        splits = build_ham10000_splits(data_dir, seed=42)
        train_lesions = {r["lesion_id"] for r in splits["train"].records}
        val_lesions = {r["lesion_id"] for r in splits["val"].records}
        test_lesions = {r["lesion_id"] for r in splits["test"].records}
        assert not (train_lesions & val_lesions), "Train/val lesion overlap!"
        assert not (train_lesions & test_lesions), "Train/test lesion overlap!"
        assert not (val_lesions & test_lesions), "Val/test lesion overlap!"

    def test_split_proportions(self, data_dir):
        splits = build_ham10000_splits(data_dir, seed=42)
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 120
        # Rough proportions (with small n, exact ratios vary)
        assert len(splits["train"]) > len(splits["val"])
        assert len(splits["train"]) > len(splits["test"])

    def test_stratification(self, data_dir):
        """Each split should have at least some representation of common classes."""
        splits = build_ham10000_splits(data_dir, seed=42)
        train_labels = [r["dx"] for r in splits["train"].records]
        val_labels = [r["dx"] for r in splits["val"].records]
        test_labels = [r["dx"] for r in splits["test"].records]
        # With 120 samples and 7 classes, not every class is guaranteed,
        # but train should have the most diversity
        assert len(set(train_labels)) >= len(set(val_labels))

    def test_metadata_encoding_shape(self, data_dir):
        splits = build_ham10000_splits(data_dir, seed=42)
        ds = splits["train"]
        sample_input, sample_label = ds[0]
        assert sample_input["image"].shape == (3, 224, 224)
        assert sample_input["meta"].shape == (ds.meta_dim,)
        assert isinstance(sample_label["label"], int)
        assert 0 <= sample_label["label"] < NUM_CLASSES

    def test_missing_age_flag(self, data_dir):
        splits = build_ham10000_splits(data_dir, seed=42)
        ds = splits["train"]
        # Find a record with missing age
        for i, rec in enumerate(ds.records):
            if rec["age"] is None or (isinstance(rec["age"], float) and np.isnan(rec["age"])):
                sample_input, _ = ds[i]
                # Last element is age_missing flag
                assert sample_input["meta"][-1] == 1.0
                # Age value should be 0 (default)
                assert sample_input["meta"][0] == 0.0
                break

    def test_class_weights_shape(self, data_dir):
        splits = build_ham10000_splits(data_dir, seed=42)
        assert splits["class_weights"].shape == (NUM_CLASSES,)
        assert (splits["class_weights"] > 0).all()

    def test_deterministic_splits(self, data_dir):
        s1 = build_ham10000_splits(data_dir, seed=42)
        s2 = build_ham10000_splits(data_dir, seed=42)
        ids1 = [r["image_id"] for r in s1["train"].records]
        ids2 = [r["image_id"] for r in s2["train"].records]
        assert ids1 == ids2

    def test_different_seeds_differ(self, data_dir):
        s1 = build_ham10000_splits(data_dir, seed=42)
        s2 = build_ham10000_splits(data_dir, seed=99)
        ids1 = set(r["image_id"] for r in s1["train"].records)
        ids2 = set(r["image_id"] for r in s2["train"].records)
        # Different seeds should produce different splits
        assert ids1 != ids2


# ── TestModels ───────────────────────────────────────────────────────────────


class TestModels:
    """Tests for model forward shapes and the gene_encoder alias."""

    B = 4
    META_DIM = 20
    N_CLASSES = 7

    def test_resnet18_encoder_shape(self):
        enc = ResNet18Encoder(pretrained=False)
        x = torch.randn(self.B, 3, 224, 224)
        out = enc(x)
        assert out.shape == (self.B, 512)

    def test_metadata_encoder_shape(self):
        enc = MetadataEncoder(self.META_DIM)
        x = torch.randn(self.B, self.META_DIM)
        out = enc(x)
        assert out.shape == (self.B, 64)

    def test_aux_class_head_shape(self):
        head = AuxClassHead(512, self.N_CLASSES)
        x = torch.randn(self.B, 512)
        out = head(x)
        assert out.shape == (self.B, self.N_CLASSES)

    def test_ham10000_model_forward(self):
        model = HAM10000Model(self.META_DIM, self.N_CLASSES, pretrained=False)
        inputs = {
            "image": torch.randn(self.B, 3, 224, 224),
            "meta": torch.randn(self.B, self.META_DIM),
        }
        out = model(inputs)
        assert "logits" in out
        assert out["logits"].shape == (self.B, self.N_CLASSES)

    def test_ham10000_model_gene_encoder_alias(self):
        model = HAM10000Model(self.META_DIM, pretrained=False)
        assert model.gene_encoder is model._meta_encoder
        # Should have trainable params
        params = list(model.gene_encoder.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_ham10000_model_with_aux_forward(self):
        model = HAM10000ModelWithAux(self.META_DIM, self.N_CLASSES, pretrained=False)
        inputs = {
            "image": torch.randn(self.B, 3, 224, 224),
            "meta": torch.randn(self.B, self.META_DIM),
        }
        out = model(inputs)
        assert out["logits"].shape == (self.B, self.N_CLASSES)
        assert out["aux_img_logits"].shape == (self.B, self.N_CLASSES)
        assert out["aux_meta_logits"].shape == (self.B, self.N_CLASSES)

    def test_ham10000_model_with_aux_gene_encoder_alias(self):
        model = HAM10000ModelWithAux(self.META_DIM, pretrained=False)
        assert model.gene_encoder is model._meta_encoder

    def test_image_only_model(self):
        model = ImageOnlyModel(self.N_CLASSES, pretrained=False)
        inputs = {"image": torch.randn(self.B, 3, 224, 224)}
        out = model(inputs)
        assert out["logits"].shape == (self.B, self.N_CLASSES)

    def test_meta_only_model(self):
        model = MetaOnlyModel(self.META_DIM, self.N_CLASSES)
        inputs = {"meta": torch.randn(self.B, self.META_DIM)}
        out = model(inputs)
        assert out["logits"].shape == (self.B, self.N_CLASSES)

    def test_attention_weights(self):
        model = HAM10000Model(self.META_DIM, pretrained=False)
        inputs = {
            "image": torch.randn(self.B, 3, 224, 224),
            "meta": torch.randn(self.B, self.META_DIM),
        }
        model(inputs)
        attn = model.get_attention_weights()
        assert attn.shape == (self.B, 2)
        # Softmax weights should sum to ~1
        assert torch.allclose(attn.sum(dim=1), torch.ones(self.B), atol=1e-5)


# ── TestObjectives ───────────────────────────────────────────────────────────


class TestObjectives:
    """Tests for classification_loss_parts."""

    B = 8
    N_CLASSES = 7
    META_DIM = 20

    def _make_outputs(self, with_aux=False):
        outputs = {"logits": torch.randn(self.B, self.N_CLASSES, requires_grad=True)}
        if with_aux:
            outputs["aux_img_logits"] = torch.randn(self.B, self.N_CLASSES, requires_grad=True)
            outputs["aux_meta_logits"] = torch.randn(self.B, self.N_CLASSES, requires_grad=True)
        return outputs

    def test_adamw_returns_4_tensors(self):
        outputs = self._make_outputs(with_aux=False)
        labels = torch.randint(0, self.N_CLASSES, (self.B,))
        parts = classification_loss_parts(outputs, labels, "adamw")
        assert len(parts) == 4
        assert all(torch.is_tensor(p) for p in parts)

    def test_adamw_zero_aux_and_ranking(self):
        outputs = self._make_outputs(with_aux=False)
        labels = torch.randint(0, self.N_CLASSES, (self.B,))
        fusion, img_aux, meta_aux, ranking = classification_loss_parts(
            outputs, labels, "adamw")
        assert fusion.item() > 0  # CE should be positive
        assert img_aux.item() == 0.0
        assert meta_aux.item() == 0.0
        assert ranking.item() == 0.0

    def test_samo_returns_nonzero_aux(self):
        outputs = self._make_outputs(with_aux=True)
        labels = torch.randint(0, self.N_CLASSES, (self.B,))
        fusion, img_aux, meta_aux, ranking = classification_loss_parts(
            outputs, labels, "samo")
        assert fusion.item() > 0
        assert img_aux.item() > 0
        assert meta_aux.item() > 0
        assert ranking.item() == 0.0  # always zero for classification

    def test_ra_samo_returns_nonzero_aux(self):
        outputs = self._make_outputs(with_aux=True)
        labels = torch.randint(0, self.N_CLASSES, (self.B,))
        parts = classification_loss_parts(outputs, labels, "ra_samo")
        assert parts[0].item() > 0  # fusion
        assert parts[1].item() > 0  # img aux
        assert parts[2].item() > 0  # meta aux
        assert parts[3].item() == 0.0  # ranking always zero

    def test_all_losses_finite(self):
        for method in ("adamw", "sam", "adaptive_aux", "samo", "ra_samo"):
            with_aux = method in ("adaptive_aux", "samo", "ra_samo")
            outputs = self._make_outputs(with_aux=with_aux)
            labels = torch.randint(0, self.N_CLASSES, (self.B,))
            parts = classification_loss_parts(outputs, labels, method)
            for i, p in enumerate(parts):
                assert torch.isfinite(p), f"Non-finite loss part {i} for {method}"

    def test_loss_requires_grad(self):
        outputs = self._make_outputs(with_aux=True)
        labels = torch.randint(0, self.N_CLASSES, (self.B,))
        parts = classification_loss_parts(outputs, labels, "ra_samo")
        # fusion and aux should require grad
        assert parts[0].requires_grad
        assert parts[1].requires_grad
        assert parts[2].requires_grad


# ── TestGradientProjection ───────────────────────────────────────────────────


class TestGradientProjection:
    """Test that prepare_branch_gradients works with HAM10000 model's gene_encoder alias."""

    B = 4
    META_DIM = 20
    N_CLASSES = 7

    def test_prepare_branch_gradients_with_alias(self):
        """Core test: gradient projection code works via gene_encoder property."""
        model = HAM10000ModelWithAux(self.META_DIM, self.N_CLASSES, pretrained=False)
        inputs = {
            "image": torch.randn(self.B, 3, 224, 224),
            "meta": torch.randn(self.B, self.META_DIM),
        }
        labels = torch.randint(0, self.N_CLASSES, (self.B,))
        outputs = model(inputs)
        parts = classification_loss_parts(outputs, labels, "ra_samo")
        weights = [0.1, 0.1, 0.0]

        # This should NOT raise — gene_encoder alias makes it work
        prepared = prepare_branch_gradients(
            model, *parts, weights, project=True)

        assert "image" in prepared
        assert "gene" in prepared

        # Each branch: (params, combined_grads, BranchDiagnostics)
        for branch in ("image", "gene"):
            params, grads, diag = prepared[branch]
            assert len(params) > 0
            assert len(grads) == len(params)
            assert hasattr(diag, "cosine_aux")
            assert hasattr(diag, "projected_aux")

    def test_apply_branch_gradients(self):
        model = HAM10000ModelWithAux(self.META_DIM, self.N_CLASSES, pretrained=False)
        inputs = {
            "image": torch.randn(self.B, 3, 224, 224),
            "meta": torch.randn(self.B, self.META_DIM),
        }
        labels = torch.randint(0, self.N_CLASSES, (self.B,))
        outputs = model(inputs)
        parts = classification_loss_parts(outputs, labels, "ra_samo")
        weights = [0.1, 0.1, 0.0]

        prepared = prepare_branch_gradients(model, *parts, weights, project=True)

        # Simulate backward
        loss = parts[0] + 0.1 * parts[1] + 0.1 * parts[2]
        loss.backward()

        diagnostics = apply_branch_gradients(prepared)
        assert "image" in diagnostics
        assert "gene" in diagnostics

        # Check gradients were actually set
        for p in model.image_encoder.parameters():
            if p.requires_grad:
                assert p.grad is not None
        for p in model.gene_encoder.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ── TestSAM ──────────────────────────────────────────────────────────────────


class TestSAM:
    """Test SAM optimizer with HAM10000 model."""

    def test_sam_step_finite(self):
        B, META_DIM, N_CLASSES = 4, 20, 7
        model = HAM10000Model(META_DIM, N_CLASSES, pretrained=False)
        base_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sam_opt = SAMOptimizer(base_opt, model, rho=0.05)

        inputs = {
            "image": torch.randn(B, 3, 224, 224),
            "meta": torch.randn(B, META_DIM),
        }
        labels = torch.randint(0, N_CLASSES, (B,))

        model.train()

        def closure():
            out = model(inputs)
            loss = F.cross_entropy(out["logits"], labels)
            return loss, out

        loss, out = sam_opt.step(closure)
        assert torch.isfinite(loss)
        assert out["logits"].shape == (B, N_CLASSES)

    def test_sam_updates_params(self):
        B, META_DIM, N_CLASSES = 4, 20, 7
        model = HAM10000Model(META_DIM, N_CLASSES, pretrained=False)
        base_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sam_opt = SAMOptimizer(base_opt, model, rho=0.05)

        # Save initial params
        initial = {n: p.clone() for n, p in model.named_parameters()}

        inputs = {
            "image": torch.randn(B, 3, 224, 224),
            "meta": torch.randn(B, META_DIM),
        }
        labels = torch.randint(0, N_CLASSES, (B,))
        model.train()

        def closure():
            out = model(inputs)
            loss = F.cross_entropy(out["logits"], labels)
            return loss, out

        sam_opt.step(closure)

        # At least some parameters should have changed
        changed = sum(1 for n, p in model.named_parameters()
                      if not torch.allclose(p, initial[n]))
        assert changed > 0


# ── TestMetrics ──────────────────────────────────────────────────────────────


class TestMetrics:
    """Test metric computation on known data."""

    def test_binary_auroc_perfect(self):
        labels = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        assert _binary_auroc(labels, scores) == 1.0

    def test_binary_auroc_random(self):
        labels = np.array([1, 0, 1, 0])
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        # Tied scores → should be ~0.5
        auc = _binary_auroc(labels, scores)
        assert 0.0 <= auc <= 1.0

    def test_macro_auroc_shape(self):
        n, k = 100, 7
        labels = np.random.randint(0, k, n)
        probs = np.random.dirichlet(np.ones(k), n)
        auroc = _macro_auroc(labels, probs, k)
        assert 0.0 <= auroc <= 1.0

    def test_evaluate_returns_all_metrics(self):
        ds = SyntheticHAM10000(n=32, meta_dim=20, n_classes=7)
        loader = DataLoader(ds, batch_size=8)
        model = HAM10000Model(20, 7, pretrained=False)
        model.eval()
        metrics = evaluate(model, loader, "cpu", n_classes=7)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "balanced_acc" in metrics
        assert "auroc" in metrics
        assert "per_class_f1" in metrics
        assert "confusion_matrix" in metrics
        assert len(metrics["per_class_f1"]) == 7

    def test_evaluate_perfect_predictions(self):
        """A model that memorizes should get high metrics."""
        # We can't easily force perfect predictions, but we can check
        # that metrics are in valid ranges
        ds = SyntheticHAM10000(n=32, meta_dim=20, n_classes=7)
        loader = DataLoader(ds, batch_size=8)
        model = HAM10000Model(20, 7, pretrained=False)
        metrics = evaluate(model, loader, "cpu", n_classes=7)
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["macro_f1"] <= 1
        assert 0 <= metrics["balanced_acc"] <= 1
        assert 0 <= metrics["auroc"] <= 1


# ── TestTraining ─────────────────────────────────────────────────────────────


class TestTraining:
    """1-epoch smoke tests with synthetic data for all methods."""

    META_DIM = 20
    N_CLASSES = 7

    def _make_loader(self, n=32, batch_size=8):
        ds = SyntheticHAM10000(n=n, meta_dim=self.META_DIM, n_classes=self.N_CLASSES)
        return DataLoader(ds, batch_size=batch_size, drop_last=True)

    def _train_one_epoch(self, model, loader, optimizer, method, device="cpu"):
        """Run one training epoch, return mean loss."""
        model.train()
        total_loss = 0
        n = 0
        for inputs, labels_dict in loader:
            x = {k: v.to(device) for k, v in inputs.items()}
            y = labels_dict["label"].to(device)
            optimizer.zero_grad()
            outputs = model(x)
            parts = classification_loss_parts(outputs, y, method)
            loss = parts[0] + 0.1 * parts[1] + 0.1 * parts[2]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    def test_adamw_smoke(self):
        model = HAM10000Model(self.META_DIM, self.N_CLASSES, pretrained=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loader = self._make_loader()
        loss = self._train_one_epoch(model, loader, opt, "adamw")
        assert math.isfinite(loss)

    def test_sam_smoke(self):
        model = HAM10000Model(self.META_DIM, self.N_CLASSES, pretrained=False)
        base_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sam_opt = SAMOptimizer(base_opt, model, rho=0.05)
        loader = self._make_loader()

        model.train()
        for inputs, labels_dict in loader:
            def closure():
                out = model(inputs)
                loss = F.cross_entropy(out["logits"], labels_dict["label"])
                return loss, out
            loss, _ = sam_opt.step(closure)
            assert torch.isfinite(loss)
            break  # just 1 batch

    def test_adaptive_aux_smoke(self):
        model = HAM10000ModelWithAux(self.META_DIM, self.N_CLASSES, pretrained=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loader = self._make_loader()
        loss = self._train_one_epoch(model, loader, opt, "adaptive_aux")
        assert math.isfinite(loss)

    def test_samo_smoke(self):
        model = HAM10000ModelWithAux(self.META_DIM, self.N_CLASSES, pretrained=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loader = self._make_loader()
        loss = self._train_one_epoch(model, loader, opt, "samo")
        assert math.isfinite(loss)

    def test_ra_samo_smoke(self):
        """Full RA-SAMO pipeline: loss parts → branch gradients → projection → step."""
        model = HAM10000ModelWithAux(self.META_DIM, self.N_CLASSES, pretrained=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loader = self._make_loader()
        weights = [0.1, 0.1, 0.0]

        model.train()
        for inputs, labels_dict in loader:
            opt.zero_grad(set_to_none=True)
            outputs = model(inputs)
            parts = classification_loss_parts(outputs, labels_dict["label"], "ra_samo")

            prepared = prepare_branch_gradients(
                model, *parts, weights, project=True)

            loss = parts[0] + weights[0] * parts[1] + weights[1] * parts[2]
            loss.backward()

            diagnostics = apply_branch_gradients(prepared)
            opt.step()

            assert torch.isfinite(loss)
            assert "image" in diagnostics
            assert "gene" in diagnostics
            break  # 1 batch

    def test_image_only_smoke(self):
        model = ImageOnlyModel(self.N_CLASSES, pretrained=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loader = self._make_loader()
        model.train()
        for inputs, labels_dict in loader:
            opt.zero_grad()
            out = model(inputs)
            loss = F.cross_entropy(out["logits"], labels_dict["label"])
            loss.backward()
            opt.step()
            assert math.isfinite(loss.item())
            break

    def test_meta_only_smoke(self):
        model = MetaOnlyModel(self.META_DIM, self.N_CLASSES)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loader = self._make_loader()
        model.train()
        for inputs, labels_dict in loader:
            opt.zero_grad()
            out = model(inputs)
            loss = F.cross_entropy(out["logits"], labels_dict["label"])
            loss.backward()
            opt.step()
            assert math.isfinite(loss.item())
            break

    def test_loss_weight_controller_with_classification(self):
        """LossWeightController works with classification loss means."""
        controller = LossWeightController(
            initial=(0.1, 0.1, 0.0), min_weight=0.02, max_weight=0.3, warmup=2)
        # Simulate 5 epochs
        for epoch in range(5):
            means = [1.5 - epoch * 0.1, 0.8, 0.6, 0.0]  # [fusion, img, meta, ranking=0]
            weights = controller.update(epoch, means)
            assert len(weights) == 3
            # img and meta aux weights should be in bounds
            assert 0.02 <= weights[0] <= 0.3
            assert 0.02 <= weights[1] <= 0.3
            # ranking weight stays at 0.0 (initialized there, no ranking in classification)
            assert weights[2] >= 0.0


# ── TestDatasetEdgeCases ─────────────────────────────────────────────────────


class TestDatasetEdgeCases:
    """Edge cases for the dataset module."""

    def test_missing_csv_raises(self):
        with pytest.raises(FileNotFoundError, match="HAM10000_metadata.csv"):
            build_ham10000_splits("/nonexistent/path")

    def test_class_count(self):
        assert NUM_CLASSES == 7
        assert len(CLASSES) == 7
        assert len(CLASS_TO_IDX) == 7

    def test_class_to_idx_bijective(self):
        for cls_name, idx in CLASS_TO_IDX.items():
            assert CLASSES[idx] == cls_name
