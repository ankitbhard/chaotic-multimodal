"""CPU-only checks for RA-SAMO encoder gradient replacement."""
import unittest
import torch
from torch import nn
from experiments.samo.representation_gradients import (
    prepare_branch_gradients, apply_branch_gradients,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Linear(3, 2, bias=False)
        self.gene_encoder = nn.Linear(3, 2, bias=False)
        self.fusion = nn.Linear(4, 1, bias=False)
        self.aux_img = nn.Linear(2, 1, bias=False)
        self.aux_gene = nn.Linear(2, 1, bias=False)

    def forward(self, x):
        i = self.image_encoder(x)
        g = self.gene_encoder(x)
        fused = self.fusion(torch.cat((i, g), dim=1)).mean()
        return fused, self.aux_img(i).mean(), self.aux_gene(g).mean()


class RepresentationGradientTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = TinyModel()
        self.x = torch.randn(6, 3)
        self.weights = (0.2, 0.3, 0.1)

    def _losses(self):
        fused, img, gene = self.model(self.x)
        return fused.square(), img.square(), gene.square(), (fused - 1).square()

    def test_no_projection_matches_normal_backward(self):
        parts = self._losses()
        loss = parts[0] + sum(w * x for w, x in zip(self.weights, parts[1:]))
        self.model.zero_grad(set_to_none=True)
        loss.backward()
        expected = {name: p.grad.clone() for name, p in self.model.named_parameters()}
        self.model.zero_grad(set_to_none=True)
        parts = self._losses()
        prepared = prepare_branch_gradients(self.model, *parts, self.weights, project=False)
        (parts[0] + sum(w * x for w, x in zip(self.weights, parts[1:]))).backward()
        apply_branch_gradients(prepared)
        for name, p in self.model.named_parameters():
            torch.testing.assert_close(p.grad, expected[name], atol=1e-6, rtol=1e-5)

    def test_projection_preserves_nonencoder_gradients(self):
        parts = self._losses()
        prepared = prepare_branch_gradients(self.model, *parts, self.weights, project=True)
        total = parts[0] + sum(w * x for w, x in zip(self.weights, parts[1:]))
        total.backward()
        shared_before = self.model.fusion.weight.grad.clone()
        aux_before = self.model.aux_img.weight.grad.clone()
        diagnostics = apply_branch_gradients(prepared)
        torch.testing.assert_close(self.model.fusion.weight.grad, shared_before)
        torch.testing.assert_close(self.model.aux_img.weight.grad, aux_before)
        self.assertEqual(set(diagnostics), {"image", "gene"})

    def test_conflicting_gradient_is_projected(self):
        from experiments.samo.representation_gradients import _project, _dot
        primary = [torch.tensor([1., 0.])]
        aux = [torch.tensor([-2., 3.])]
        projected, flag = _project(aux, primary)
        self.assertTrue(flag)
        self.assertAlmostEqual(float(_dot(projected, primary)), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
