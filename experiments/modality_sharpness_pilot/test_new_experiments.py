"""Smoke tests for experimental optimizer and ROI feature selection."""
import unittest
import numpy as np
import torch
from experiments.modality_sharpness_pilot.new_optimizer_prototypes import (
    OptimizerExperiment,install_shared_gradient_hooks)
from experiments.modality_sharpness_pilot.roi_patch_pilot import select_roi


class ToyFusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.img_proj=torch.nn.Linear(3,4)
        self.gene_proj=torch.nn.Linear(3,4)
    def forward(self,x):
        return self.img_proj(x)+self.gene_proj(x)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion=ToyFusion()
    def forward(self,x):
        return self.fusion(x)


class TestNewExperiments(unittest.TestCase):
    def test_optimizers_finite(self):
        for mode in ("sgd","adamw","adaptive_step","conflict_projection"):
            torch.manual_seed(1)
            model=Toy()
            if mode=="conflict_projection":
                install_shared_gradient_hooks(model)
            opt=OptimizerExperiment(model,mode,lr=.01)
            x=torch.randn(8,3)
            for _ in range(4):
                opt.step(lambda:model(x).square().mean())
            self.assertEqual(opt.steps,4)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
            if mode=="conflict_projection":
                self.assertIsNotNone(opt.history[-1]["gradient_cosine"])

    def test_roi(self):
        x=np.random.default_rng(4).normal(size=(25,8)).astype("float32")
        idx,pooled=select_roi(x,k=10)
        self.assertEqual(len(set(idx)),10)
        np.testing.assert_allclose(pooled,x[idx].mean(0))
        self.assertRaises(ValueError,select_roi,np.zeros((0,8)))


if __name__=="__main__": unittest.main()
