"""Tests for block Hessian-vector curvature and diagnostic isolation."""
import unittest
import torch
from experiments.modality_sharpness_pilot.hessian_diagnostics import block_hvp_curvatures
from experiments.modality_sharpness_pilot.block_sam import BlockSAM


class Quadratic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder=torch.nn.Linear(1,1,bias=False,dtype=torch.float64)
        self.gene_encoder=torch.nn.Linear(1,1,bias=False,dtype=torch.float64)
        with torch.no_grad():
            self.image_encoder.weight.fill_(2.0)
            self.gene_encoder.weight.fill_(3.0)

    def forward(self):
        x=self.image_encoder.weight.flatten()[0]
        y=self.gene_encoder.weight.flatten()[0]
        return 0.5*4*x*x - 0.5*2*y*y + x*y


class TestHVP(unittest.TestCase):
    def test_known_block_curvature(self):
        m=Quadratic()
        groups={"image":list(m.image_encoder.parameters()),
                "gene":list(m.gene_encoder.parameters())}
        result=block_hvp_curvatures(m.forward,groups)
        self.assertAlmostEqual(result["image"]["directional_curvature_hvp"],4.0,places=10)
        self.assertAlmostEqual(result["gene"]["directional_curvature_hvp"],-2.0,places=10)

    def test_does_not_change_gradients(self):
        m=Quadratic()
        m.forward().backward()
        grads=[p.grad.clone() for p in m.parameters()]
        weights=[p.detach().clone() for p in m.parameters()]
        groups={"image":list(m.image_encoder.parameters()),
                "gene":list(m.gene_encoder.parameters())}
        block_hvp_curvatures(m.forward,groups)
        for p,g,w in zip(m.parameters(),grads,weights):
            self.assertTrue(torch.equal(p.grad,g))
            self.assertTrue(torch.equal(p,w))

    def test_hvp_diagnostic_does_not_change_step(self):
        torch.manual_seed(3)
        a=Quadratic();b=Quadratic()
        b.load_state_dict(a.state_dict())
        oa=BlockSAM(a,lr=0.01,adaptive=True,probe_every=1)
        ob=BlockSAM(b,lr=0.01,adaptive=True,probe_every=1,
                    hvp_diagnostics=True,curvature_diagnostics=True)
        for _ in range(3):
            oa.step(a.forward)
            ob.step(b.forward)
        for pa,pb in zip(a.parameters(),b.parameters()):
            self.assertTrue(torch.allclose(pa,pb,atol=1e-10,rtol=1e-10))
        self.assertEqual(len(ob.hvp_history),3)
        self.assertEqual(len(ob.curvature_history),3)


if __name__=="__main__":
    unittest.main()
