"""CPU smoke tests for isolated blockwise SAM."""
import unittest
import torch
from experiments.modality_sharpness_pilot.block_sam import BlockSAM

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder=torch.nn.Linear(3,4)
        self.gene_encoder=torch.nn.Linear(2,4)
        self.head=torch.nn.Linear(4,1)
    def forward(self,a,b):
        return self.head(self.image_encoder(a)+self.gene_encoder(b))

class TestBlockSAM(unittest.TestCase):
    def run_case(self,adaptive):
        torch.manual_seed(7)
        model=Tiny()
        opt=BlockSAM(model,lr=0.01,adaptive=adaptive,probe_every=2)
        x=torch.randn(6,3);g=torch.randn(6,2);y=torch.randn(6,1)
        before=[p.detach().clone() for p in model.parameters()]
        def closure(): return (model(x,g)-y).square().mean()
        for _ in range(4):
            loss=opt.step(closure)
            self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.parameters())))
        self.assertEqual(len(opt.history),4)
        if adaptive:
            self.assertIn("image",opt.sensitivity)
            self.assertIn("gene",opt.sensitivity)
    def test_fixed(self): self.run_case(False)
    def test_adaptive(self): self.run_case(True)

if __name__=="__main__":unittest.main()
