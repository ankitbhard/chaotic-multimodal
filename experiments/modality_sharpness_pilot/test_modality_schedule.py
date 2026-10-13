"""Tests for independent image/gene SAM perturbation schedules."""
import unittest
import torch
from experiments.modality_sharpness_pilot.modality_schedule import (
    ModalitySchedule,ModalityScheduledSAM)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder=torch.nn.Linear(2,2)
        self.gene_encoder=torch.nn.Linear(2,2)
        self.head=torch.nn.Linear(4,1)
    def forward(self,x):
        return self.head(torch.cat((self.image_encoder(x),self.gene_encoder(x)),dim=1))


class TestModalitySchedule(unittest.TestCase):
    def test_exact_counts_and_reproducibility(self):
        for mode in ("periodic","random","chaotic"):
            a=ModalitySchedule(mode,100,0.4,0.6,seed=9)
            b=ModalitySchedule(mode,100,0.4,0.6,seed=9)
            self.assertEqual(a.counts,b.counts)
            self.assertEqual(a.counts["image_only"]+a.counts["both"],40)
            self.assertEqual(a.counts["gene_only"]+a.counts["both"],60)
            self.assertEqual(sum(a.counts.values()),100)
            self.assertEqual(a.sam_budget,100-a.counts["neither"])

    def test_all_four_states_and_updates(self):
        torch.manual_seed(7)
        model=ToyModel()
        schedule=ModalitySchedule("random",60,0.5,0.5,seed=11)
        self.assertTrue(all(v>0 for v in schedule.counts.values()))
        opt=ModalityScheduledSAM(model,schedule,lr=0.01,rho=0.05)
        x=torch.randn(8,2)
        for _ in range(60):
            opt.step(lambda: model(x).square().mean())
        self.assertEqual(opt.sam_steps,schedule.sam_budget)
        self.assertEqual(opt.steps,60)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_no_perturbation_matches_sgd(self):
        torch.manual_seed(2)
        a=ToyModel();b=ToyModel()
        b.load_state_dict(a.state_dict())
        x=torch.randn(8,2)
        oa=ModalityScheduledSAM(a,ModalitySchedule("periodic",8,0,0),lr=0.01)
        ob=torch.optim.SGD(b.parameters(),lr=0.01,momentum=0.9)
        for _ in range(8):
            oa.step(lambda: a(x).square().mean())
            ob.zero_grad(set_to_none=True)
            b(x).square().mean().backward()
            ob.step()
        for p,q in zip(a.parameters(),b.parameters()):
            self.assertTrue(torch.allclose(p,q,atol=1e-7))


if __name__=="__main__":
    unittest.main()
