import unittest
import torch
from experiments.samo.objectives import cox_efron_free, ranking_loss, comparable_pairs, LossWeightController

class ObjectiveTests(unittest.TestCase):
    def test_pairs_and_censoring(self):
        t=torch.tensor([1.,2.,3.,1.])
        e=torch.tensor([1.,0.,1.,0.])
        mask=comparable_pairs(t,e)
        self.assertEqual(int(mask.sum()),2)
        self.assertTrue(bool(mask[0,1] and mask[0,2]))
        self.assertFalse(bool(mask[3].any()))
    def test_zero_events(self):
        x=torch.randn(4,requires_grad=True)
        t=torch.arange(4.).float()
        e=torch.zeros(4)
        loss=cox_efron_free(x,t,e)+ranking_loss(x,t,e)
        loss.backward()
        self.assertTrue(torch.equal(x.grad,torch.zeros_like(x)))
    def test_gradients(self):
        x=torch.tensor([0.1,0.2,0.3],requires_grad=True)
        t=torch.tensor([1.,2.,3.]);e=torch.ones(3)
        loss=cox_efron_free(x,t,e)+ranking_loss(x,t,e)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()),0)
    def test_tied_times(self):
        t=torch.tensor([1.,1.,2.])
        e=torch.tensor([1.,1.,0.])
        self.assertEqual(int(comparable_pairs(t,e).sum()),2)
    def test_controller_bounds_and_restore(self):
        c=LossWeightController()
        for i in range(100):c.update(i,[1.,0.5,1.,2.])
        self.assertTrue(all(0.02<=w<=0.3 for w in c.weights))
        other=LossWeightController()
        other.load_state_dict(c.state_dict())
        self.assertEqual(c.weights,other.weights)

if __name__=="__main__":unittest.main()
