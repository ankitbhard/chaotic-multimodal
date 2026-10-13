"""CPU tests for equal-budget scheduled SAM."""
import unittest
import torch
from experiments.modality_sharpness_pilot.chaotic_schedule import SAMSchedule, ScheduledSAM


class TestSchedule(unittest.TestCase):
    def test_exact_budget_and_reproducibility(self):
        for mode in ("periodic","random","chaotic"):
            a=SAMSchedule(mode,101,0.4,seed=17)
            b=SAMSchedule(mode,101,0.4,seed=17)
            self.assertEqual(a.selected,b.selected)
            self.assertEqual(len(a.selected),40)
            self.assertEqual(sum(a.use_sam(i) for i in range(101)),40)

    def test_extreme_budgets(self):
        for mode in ("periodic","random","chaotic"):
            self.assertEqual(len(SAMSchedule(mode,20,0.0).selected),0)
            self.assertEqual(len(SAMSchedule(mode,20,1.0).selected),20)

    def test_step_count_and_finite_weights(self):
        for mode in ("periodic","random","chaotic"):
            torch.manual_seed(9)
            model=torch.nn.Linear(2,1)
            schedule=SAMSchedule(mode,10,0.5,seed=3)
            opt=ScheduledSAM(model,schedule,lr=0.01,rho=0.05)
            x=torch.randn(5,2)
            for _ in range(10):
                opt.step(lambda: model(x).square().mean())
            self.assertEqual(opt.sam_steps,5)
            self.assertEqual(opt.steps,10)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_zero_rho_matches_sgd(self):
        torch.manual_seed(5)
        a=torch.nn.Linear(3,1);b=torch.nn.Linear(3,1)
        b.load_state_dict(a.state_dict())
        x=torch.randn(7,3)
        oa=ScheduledSAM(a,SAMSchedule("periodic",6,0.0),lr=0.01,rho=0.0)
        ob=ScheduledSAM(b,SAMSchedule("periodic",6,1.0),lr=0.01,rho=0.0)
        for _ in range(6):
            oa.step(lambda: a(x).square().mean())
            ob.step(lambda: b(x).square().mean())
        for pa,pb in zip(a.parameters(),b.parameters()):
            self.assertTrue(torch.allclose(pa,pb,atol=1e-7))


if __name__=="__main__":
    unittest.main()
