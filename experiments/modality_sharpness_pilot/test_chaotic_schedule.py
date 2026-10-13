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

    def test_parameter_restoration_after_sam(self):
        """After SAM perturbation, parameters must be restored before SGD update."""
        torch.manual_seed(10)
        model=torch.nn.Linear(4,1)
        schedule=SAMSchedule("periodic",1,1.0,seed=1)
        opt=ScheduledSAM(model,schedule,lr=0.01,rho=0.05)
        x=torch.randn(8,4)
        before=[p.detach().clone() for p in model.parameters()]
        opt.step(lambda: model(x).square().mean())
        self.assertTrue(any(not torch.equal(a,b) for a,b
                            in zip(before,[p.detach() for p in model.parameters()])))
        for p in model.parameters():
            self.assertTrue(torch.all(torch.isfinite(p)))

    def test_fraction_one_trajectory_reproducible(self):
        """Two identical fraction=1.0 runs produce identical trajectories."""
        torch.manual_seed(20)
        a=torch.nn.Linear(3,1);b=torch.nn.Linear(3,1)
        b.load_state_dict(a.state_dict())
        x=torch.randn(8,3)
        oa=ScheduledSAM(a,SAMSchedule("periodic",5,1.0),lr=0.01,rho=0.05)
        ob=ScheduledSAM(b,SAMSchedule("periodic",5,1.0),lr=0.01,rho=0.05)
        for _ in range(5):
            oa.step(lambda: a(x).square().mean())
            ob.step(lambda: b(x).square().mean())
        for pa,pb in zip(a.parameters(),b.parameters()):
            self.assertTrue(torch.allclose(pa,pb,atol=1e-7))

    def test_different_seeds_different_schedules(self):
        """Different seeds produce different schedules."""
        a=SAMSchedule("random",50,0.4,seed=42)
        b=SAMSchedule("random",50,0.4,seed=99)
        self.assertNotEqual(a.selected,b.selected)
        c=SAMSchedule("chaotic",50,0.4,seed=42)
        d=SAMSchedule("chaotic",50,0.4,seed=42)
        self.assertEqual(c.selected,d.selected)


if __name__=="__main__":
    unittest.main()
