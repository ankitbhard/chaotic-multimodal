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
    def test_probe_every_one(self):
        torch.manual_seed(4)
        model=Tiny()
        opt=BlockSAM(model,lr=0.01,adaptive=True,probe_every=1)
        a=torch.randn(5,3); b=torch.randn(5,2)
        def closure(): return model(a,b).square().mean()
        opt.step(closure)
        self.assertEqual(len(opt.sensitivity),2)
        self.assertEqual(opt.history[0]["step"],1)
        self.assertAlmostEqual(
            (opt.history[0]["rho_image"]**2+opt.history[0]["rho_gene"]**2)**0.5,
            0.05,places=5)

    def test_dropout_probe_reproducibility(self):
        class DropTiny(Tiny):
            def forward(self,a,b):
                return self.head(torch.nn.functional.dropout(
                    self.image_encoder(a)+self.gene_encoder(b),
                    p=0.5,training=self.training))
        torch.manual_seed(8)
        model=DropTiny()
        opt=BlockSAM(model,adaptive=True,probe_every=1,lr=0.01)
        a=torch.randn(8,3); b=torch.randn(8,2)
        def closure(): return model(a,b).square().mean()
        opt.step(closure)
        self.assertTrue(all(v>=0 for v in opt.sensitivity.values()))

    def test_quadratic_directional_curvature(self):
        torch.manual_seed(11)
        model=Tiny()
        opt=BlockSAM(model,lr=0.0,curvature_diagnostics=True,probe_every=1)
        a=torch.randn(5,3); b=torch.randn(5,2)
        def closure(): return model(a,b).square().mean()
        before=[p.detach().clone() for p in model.parameters()]
        opt.step(closure)
        self.assertEqual(len(opt.curvature_history),1)
        for entry in opt.curvature_history[0]["blocks"].values():
            self.assertEqual(entry["status"],"ok")
            self.assertAlmostEqual(entry["slope_estimate"],
                                   entry["first_order_slope"],delta=0.005)
            self.assertGreaterEqual(entry["directional_curvature"],-0.005)
        for x,y in zip(before,model.parameters()):
            self.assertTrue(torch.allclose(x,y,atol=1e-6))

    def test_trajectory_invariance_with_diagnostics(self):
        """Diagnostics must not alter the optimization trajectory."""
        torch.manual_seed(42)
        model_a=Tiny(); model_b=Tiny()
        model_b.load_state_dict(model_a.state_dict())
        a=torch.randn(6,3); b=torch.randn(6,2); y=torch.randn(6,1)
        opt_a=BlockSAM(model_a,lr=0.01,adaptive=True,probe_every=2,
                        curvature_diagnostics=False)
        opt_b=BlockSAM(model_b,lr=0.01,adaptive=True,probe_every=2,
                        curvature_diagnostics=True)
        for _ in range(6):
            torch.manual_seed(99)
            opt_a.step(lambda: (model_a(a,b)-y).square().mean())
            torch.manual_seed(99)
            opt_b.step(lambda: (model_b(a,b)-y).square().mean())
        for pa,pb in zip(model_a.parameters(),model_b.parameters()):
            self.assertTrue(torch.allclose(pa,pb,atol=1e-6),
                            "Diagnostics changed the trajectory")

    def test_exact_parameter_restoration(self):
        """After curvature probe, params must match pre-probe values."""
        torch.manual_seed(55)
        model=Tiny()
        opt=BlockSAM(model,lr=0.0,curvature_diagnostics=True,probe_every=1)
        a=torch.randn(5,3); b=torch.randn(5,2)
        def closure(): return model(a,b).square().mean()
        before=[p.detach().clone() for p in model.parameters()]
        opt.step(closure)
        for x,y in zip(before,model.parameters()):
            diff=torch.max(torch.abs(x-y)).item()
            self.assertLess(diff,1e-6,f"Restoration error {diff}")

    def test_dropout_determinism_across_probes(self):
        """All three evaluations (baseline, plus, minus) must see the same dropout."""
        class DropTiny(Tiny):
            def __init__(self):
                super().__init__()
                self.calls=[]
            def forward(self,a,b):
                out=torch.nn.functional.dropout(
                    self.image_encoder(a)+self.gene_encoder(b),
                    p=0.5,training=self.training)
                self.calls.append(out.detach().clone())
                return self.head(out)
        torch.manual_seed(77)
        model=DropTiny()
        opt=BlockSAM(model,lr=0.0,curvature_diagnostics=True,probe_every=1)
        a=torch.randn(4,3); b=torch.randn(4,2)
        def closure(): return model(a,b).square().mean()
        opt.step(closure)
        baseline_mask=(model.calls[0]!=0).float()
        for i in range(1,5):
            probe_mask=(model.calls[i]!=0).float()
            self.assertTrue(torch.equal(baseline_mask,probe_mask),
                            f"Dropout mask mismatch: call 0 vs call {i}")

    def test_analytic_quadratic_curvature(self):
        """On L = ||f(x)||², curvature along grad direction is non-negative (convex)."""
        torch.manual_seed(13)
        model=Tiny()
        opt=BlockSAM(model,lr=0.0,curvature_diagnostics=True,probe_every=1,
                     rho_image=0.01,rho_gene=0.01)
        x=torch.randn(10,3); g=torch.randn(10,2)
        def closure(): return model(x,g).square().mean()
        before=[p.detach().clone() for p in model.parameters()]
        opt.step(closure)
        diag=opt.curvature_history[0]["blocks"]
        for name in ("image","gene"):
            self.assertEqual(diag[name]["status"],"ok")
            self.assertGreater(diag[name]["directional_curvature"],0.0,
                               f"{name} curvature should be positive for convex quadratic")
            self.assertAlmostEqual(diag[name]["slope_estimate"],
                                   diag[name]["first_order_slope"],delta=0.01)
        for a,b in zip(before,model.parameters()):
            self.assertTrue(torch.allclose(a,b,atol=1e-6))

    def test_nonfinite_handling(self):
        """Non-finite closure values should not crash."""
        torch.manual_seed(9)
        model=Tiny()
        opt=BlockSAM(model,lr=0.0,curvature_diagnostics=True,probe_every=1)
        call_count=[0]
        a=torch.randn(4,3); b=torch.randn(4,2)
        def closure():
            call_count[0]+=1
            out=model(a,b).square().mean()
            if call_count[0]==3: return out+float('inf')
            return out
        opt.step(closure)
        self.assertEqual(len(opt.curvature_history),1)

    def test_fixed(self): self.run_case(False)
    def test_adaptive(self): self.run_case(True)

if __name__=="__main__":unittest.main()
