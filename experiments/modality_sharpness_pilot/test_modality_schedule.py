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

    def test_image_gene_groups_no_overlap(self):
        """Image and gene parameter groups must be disjoint."""
        torch.manual_seed(3)
        model=ToyModel()
        opt=ModalityScheduledSAM(model,ModalitySchedule("periodic",10,0.5,0.5),lr=0.01)
        img_ids={id(p) for p in opt.groups["image"]}
        gene_ids={id(p) for p in opt.groups["gene"]}
        self.assertTrue(len(img_ids)>0)
        self.assertTrue(len(gene_ids)>0)
        self.assertEqual(len(img_ids & gene_ids),0,"Image and gene groups overlap")

    def test_fusion_head_always_receives_gradients(self):
        """Head parameters must get gradients in all 4 states."""
        for img_frac,gene_frac,label in [(1,1,"both"),(1,0,"img_only"),
                                          (0,1,"gene_only"),(0,0,"neither")]:
            torch.manual_seed(4)
            model=ToyModel()
            opt=ModalityScheduledSAM(model,ModalitySchedule("periodic",1,img_frac,gene_frac),
                                      lr=0.01,rho=0.05)
            x=torch.randn(8,2)
            before=[p.detach().clone() for p in model.head.parameters()]
            opt.step(lambda: model(x).square().mean())
            changed=any(not torch.equal(a,b) for a,b
                        in zip(before,[p.detach().clone() for p in model.head.parameters()]))
            self.assertTrue(changed,f"Head not updated in state={label}")

    def test_image_only_does_not_perturb_gene(self):
        """When only image is scheduled, gene encoder weights must not be perturbed."""
        torch.manual_seed(5)
        model=ToyModel()
        schedule=ModalitySchedule("periodic",1,1.0,0.0,seed=1)
        opt=ModalityScheduledSAM(model,schedule,lr=0.0,rho=0.05)
        x=torch.randn(8,2)
        gene_before=[p.detach().clone() for p in model.gene_encoder.parameters()]
        gene_during_perturbation=[]
        call_count=[0]
        orig_forward=model.forward
        def patched_forward(inp):
            call_count[0]+=1
            if call_count[0]==2:
                gene_during_perturbation.extend(
                    [p.detach().clone() for p in model.gene_encoder.parameters()])
            return orig_forward(inp)
        model.forward=patched_forward
        opt.step(lambda: model(x).square().mean())
        for gb,gd in zip(gene_before,gene_during_perturbation):
            self.assertTrue(torch.equal(gb,gd),
                            "Gene encoder was perturbed during image-only step")

    def test_modality_schedules_use_different_seeds(self):
        """Image and gene schedules must differ (seed offset by 7919)."""
        ms=ModalitySchedule("random",100,0.5,0.5,seed=42)
        self.assertNotEqual(ms.image.selected,ms.gene.selected,
                            "Image and gene schedules are identical — seed offset missing")


if __name__=="__main__":
    unittest.main()
