"""Experimental modality-specific SAM perturbation scheduling.

Image and gene encoder perturbations can be enabled independently. Fusion/head
parameters are always updated by the joint Cox loss, but are never perturbed.
A two-pass SAM step is required whenever EITHER modality is selected; selecting
only one modality does not save a backward pass. Compare matched union budgets.
"""
import math
import torch
from experiments.modality_sharpness_pilot.chaotic_schedule import SAMSchedule


class ModalitySchedule:
    """Independent reproducible selections; per-modality quotas are exact."""
    def __init__(self, mode, total_steps, image_fraction=0.5, gene_fraction=0.5,
                 seed=42):
        if mode not in ("periodic","random","chaotic"):
            raise ValueError(mode)
        self.image=SAMSchedule(mode,total_steps,image_fraction,seed=seed)
        self.gene=SAMSchedule(mode,total_steps,gene_fraction,seed=seed+7919)
        self.total_steps=total_steps
        self.counts={
            "image_only":len(self.image.selected-self.gene.selected),
            "gene_only":len(self.gene.selected-self.image.selected),
            "both":len(self.image.selected & self.gene.selected),
            "neither":total_steps-len(self.image.selected | self.gene.selected),
        }
        self.sam_budget=total_steps-self.counts["neither"]

    def state(self,step):
        return (self.image.use_sam(step),self.gene.use_sam(step))


class ModalityScheduledSAM:
    """Two-pass block SAM for selected encoder(s), SGD for neither.

    The entire model is differentiated and updated on every step; 'SGD' for an
    unselected modality means NO SAM perturbation, not a separate first-pass
    optimizer update. All parameters receive the second-pass gradient if either
    encoder is selected. This is deliberately NOT equivalent to blockwise SGD
    plus SAM on disjoint blocks.
    """
    def __init__(self,model,schedule,lr=1e-3,weight_decay=0.0,rho=0.05):
        if rho<0: raise ValueError("rho must be nonnegative")
        self.model=model
        self.schedule=schedule
        self.rho=float(rho)
        self.base=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,
                                  weight_decay=weight_decay)
        self.steps=0
        self.sam_steps=0
        self.history=[]
        self.groups={
            "image":list(model.image_encoder.parameters()),
            "gene":list(model.gene_encoder.parameters()),
        }
        image_ids={id(p) for p in self.groups["image"]}
        gene_ids={id(p) for p in self.groups["gene"]}
        if image_ids & gene_ids:
            raise ValueError("Image and gene encoder parameter blocks overlap")

    def step(self,closure):
        image_on,gene_on=self.schedule.state(self.steps)
        self.base.zero_grad(set_to_none=True)
        loss=closure()
        loss.backward()
        changed=[]
        if image_on or gene_on:
            selected=(self.groups["image"] if image_on else [])+(
                self.groups["gene"] if gene_on else [])
            grads=[p.grad for p in selected if p.grad is not None]
            norm=(torch.linalg.vector_norm(torch.stack([
                torch.linalg.vector_norm(g.detach().float()) for g in grads]))
                  if grads else None)
            try:
                if norm is not None and bool(torch.isfinite(norm)) and norm.item()>1e-12:
                    with torch.no_grad():
                        for p in selected:
                            if p.grad is not None:
                                e=(self.rho*p.grad.detach()/norm).clone()
                                p.add_(e)
                                changed.append((p,e))
                self.base.zero_grad(set_to_none=True)
                loss=closure()
                loss.backward()
            finally:
                with torch.no_grad():
                    for p,e in changed:
                        p.sub_(e)
            self.sam_steps+=1
        self.base.step()
        self.steps+=1
        state=("both" if image_on and gene_on else "image_only" if image_on
               else "gene_only" if gene_on else "neither")
        self.history.append({"step":self.steps,"state":state,"sam":image_on or gene_on,
                             "image_perturbed":image_on,"gene_perturbed":gene_on,
                             "lr":float(self.base.param_groups[0]["lr"])})
        return loss.detach()
