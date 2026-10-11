"""Isolated blockwise SAM experiments; does not modify legacy optimizers."""
import math
import torch

class BlockSAM:
    """Two-pass SAM; image/gene perturbations have independent radii.

    Adaptive mode estimates per-block loss increase under a probe perturbation
    every probe_every steps, then adjusts radii inversely to estimated sensitivity.
    This is an experimental heuristic, not an established sharpness estimator.
    """
    def __init__(self, model, lr=1e-3, weight_decay=1e-4,
                 rho_image=0.05, rho_gene=0.05, adaptive=False,
                 probe_every=20, ema=0.9, min_scale=0.5, max_scale=2.0):
        self.model=model
        self.base=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=weight_decay)
        self.rhos={"image":float(rho_image),"gene":float(rho_gene)}
        self.adaptive=adaptive
        self.probe_every=int(probe_every)
        self.ema=float(ema)
        self.min_scale=float(min_scale)
        self.max_scale=float(max_scale)
        self.sensitivity={}
        self.steps=0
        self.history=[]
        if self.probe_every<1: raise ValueError("probe_every must be >=1")

    def groups(self):
        return {"image":list(self.model.image_encoder.parameters()),
                "gene":list(self.model.gene_encoder.parameters())}

    @staticmethod
    def perturb(params,rho):
        grads=[p.grad for p in params if p.grad is not None]
        if not grads: return []
        norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach().float()) for g in grads]))
        if not torch.isfinite(norm) or norm.item()<1e-12: return []
        changed=[]
        with torch.no_grad():
            for p in params:
                if p.grad is None: continue
                e=(p.grad.detach() * (rho/(norm+1e-12))).clone()
                p.add_(e)
                changed.append((p,e))
        return changed

    @staticmethod
    def restore(changed):
        with torch.no_grad():
            for p,e in changed: p.sub_(e)

    def step(self,closure):
        self.steps+=1
        self.base.zero_grad(set_to_none=True)
        loss=closure()
        loss.backward()
        groups=self.groups()
        radii=dict(self.rhos)
        if self.adaptive and self.steps%self.probe_every==1:
            baseline=float(loss.detach())
            measured={}
            # Evaluate sensitivity with independent probes, using the SAME batch.
            for name,params in groups.items():
                changed=self.perturb(params,self.rhos[name])
                try:
                    with torch.no_grad():
                        probe=float(closure().detach())
                finally:
                    self.restore(changed)
                measured[name]=max(0.0,probe-baseline)
                old=self.sensitivity.get(name,measured[name])
                self.sensitivity[name]=self.ema*old+(1-self.ema)*measured[name]
            mean_s=sum(self.sensitivity.values())/max(len(self.sensitivity),1)
            for name in groups:
                scale=math.sqrt((mean_s+1e-8)/(self.sensitivity[name]+1e-8))
                radii[name]=self.rhos[name]*min(self.max_scale,max(self.min_scale,scale))
            self._last_radii=radii
        elif self.adaptive:
            radii=getattr(self,"_last_radii",radii)
        changed=[]
        try:
            for name,params in groups.items():
                changed.extend(self.perturb(params,radii[name]))
            self.base.zero_grad(set_to_none=True)
            second=closure()
            second.backward()
        finally:
            self.restore(changed)
        self.base.step()
        self.history.append({"step":self.steps,"rho_image":radii["image"],
                             "rho_gene":radii["gene"],
                             "sensitivity":dict(self.sensitivity)})
        return second.detach()
