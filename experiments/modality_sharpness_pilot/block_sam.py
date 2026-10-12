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
                 probe_every=20, ema=0.9, min_scale=0.5, max_scale=2.0,
                 match_global_budget=True, curvature_diagnostics=False):
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
        self.match_global_budget=bool(match_global_budget)
        self.curvature_diagnostics=bool(curvature_diagnostics)
        self.curvature_history=[]
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

    @staticmethod
    def _rng_state():
        return (torch.get_rng_state().clone(),
                [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None)

    @staticmethod
    def _set_rng_state(state):
        torch.set_rng_state(state[0])
        if state[1] is not None:
            torch.cuda.set_rng_state_all(state[1])

    def diagnose_curvature(self, closure, groups, baseline, rng_before, rng_after):
        """Same-batch central differences along each block's normalized gradient.

        Diagnostic only: never alters gradients, optimizer state, or the radius
        controller. Restores parameters and RNG after each forward probe.
        """
        diagnostics={}
        for name,params in groups.items():
            grad_sq=sum(float(p.grad.detach().float().square().sum())
                        for p in params if p.grad is not None)
            grad_norm=math.sqrt(grad_sq)
            rho=self.rhos[name]
            if grad_norm<1e-12 or rho<=0:
                diagnostics[name]={"gradient_norm":grad_norm,
                                   "status":"no_direction"}
                continue
            values={}
            for sign,label in ((1,"plus"),(-1,"minus")):
                changed=self.perturb(params,sign*rho)
                try:
                    self._set_rng_state(rng_before)
                    with torch.no_grad():
                        values[label]=float(closure().detach())
                finally:
                    self.restore(changed)
                    self._set_rng_state(rng_after)
            plus=values["plus"]-baseline
            minus=values["minus"]-baseline
            diagnostics[name]={
                "status":"ok","gradient_norm":grad_norm,"rho":rho,
                "baseline_loss":baseline,"loss_plus":values["plus"],
                "loss_minus":values["minus"],
                "delta_plus":plus,"delta_minus":minus,
                "directional_curvature":(plus+minus)/(rho*rho),
                "slope_estimate":(plus-minus)/(2*rho),
                "first_order_slope":grad_norm}
        self.curvature_history.append({"step":self.steps,"blocks":diagnostics})
        return diagnostics

    def step(self,closure):
        self.steps+=1
        self.base.zero_grad(set_to_none=True)
        rng_before=self._rng_state() if (self.adaptive or self.curvature_diagnostics) else None
        loss=closure()
        rng_after=self._rng_state() if (self.adaptive or self.curvature_diagnostics) else None
        loss.backward()
        groups=self.groups()
        radii=dict(self.rhos)
        if self.curvature_diagnostics and (self.steps-1)%self.probe_every==0:
            self.diagnose_curvature(closure,groups,float(loss.detach()),
                                    rng_before,rng_after)
        if self.adaptive and (self.steps-1)%self.probe_every==0:
            baseline=float(loss.detach())
            measured={}
            # Evaluate sensitivity with independent probes, using the SAME batch.
            for name,params in groups.items():
                changed=self.perturb(params,self.rhos[name])
                try:
                    self._set_rng_state(rng_before)
                    with torch.no_grad():
                        probe=float(closure().detach())
                finally:
                    self.restore(changed)
                    self._set_rng_state(rng_after)
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
        # Match the global L2 budget for equal base radii across two blocks.
        if self.match_global_budget:
            budget=math.sqrt(sum(r*r for r in radii.values()))
            factor=max(self.rhos.values())/budget if budget>0 else 1.0
            radii={name:r*factor for name,r in radii.items()}
        grad_norms={}
        for name,params in groups.items():
            grad_norms[name]=math.sqrt(sum(float(p.grad.detach().float().square().sum())
                for p in params if p.grad is not None))
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
                             "sensitivity":dict(self.sensitivity),
                             "grad_norms":grad_norms,
                             "lr":float(self.base.param_groups[0]["lr"])})
        return second.detach()
