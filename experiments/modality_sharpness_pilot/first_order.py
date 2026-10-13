"""Experimental first-order optimizers; no SAM or curvature probes.

These are ablation candidates, not implementations of RGM/DGL/Polyak-SAM.
"""
import torch


class GradientCentralizedAdamW(torch.optim.AdamW):
    """AdamW with gradient centralization for tensors with ndim > 1."""
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and p.grad.ndim>1 and not p.grad.is_sparse:
                    axes=tuple(range(1,p.grad.ndim))
                    p.grad.sub_(p.grad.mean(dim=axes,keepdim=True))
        return super().step(closure)


class CautiousAdamW(torch.optim.AdamW):
    """AdamW variant: mask moment updates that oppose current gradient.

    Research ablation; not a reproduction of a particular published method.
    Weight decay is handled by AdamW; step-size uses standard AdamW state.
    """
    @torch.no_grad()
    def step(self,closure=None):
        # Use AdamW's implementation with a temporary gradient mask based on
        # previous EMA; restores gradients after the step.
        saved=[]
        for group in self.param_groups:
            beta1=group["betas"][0]
            for p in group["params"]:
                if p.grad is None: continue
                state=self.state[p]
                ema=state.get("exp_avg")
                if ema is None: continue
                g=p.grad
                direction=beta1*ema+(1-beta1)*g
                mask=(direction*g>0).to(g.dtype)
                # Normalize so masking does not automatically shrink LR.
                scale=mask.numel()/mask.sum().clamp_min(1)
                saved.append((p,g.clone()))
                g.mul_(mask*scale)
        try:
            return super().step(closure)
        finally:
            for p,g in saved: p.grad.copy_(g)


def make_first_order(name,model,lr,weight_decay):
    if name=="adamw":
        return torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    if name=="sgd_momentum":
        return torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=weight_decay)
    if name=="grad_central_adamw":
        return GradientCentralizedAdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    if name=="cautious_adamw":
        return CautiousAdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    raise ValueError(name)
