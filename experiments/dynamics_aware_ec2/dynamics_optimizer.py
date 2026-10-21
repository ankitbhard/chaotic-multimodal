import math
import torch
from torch.optim import Optimizer

class DynamicsAwareAdamW(Optimizer):
    """Experimental per-tensor adaptive second-moment decay; not a validated optimizer."""
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8,
                 weight_decay=1e-4, beta2_min=0.95, sensitivity=3.0,
                 change_ema=0.9, eps_change=1e-6):
        if not (0 < beta2_min <= betas[1] < 1 and 0 <= betas[0] < 1):
            raise ValueError("invalid beta bounds")
        if not (lr>0 and eps>0 and eps_change>0 and weight_decay>=0 and sensitivity>=0 and 0<=change_ema<1):
            raise ValueError("invalid hyperparameters")
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay,
            beta2_min=beta2_min,sensitivity=sensitivity,change_ema=change_ema,eps_change=eps_change))

    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad(): loss=closure()
        for group in self.param_groups:
            lr=group["lr"]; b1,b2max=group["betas"]
            for p in group["params"]:
                if p.grad is None: continue
                g=p.grad
                if g.is_sparse: raise RuntimeError("Sparse gradients not supported")
                s=self.state[p]
                if not s:
                    s["step"]=0; s["exp_avg"]=torch.zeros_like(p)
                    s["exp_avg_sq"]=torch.zeros_like(p)
                    s["beta2_product"]=1.0; s["change_ema_value"]=0.0
                    s["last_beta2"]=b2max
                s["step"]+=1
                m=s["exp_avg"]; v=s["exp_avg_sq"]; g2=g.square()
                if s["step"]==1: b2=b2max
                else:
                    discrepancy=((g2-v).abs()/(g2+v+group["eps_change"])).mean().item()
                    s["change_ema_value"]=group["change_ema"]*s["change_ema_value"]+(1-group["change_ema"])*discrepancy
                    b2=group["beta2_min"]+(b2max-group["beta2_min"])*math.exp(-group["sensitivity"]*s["change_ema_value"])
                s["last_beta2"]=b2; s["beta2_product"]*=b2
                m.mul_(b1).add_(g,alpha=1-b1)
                v.mul_(b2).add_(g2,alpha=1-b2)
                mh=m/(1-b1**s["step"])
                vh=v/(1-s["beta2_product"])
                p.mul_(1-lr*group["weight_decay"])
                p.addcdiv_(mh,vh.sqrt().add(group["eps"]),value=-lr)
        return loss
