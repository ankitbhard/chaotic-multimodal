"""Matched-budget SGD/SAM scheduling pilot.

All scheduled methods share the same full-model SAM perturbation, SGD optimizer,
learning-rate scheduler, and training loop. Only the SAM step indices differ.
"""
import math
import random
import torch


class SAMSchedule:
    def __init__(self, mode, total_steps, fraction=0.5, seed=42):
        if mode not in ("periodic","random","chaotic"):
            raise ValueError(mode)
        if not 0 <= fraction <= 1:
            raise ValueError("fraction must be in [0,1]")
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        self.mode=mode
        self.total_steps=int(total_steps)
        self.budget=round(self.total_steps*fraction)
        self.seed=int(seed)
        # Exactly matched SAM budgets. No method sees validation/test metrics.
        if mode=="periodic":
            # Evenly distribute B selections across T steps.
            chosen=[((i+1)*self.total_steps)//self.budget - 1
                    for i in range(self.budget)] if self.budget else []
        elif mode=="random":
            chosen=random.Random(seed).sample(range(self.total_steps),self.budget)
        else:
            # Logistic map invariant distribution is nonuniform. Rank transformed
            # chaotic scores to achieve an exact budget, with no q-as-threshold error.
            z=((random.Random(seed).random()*0.8)+0.1)
            scores=[]
            for i in range(self.total_steps):
                z=4.0*z*(1.0-z)
                if z<=0.0 or z>=1.0:
                    raise RuntimeError("Chaotic orbit collapsed in floating point")
                scores.append((z,i))
            chosen=[i for _,i in sorted(scores)[:self.budget]]
        self.selected=frozenset(chosen)

    def use_sam(self, step):
        if not 0 <= step < self.total_steps:
            raise IndexError(step)
        return step in self.selected


class ScheduledSAM:
    """SGD on skipped steps; full-parameter two-pass SAM on scheduled steps."""
    def __init__(self, model, schedule, lr=1e-3, weight_decay=0.0, rho=0.05):
        self.model=model
        self.schedule=schedule
        self.rho=float(rho)
        if self.rho<0: raise ValueError("rho must be nonnegative")
        self.base=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,
                                  weight_decay=weight_decay)
        self.steps=0
        self.sam_steps=0
        self.history=[]

    def step(self, closure):
        t=self.steps
        do_sam=self.schedule.use_sam(t)
        self.base.zero_grad(set_to_none=True)
        loss=closure()
        loss.backward()
        if do_sam:
            grads=[p.grad for p in self.model.parameters() if p.grad is not None]
            if grads:
                norm=torch.linalg.vector_norm(torch.stack(
                    [torch.linalg.vector_norm(g.detach().float()) for g in grads]))
            else:
                norm=None
            changed=[]
            try:
                if norm is not None and bool(torch.isfinite(norm)) and norm.item()>1e-12:
                    with torch.no_grad():
                        for p in self.model.parameters():
                            if p.grad is not None:
                                eps=(self.rho*p.grad.detach()/norm).clone()
                                p.add_(eps)
                                changed.append((p,eps))
                self.base.zero_grad(set_to_none=True)
                loss=closure()
                loss.backward()
            finally:
                with torch.no_grad():
                    for p,eps in changed:
                        p.sub_(eps)
            self.sam_steps+=1
        self.base.step()
        self.steps+=1
        self.history.append({"step":self.steps,"sam":do_sam,
                             "lr":float(self.base.param_groups[0]["lr"])})
        return loss.detach()
