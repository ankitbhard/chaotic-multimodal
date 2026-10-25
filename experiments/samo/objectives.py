"""Differentiable, censoring-aware survival objectives and training-only controller."""
import torch
import torch.nn.functional as F

def cox_efron_free(risk, time, event):
    """Breslow Cox partial likelihood; normalize by number of observed events.

    All tied event times use Breslow's approximation.
    """
    risk, time, event = risk.reshape(-1), time.reshape(-1), event.reshape(-1).bool()
    if not event.any():
        return risk.sum() * 0.
    # Risk set includes all patients with observed duration >= event duration.
    log_denom = torch.logsumexp(
        risk.unsqueeze(0).expand(risk.numel(), -1).masked_fill(
            time.unsqueeze(0) < time.unsqueeze(1), -torch.inf), dim=1)
    return (log_denom[event] - risk[event]).mean()

def comparable_pairs(time, event):
    """(i,j) where i has observed event strictly before j's follow-up end."""
    t = time.reshape(-1)
    e = event.reshape(-1).bool()
    return e[:, None] & (t[:, None] < t[None, :])

def ranking_loss(risk, time, event, temperature=1.0):
    """Higher risk expected for earlier observed event; exclude censored/tied anchors."""
    if temperature <= 0: raise ValueError("temperature must be positive")
    mask = comparable_pairs(time, event)
    if not mask.any():
        return risk.sum() * 0.
    margin = (risk.reshape(-1)[None, :] - risk.reshape(-1)[:, None]) / temperature
    return F.softplus(margin[mask]).mean()

class LossWeightController:
    """EMA loss-ratio weighting, bounded and updated only from training batches.

    This is a heuristic, not a gradient-balancing or curvature controller.
    """
    def __init__(self, initial=(0.1, 0.1, 0.1), min_weight=0.02,
                 max_weight=0.3, ema=0.9, warmup=2, max_change=0.02):
        if not (0 <= ema < 1 and 0 < min_weight <= max_weight and max_change > 0):
            raise ValueError("invalid controller configuration")
        self.weights=list(initial)
        self.min_weight=min_weight
        self.max_weight=max_weight
        self.ema=ema
        self.warmup=warmup
        self.max_change=max_change
        self.baseline=None
        self.smooth=None
    def update(self, epoch, losses):
        """Update once per epoch from [fusion,image,gene,ranking] epoch means."""
        values=[float(v) for v in losses]
        if not all(__import__("math").isfinite(v) and v >= 0 for v in values):
            return self.weights[:]
        if self.baseline is None:
            self.baseline=[max(v,1e-8) for v in values]
            self.smooth=values[:]
        else:
            self.smooth=[self.ema*a+(1-self.ema)*b for a,b in zip(self.smooth,values)]
        if epoch < self.warmup:
            return self.weights[:]
        # Weight objectives whose normalized losses are lagging, bounded around 0.1.
        ratios=[self.smooth[i]/self.baseline[i] for i in range(1,4)]
        center=max(sum(ratios)/len(ratios),1e-8)
        for i,ratio in enumerate(ratios):
            target=max(self.min_weight,min(self.max_weight,0.1*ratio/center))
            delta=max(-self.max_change,min(self.max_change,target-self.weights[i]))
            self.weights[i]=max(self.min_weight,min(self.max_weight,self.weights[i]+delta))
        return self.weights[:]
    def state_dict(self):
        return dict(weights=self.weights[:],baseline=self.baseline,smooth=self.smooth)
    def load_state_dict(self,state):
        self.weights=list(state["weights"])
        self.baseline=state["baseline"]
        self.smooth=state["smooth"]
