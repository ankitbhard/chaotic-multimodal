"""Censoring-aware objective gates. All decisions use TRAINING statistics only.
v3.3 adds an optional validation-guided mode choice supplied by the runner.
This is an experimental heuristic; no guarantee of C-index improvement.
"""
import math
from dataclasses import dataclass

MODES=("cox","aux","rank")

def survival_information(times, events):
    """Event fraction and ordered comparable-pair density; use training labels only."""
    import numpy as np
    t=np.asarray(times,dtype=float).reshape(-1)
    e=np.asarray(events,dtype=float).reshape(-1)>0
    n=len(t)
    if n<2: return dict(event_rate=float(e.mean()) if n else 0., pair_density=0., pair_effective=0.)
    pairs=e[:,None] & (t[:,None]<t[None,:])
    per_anchor=pairs.sum(axis=1).astype(float)
    count=per_anchor.sum()
    effective=float(count*count/(sum(per_anchor*per_anchor)+1e-12)) if count else 0.
    return dict(event_rate=float(e.mean()),pair_density=float(count/(n*(n-1))),
                pair_effective=effective)

@dataclass
class ControllerState:
    mode: str
    weights: tuple
    reason: str

class CensorAwareController:
    """Three-mode controller. All four variants can set all secondary lambdas to zero.

    v3_0: training-loss progress and stability.
    v3_1: additionally event/pair information.
    v3_2: additionally branch gradient alignment.
    v3_3: additionally inner-validation *candidate rollout* scores (if provided).
    Mode choices are not based on cohort names or outer test C-index.
    """
    def __init__(self, variant, aux_weight=.1, rank_weight=.1,
                 warmup=2, switch_margin=.035, min_dwell=4, ema=.85):
        if variant not in ("ra_samo_v3_0","ra_samo_v3_1","ra_samo_v3_2","ra_samo_v3_3"):
            raise ValueError("Unknown v3 variant")
        self.variant=variant
        self.aux_weight=float(aux_weight)
        self.rank_weight=float(rank_weight)
        self.warmup=warmup
        self.switch_margin=switch_margin
        self.min_dwell=min_dwell
        self.ema=ema
        self.mode="cox"
        self.last_switch=-min_dwell
        self.reference=None
        self.smooth=None
        self.state=ControllerState("cox",(0.,0.,0.),"initial")

    def update(self, epoch, losses, information, alignment=None, validation_scores=None):
        import numpy as np
        x=np.asarray(losses,dtype=float)
        if x.shape!=(4,) or not np.isfinite(x).all():
            return self.state
        if self.reference is None:
            self.reference=np.maximum(x,1e-8)
            self.smooth=x.copy()
        else:
            self.smooth=self.ema*self.smooth+(1-self.ema)*x
        if epoch<self.warmup:
            return self.state
        progress=1.-self.smooth/self.reference
        # Normalized improvement of the aux heads versus fused task.
        aux_progress=float((progress[1]+progress[2])/2.-progress[0])
        rank_progress=float(progress[3]-progress[0])
        scores={"cox":0.,"aux":aux_progress,"rank":rank_progress}
        if self.variant!="ra_samo_v3_0":
            event=float(information.get("event_rate",0.))
            density=float(information.get("pair_density",0.))
            # Centered, bounded reliability indicators; no cohort-specific thresholds.
            scores["aux"]+=.15*(.3-event)
            scores["rank"]+=.15*(density-.15)
        if self.variant in ("ra_samo_v3_2","ra_samo_v3_3") and alignment:
            a=float(alignment.get("aux",0.))
            r=float(alignment.get("rank",0.))
            if math.isfinite(a):scores["aux"]+=.15*a
            if math.isfinite(r):scores["rank"]+=.15*r
        if self.variant=="ra_samo_v3_3" and validation_scores:
            # Scores must come from INNER-validation matched rollouts, never test.
            finite={k:float(v) for k,v in validation_scores.items()
                    if k in MODES and math.isfinite(float(v))}
            if len(finite)==3:
                base=finite["cox"]
                for m in MODES:scores[m]+=(finite[m]-base)
        proposed=max(MODES,key=lambda m:scores[m])
        if proposed!=self.mode and epoch-self.last_switch>=self.min_dwell:
            if scores[proposed]>scores[self.mode]+self.switch_margin:
                self.mode=proposed
                self.last_switch=epoch
        weights={"cox":(0.,0.,0.),
                 "aux":(self.aux_weight,self.aux_weight,0.),
                 "rank":(0.,0.,self.rank_weight)}[self.mode]
        self.state=ControllerState(self.mode,weights,
             "scores:"+",".join(f"{k}={v:.4f}" for k,v in scores.items()))
        return self.state
