"""Conservative history-guided lambda controller.

Historical rows are observational; this is a nearest-state heuristic, NOT a
causal optimizer or guarantee of improved C-index. Use histories from a
patient-disjoint development cohort only. Never feed evaluation-fold histories.
"""
import csv
import math
from pathlib import Path
import numpy as np

FEATURES=("train_fusion_loss","train_img_loss","train_gene_loss","train_rank_loss")
WEIGHTS=("weight_img","weight_gene","weight_rank")

def load_history(paths, horizon=3):
    """Training state at epoch t paired with later validation C-index (t+horizon)."""
    examples=[]
    for path in paths:
        with Path(path).open(newline="") as f:
            rows=list(csv.DictReader(f))
        for i in range(len(rows)-horizon):
            a,b=rows[i],rows[i+horizon]
            try:
                state=np.array([float(a[k]) for k in FEATURES],dtype=float)
                weights=np.array([float(a[k]) for k in WEIGHTS],dtype=float)
                score=float(b["val_ci"])
                if np.isfinite(state).all() and np.isfinite(weights).all() and math.isfinite(score):
                    examples.append((state,weights,score))
            except (ValueError,KeyError):
                continue
    return examples

class HistoryController:
    """Suggest bounded lambdas using similar historical training states.

    Recommendations are smoothed, capped per epoch, and only applied after
    warmup. History is loaded once; no evaluation C-index is used for updates.
    """
    def __init__(self, examples, min_weight=0.02,max_weight=0.3,
                 max_change=0.02,warmup=2,k=12,shrink=0.5):
        self.examples=list(examples)
        self.lo,self.hi=min_weight,max_weight
        self.max_change=max_change
        self.warmup=warmup
        self.k=k
        self.shrink=shrink
        if not (0<self.lo<=self.hi and self.max_change>0 and
                self.k>0 and 0<=self.shrink<=1):
            raise ValueError("Invalid history controller settings")
        if self.examples:
            self.states=np.stack([x[0] for x in self.examples])
            self.weights=np.stack([x[1] for x in self.examples])
            self.scores=np.array([x[2] for x in self.examples])
            self.scale=np.maximum(self.states.std(axis=0),0.1)

    def update(self,epoch,losses,current):
        current=np.asarray(current,dtype=float)
        if epoch<self.warmup or not self.examples:
            return current.tolist()
        state=np.asarray(losses,dtype=float)
        if not np.isfinite(state).all():
            return current.tolist()
        distance=np.linalg.norm((self.states-state)/self.scale,axis=1)
        near=np.argsort(distance)[:self.k]
        # Prefer high-performing nearby historical states; avoid selecting a
        # single lucky epoch by averaging the upper half of neighbors.
        selected=near[np.argsort(self.scores[near])[len(near)//2:]]
        target=self.weights[selected].mean(axis=0)
        target=(1-self.shrink)*current+self.shrink*target
        delta=np.clip(target-current,-self.max_change,self.max_change)
        return np.clip(current+delta,self.lo,self.hi).tolist()
