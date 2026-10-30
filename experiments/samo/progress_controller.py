"""Training-only relative learning progress weighting for unimodal Cox heads."""
import math

class ProgressController:
    def __init__(self, initial=0.1, ema=0.8, gain=2.0, warmup=2, low=0.02, high=0.3):
        self.initial, self.ema, self.gain = initial, ema, gain
        self.warmup, self.low, self.high = warmup, low, high
        self.reference = self.smoothed = None

    def update(self, epoch, image_loss, gene_loss):
        values = (float(image_loss), float(gene_loss))
        if not all(math.isfinite(v) and v >= 0 for v in values):
            return (self.initial, self.initial)
        if self.reference is None:
            self.reference = tuple(max(v, 1e-8) for v in values)
            self.smoothed = values
        else:
            self.smoothed = tuple(self.ema * old + (1 - self.ema) * new for old, new in zip(self.smoothed, values))
        if epoch < self.warmup:
            return (self.initial, self.initial)
        progress = [1 - self.smoothed[i] / self.reference[i] for i in range(2)]
        gap = progress[1] - progress[0]
        weights = (self.initial * math.exp(self.gain * gap / 2), self.initial * math.exp(-self.gain * gap / 2))
        return tuple(max(self.low, min(self.high, w)) for w in weights)
