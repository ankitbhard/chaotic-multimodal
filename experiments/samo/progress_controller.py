"""Train-only modality learning-progress *multipliers* for RA-SAMO v1.

The SAMO controller supplies base auxiliary weights. This controller supplies
bounded dimensionless multipliers (neutral = 1), avoiding double weighting.
"""
import math


class ProgressController:
    def __init__(self, ema=0.9, gain=1.0, min_scale=0.5,
                 max_scale=1.5, warmup=2):
        if not (0 <= ema < 1 and gain >= 0 and
                0 < min_scale <= 1 <= max_scale and warmup >= 0):
            raise ValueError("Invalid progress-controller configuration")
        self.ema = ema
        self.gain = gain
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.warmup = warmup
        self.reference = None
        self.smoothed = None
        self.scales = (1.0, 1.0)

    def update(self, epoch, image_loss, gene_loss):
        """Update once per epoch using only mean TRAINING auxiliary Cox losses.

        Epoch indexing is zero-based, as in the experiment runner.
        Returned multipliers are applied to the *next* epoch, never the
        batches used to calculate them.
        """
        values = (float(image_loss), float(gene_loss))
        if not all(math.isfinite(v) and v >= 0 for v in values):
            return self.scales
        if self.reference is None:
            self.reference = tuple(max(v, 1e-8) for v in values)
            self.smoothed = values
        else:
            self.smoothed = tuple(
                self.ema * old + (1 - self.ema) * new
                for old, new in zip(self.smoothed, values)
            )
        if epoch < self.warmup:
            return self.scales
        progress = tuple(
            1.0 - self.smoothed[i] / self.reference[i] for i in range(2)
        )
        gap = progress[1] - progress[0]  # positive: image is lagging
        exponent = max(-50.0, min(50.0, self.gain * gap / 2.0))
        proposed = (math.exp(exponent), math.exp(-exponent))
        self.scales = tuple(
            max(self.min_scale, min(self.max_scale, value))
            for value in proposed
        )
        return self.scales
