"""RA-SAMO v4: Validation-Directed Probe Controller.

v4 fixes v3's root cause (loss-progress is not a proxy for C-index) by
using val C-index directly via a bounded exploration phase.

Architecture:
  Phase 1 — Warmup (epochs 0-1): Train with cox-only, establish baseline
  Phase 2 — Probe (epochs 2 to probe_end): Test 3 weight configs from shared
             checkpoint. Each probe: probe_epochs from same saved state.
             Score by val C-index.
  Phase 3 — Commit (probe_end to end): Lock in winner's weights. No switching.

Key properties:
  - Uses val_ci directly (not loss progress)
  - Bounded exploration cost (probe_candidates * probe_epochs epochs)
  - No per-cancer tuning — same 3 candidates everywhere
  - Commits early — no switching instability
  - Saves checkpoints during probes — best-overall checkpoint used regardless
"""

import copy
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

import torch


@dataclass
class ProbeCandidate:
    name: str
    weights: Tuple[float, float, float]  # (img_aux, gene_aux, rank)


@dataclass
class V4State:
    phase: str              # "warmup", "probe", "commit"
    active_candidate: str   # current candidate name
    weights: Tuple[float, float, float]
    reason: str
    committed_candidate: Optional[str] = None


# The three probe candidates (same everywhere, no per-cancer tuning)
PROBE_CANDIDATES = [
    ProbeCandidate("cox_only", (0.0, 0.0, 0.0)),
    ProbeCandidate("aux_focus", (0.1, 0.1, 0.0)),
    ProbeCandidate("balanced", (0.1, 0.1, 0.1)),
]


class V4Controller:
    """Validation-directed probe selection controller.

    Usage in training loop:
        controller = V4Controller(warmup=2, probe_epochs=4)

        for epoch in range(total_epochs):
            action = controller.get_action(epoch)

            if action == "train":
                # Normal training with controller.state.weights
                ...
                val_ci = evaluate(...)
                controller.report_val_ci(epoch, val_ci)

            elif action == "save_checkpoint":
                # Save model/optimizer/rng state for probe rollbacks
                controller.save_checkpoint(model, optimizer, generator)
                # Then train normally
                ...
                val_ci = evaluate(...)
                controller.report_val_ci(epoch, val_ci)

            elif action == "restore_and_train":
                # Restore to probe checkpoint, train with current candidate weights
                controller.restore_checkpoint(model, optimizer, generator)
                ...
                val_ci = evaluate(...)
                controller.report_val_ci(epoch, val_ci)
    """

    def __init__(self, warmup: int = 2, probe_epochs: int = 4):
        self.warmup = warmup
        self.probe_epochs = probe_epochs
        self.candidates = list(PROBE_CANDIDATES)
        self.n_candidates = len(self.candidates)

        # Probe phase spans: [warmup, warmup + n_candidates * probe_epochs)
        self.probe_start = warmup
        self.probe_end = warmup + self.n_candidates * self.probe_epochs

        # State
        self.state = V4State(
            phase="warmup",
            active_candidate="cox_only",
            weights=(0.0, 0.0, 0.0),
            reason="warmup",
        )

        # Probe scoring
        self.probe_scores: Dict[str, List[float]] = {c.name: [] for c in self.candidates}

        # Checkpoint storage (set by save_checkpoint)
        self._checkpoint = None

    @property
    def committed(self) -> bool:
        return self.state.phase == "commit"

    def _current_probe_candidate(self, epoch: int) -> ProbeCandidate:
        """Which candidate is active at this probe epoch?"""
        probe_idx = (epoch - self.probe_start) // self.probe_epochs
        probe_idx = min(probe_idx, self.n_candidates - 1)
        return self.candidates[probe_idx]

    def get_action(self, epoch: int) -> str:
        """Determine what the training loop should do this epoch.

        Returns: "train", "save_checkpoint", or "restore_and_train"
        """
        if epoch < self.probe_start:
            # Warmup phase — plain cox training
            self.state = V4State("warmup", "cox_only", (0.0, 0.0, 0.0), "warmup")
            return "train"

        if epoch >= self.probe_end:
            # Commit phase — locked weights
            if self.state.phase != "commit":
                self._commit()
            return "train"

        # Probe phase
        candidate = self._current_probe_candidate(epoch)
        epoch_within_probe = (epoch - self.probe_start) % self.probe_epochs

        if epoch_within_probe == 0:
            # Start of a new candidate probe
            self.state = V4State(
                "probe", candidate.name, candidate.weights,
                f"probe:{candidate.name} ep_in_probe=0"
            )
            if epoch == self.probe_start:
                # First probe — need to save checkpoint
                return "save_checkpoint"
            else:
                # Subsequent probes — restore to shared checkpoint
                return "restore_and_train"
        else:
            # Continue current probe
            self.state = V4State(
                "probe", candidate.name, candidate.weights,
                f"probe:{candidate.name} ep_in_probe={epoch_within_probe}"
            )
            return "train"

    def report_val_ci(self, epoch: int, val_ci: float):
        """Report validation C-index after each epoch."""
        if self.state.phase == "probe":
            self.probe_scores[self.state.active_candidate].append(val_ci)

    def _commit(self):
        """Select best probe candidate and lock in."""
        # Score each candidate by mean val_ci across its probe epochs
        scores = {}
        for name, vals in self.probe_scores.items():
            if vals:
                scores[name] = sum(vals) / len(vals)

        if not scores:
            # Fallback to cox_only if no scores
            winner = self.candidates[0]
        else:
            winner_name = max(scores, key=scores.get)
            winner = next(c for c in self.candidates if c.name == winner_name)

        reason_parts = [f"{k}={v:.4f}" for k, v in sorted(scores.items())]
        self.state = V4State(
            "commit", winner.name, winner.weights,
            f"committed:{winner.name} scores={','.join(reason_parts)}"
        )
        self.state.committed_candidate = winner.name

    def save_checkpoint(self, model, optimizer, generator):
        """Save full state for probe rollbacks."""
        self._checkpoint = {
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "rng": torch.random.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "gen": generator.get_state(),
        }

    def restore_checkpoint(self, model, optimizer, generator):
        """Restore to probe checkpoint."""
        if self._checkpoint is None:
            raise RuntimeError("No checkpoint saved — call save_checkpoint first")
        model.load_state_dict(copy.deepcopy(self._checkpoint["model"]))
        optimizer.load_state_dict(copy.deepcopy(self._checkpoint["optimizer"]))
        torch.random.set_rng_state(self._checkpoint["rng"].clone())
        if self._checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state(self._checkpoint["cuda_rng"].clone())
        generator.set_state(self._checkpoint["gen"].clone())

    def summary(self) -> dict:
        """Return summary for logging."""
        return {
            "phase": self.state.phase,
            "active_candidate": self.state.active_candidate,
            "weights": list(self.state.weights),
            "reason": self.state.reason,
            "committed_candidate": self.state.committed_candidate,
            "probe_scores": {k: list(v) for k, v in self.probe_scores.items()},
        }
