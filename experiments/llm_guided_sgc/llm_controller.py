"""Experimental bounded LLM supervisor for AdamW/SGC-AdamW.

The supervisor does not call a model itself. An agent supplies a proposal
from an external LLM; this module validates, limits, and records it.
No held-out validation/test metrics should enter the online prompt.
"""
from dataclasses import dataclass, asdict
import json
import math


@dataclass(frozen=True)
class ControllerLimits:
    beta2_min_low: float = 0.95
    beta2_min_high: float = 0.999
    lr_multiplier_low: float = 0.8
    lr_multiplier_high: float = 1.2
    max_beta2_change: float = 0.005
    max_lr_multiplier_change: float = 0.05


def validate_proposal(raw, previous, limits=ControllerLimits()):
    """Return (accepted_settings, audit). Reject malformed/out-of-bounds actions.

    previous: dict with beta2_min and lr_multiplier. Caller must apply settings
    to its optimizer only at a well-defined optimizer-step boundary.
    """
    try:
        proposal = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(proposal, dict):
            raise ValueError("proposal must be an object")
        if set(proposal) != {"beta2_min", "lr_multiplier", "reason"}:
            raise ValueError("unexpected or missing proposal fields")
        if not isinstance(proposal["reason"], str) or not proposal["reason"].strip():
            raise ValueError("reason required")
        settings = {}
        for key in ("beta2_min", "lr_multiplier"):
            val = proposal[key]
            if isinstance(val, bool) or not isinstance(val, (float, int)):
                raise ValueError(f"{key} must be numeric")
            if not math.isfinite(val):
                raise ValueError(f"{key} must be finite")
            settings[key] = float(val)
        b = settings["beta2_min"]
        l = settings["lr_multiplier"]
        if not limits.beta2_min_low <= b <= limits.beta2_min_high:
            raise ValueError("beta2_min outside hard bounds")
        if not limits.lr_multiplier_low <= l <= limits.lr_multiplier_high:
            raise ValueError("lr_multiplier outside hard bounds")
        if abs(b - previous["beta2_min"]) > limits.max_beta2_change:
            raise ValueError("beta2_min change exceeds step limit")
        if abs(l - previous["lr_multiplier"]) > limits.max_lr_multiplier_change:
            raise ValueError("lr_multiplier change exceeds step limit")
        return settings, {"accepted": True, "proposal": proposal, "applied": settings}
    except (ValueError, TypeError, KeyError) as exc:
        return dict(previous), {"accepted": False, "reason": str(exc), "proposal": raw}


def telemetry_prompt_payload(epoch, group_metrics, recent_train_losses):
    """Training-only, JSON-serializable summary for a model adapter."""
    return {
        "epoch": int(epoch),
        "group_metrics": group_metrics,
        "recent_train_losses": [float(v) for v in recent_train_losses],
        "allowed_actions": ["beta2_min", "lr_multiplier"],
        "do_not_use": ["test_metrics", "validation_metrics"],
    }
