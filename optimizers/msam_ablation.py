"""
M-SAM Ablation Variants.

Two subclasses of MultimodalChaoticOptimizerMSAM for ablation study:

  MSAMNoOGM    — Shapley + SAM on dominant encoder, but NO OGM-GE modulation.
                 Isolates the SAM perturbation contribution.

  MSAMUniform  — OGM-GE + SAM, but random/uniform modality selection
                 (50/50) instead of Shapley. Shows Shapley matters.

Usage:
    from optimizers.msam_ablation import MSAMNoOGM, MSAMUniform
"""

import torch
from typing import Dict

try:
    from optimizers.chaotic_optimizer_msam import MultimodalChaoticOptimizerMSAM
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from optimizers.chaotic_optimizer_msam import MultimodalChaoticOptimizerMSAM


__all__ = ["MSAMNoOGM", "MSAMUniform", "MSAMShapleyGMOnly"]


class MSAMNoOGM(MultimodalChaoticOptimizerMSAM):
    """
    M-SAM WITHOUT OGM-GE gradient modulation.

    Still uses Shapley values to identify dominant modality and applies
    SAM perturbation to that encoder, but skips the gradient suppression
    and GE noise injection. This isolates SAM's contribution.
    """

    @torch.no_grad()
    def _modulate_gradients(self,
                            contributions: Dict[str, float],
                            dominant: str) -> None:
        """No-op: skip OGM-GE gradient modulation."""
        pass


class MSAMShapleyGMOnly(MultimodalChaoticOptimizerMSAM):
    """
    GM + Shapley WITHOUT SAM perturbation.

    Uses Shapley values to identify dominant modality and applies OGM-GE
    gradient modulation, but skips the SAM perturbation step entirely.
    This isolates the Shapley-guided gradient modulation contribution
    (no sharpness-aware step at all).
    """

    def _sam_perturb(self, dominant: str) -> Dict[str, torch.Tensor]:
        """No-op: skip SAM perturbation."""
        return {}

    def _sam_restore(self, dominant: str, eps_store: Dict[str, torch.Tensor]):
        """No-op: nothing to restore."""
        pass


class MSAMUniform(MultimodalChaoticOptimizerMSAM):
    """
    M-SAM with UNIFORM/RANDOM modality selection instead of Shapley.

    Still applies OGM-GE and SAM perturbation, but selects the "dominant"
    modality randomly (50/50) rather than using Shapley values. This shows
    that principled modality identification via Shapley is important.
    """

    @torch.no_grad()
    def _compute_shapley(self,
                         inputs: Dict[str, torch.Tensor],
                         labels: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Return uniform 50/50 contributions (no Shapley computation)."""
        n = len(self.modality_names)
        return {m: 1.0 / n for m in self.modality_names}


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.joint_model import build_model

    B, N_GENES, IMG_DIM = 4, 1000, 256

    model = build_model(n_genes=N_GENES, image_dim=IMG_DIM, verbose=False)

    # Test MSAMNoOGM
    base_opt = torch.optim.SGD(model.parameters(), lr=1e-3,
                                momentum=0.9, weight_decay=1e-4)
    no_ogm = MSAMNoOGM(
        base_opt, model,
        modality_names=["image", "gene"],
        base_lr=1e-3, rho=0.05, alpha=0.5,
    )

    inputs = {"image": torch.randn(B, IMG_DIM), "gene": torch.randn(B, N_GENES)}
    labels = {"survival": torch.randint(0, 2, (B,))}

    model.train()
    def closure():
        out = model(inputs)
        loss = torch.nn.CrossEntropyLoss()(out["survival"], labels["survival"])
        return loss, out

    loss, _ = no_ogm.step(closure, inputs, labels)
    print(f"MSAMNoOGM — loss={loss.item():.4f}  ✓")

    # Test MSAMUniform
    model2 = build_model(n_genes=N_GENES, image_dim=IMG_DIM, verbose=False)
    base_opt2 = torch.optim.SGD(model2.parameters(), lr=1e-3,
                                 momentum=0.9, weight_decay=1e-4)
    uniform = MSAMUniform(
        base_opt2, model2,
        modality_names=["image", "gene"],
        base_lr=1e-3, rho=0.05, alpha=0.5,
    )

    inputs2 = {"image": torch.randn(B, IMG_DIM), "gene": torch.randn(B, N_GENES)}
    labels2 = {"survival": torch.randint(0, 2, (B,))}

    model2.train()
    def closure2():
        out = model2(inputs2)
        loss = torch.nn.CrossEntropyLoss()(out["survival"], labels2["survival"])
        return loss, out

    loss2, _ = uniform.step(closure2, inputs2, labels2)
    hist = uniform.get_contribution_history()
    # Uniform should always give 0.5 / 0.5
    assert all(abs(v - 0.5) < 1e-6 for v in hist["image"]), \
        f"Expected 0.5, got {hist['image']}"
    print(f"MSAMUniform — loss={loss2.item():.4f}  contributions={hist}  ✓")

    # Test MSAMShapleyGMOnly
    model3 = build_model(n_genes=N_GENES, image_dim=IMG_DIM, verbose=False)
    base_opt3 = torch.optim.SGD(model3.parameters(), lr=1e-3,
                                 momentum=0.9, weight_decay=1e-4)
    shapley_gm = MSAMShapleyGMOnly(
        base_opt3, model3,
        modality_names=["image", "gene"],
        base_lr=1e-3, rho=0.05, alpha=0.5,
    )

    inputs3 = {"image": torch.randn(B, IMG_DIM), "gene": torch.randn(B, N_GENES)}
    labels3 = {"survival": torch.randint(0, 2, (B,))}

    model3.train()
    def closure3():
        out = model3(inputs3)
        loss = torch.nn.CrossEntropyLoss()(out["survival"], labels3["survival"])
        return loss, out

    loss3, _ = shapley_gm.step(closure3, inputs3, labels3)
    print(f"MSAMShapleyGMOnly — loss={loss3.item():.4f}  ✓")

    print("All ablation self-tests PASSED.")
