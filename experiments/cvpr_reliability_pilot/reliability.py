"""
Perturbation calibration and reliability diagnostics.

This module provides:
  - PerturbationCalibrator: tracks running statistics of perturbation sensitivity
  - Diagnostic functions for analyzing reliability estimates
  - Correlation analysis between reliability and prediction utility
"""

import torch
import numpy as np
from collections import defaultdict


class PerturbationCalibrator:
    """
    Tracks running statistics of perturbation sensitivity during training.

    Maintains exponential moving averages of sensitivity mean and variance
    for each modality, computed from training data only.

    Used for:
    1. Normalizing reliability targets with training-set statistics
    2. Monitoring whether reliability estimates are meaningful
    3. Detecting reliability collapse (all weights → one modality)
    """

    def __init__(self, decay=0.99):
        self.decay = decay
        self._initialized = False
        self.img_mean = 0.0
        self.img_var = 1.0
        self.gene_mean = 0.0
        self.gene_var = 1.0
        self.n_updates = 0

    def update(self, img_sensitivity, gene_sensitivity):
        """
        Update running stats with batch sensitivities.

        Args:
            img_sensitivity: [B] perturbation sensitivity for image
            gene_sensitivity: [B] perturbation sensitivity for gene
        """
        img_s = img_sensitivity.detach().cpu()
        gene_s = gene_sensitivity.detach().cpu()

        batch_img_mean = img_s.mean().item()
        batch_img_var = img_s.var().item() if len(img_s) > 1 else 0.0
        batch_gene_mean = gene_s.mean().item()
        batch_gene_var = gene_s.var().item() if len(gene_s) > 1 else 0.0

        if not self._initialized:
            self.img_mean = batch_img_mean
            self.img_var = batch_img_var
            self.gene_mean = batch_gene_mean
            self.gene_var = batch_gene_var
            self._initialized = True
        else:
            d = self.decay
            self.img_mean = d * self.img_mean + (1 - d) * batch_img_mean
            self.img_var = d * self.img_var + (1 - d) * batch_img_var
            self.gene_mean = d * self.gene_mean + (1 - d) * batch_gene_mean
            self.gene_var = d * self.gene_var + (1 - d) * batch_gene_var

        self.n_updates += 1

    def get_stats(self):
        return {
            "img_mean": self.img_mean,
            "img_var": self.img_var,
            "gene_mean": self.gene_mean,
            "gene_var": self.gene_var,
            "n_updates": self.n_updates,
        }


def compute_reliability_diagnostics(model, loader, device, sigma=0.1, n_pert=5):
    """
    Compute detailed reliability diagnostics on a dataset.

    Returns per-patient:
    - reliability weights (alpha_img, alpha_gene)
    - perturbation sensitivity per modality
    - auxiliary risk scores per modality
    - joint risk score

    This enables post-hoc analysis of whether reliability correlates
    with prediction utility.
    """
    from .losses import perturbation_sensitivity

    model.eval()
    results = defaultdict(list)

    with torch.no_grad():
        for inputs, labels in loader:
            img = inputs["image"].to(device)
            gene = inputs["gene"].to(device)
            os_m = labels["os_months"].float()
            vs = labels["vital_status"].float()

            outputs = model({"image": img, "gene": gene})

            # Reliability weights
            if "reliability_weights" in outputs:
                w = outputs["reliability_weights"].cpu().numpy()
                results["alpha_img"].extend(w[:, 0].tolist())
                results["alpha_gene"].extend(w[:, 1].tolist())

            # Joint risk
            risk = outputs["survival"][:, 1].cpu().numpy()
            results["joint_risk"].extend(risk.tolist())

            # Aux risks
            if "aux_img_risk" in outputs:
                results["aux_img_risk"].extend(
                    outputs["aux_img_risk"].cpu().numpy().tolist()
                )
                results["aux_gene_risk"].extend(
                    outputs["aux_gene_risk"].cpu().numpy().tolist()
                )

            # Perturbation sensitivity
            if hasattr(model, "aux_img"):
                img_feat, gene_feat = model.get_encoder_features()
                d_img = perturbation_sensitivity(
                    model.aux_img, img_feat, sigma, n_pert
                )
                d_gene = perturbation_sensitivity(
                    model.aux_gene, gene_feat, sigma, n_pert
                )
                results["sensitivity_img"].extend(d_img.cpu().numpy().tolist())
                results["sensitivity_gene"].extend(d_gene.cpu().numpy().tolist())

            results["os_months"].extend(os_m.numpy().tolist())
            results["vital_status"].extend(vs.numpy().tolist())

    return dict(results)


def check_reliability_collapse(diagnostics, threshold=0.05):
    """
    Check if reliability has collapsed (all weight on one modality).

    Returns:
        dict with collapse diagnosis
    """
    alpha_img = np.array(diagnostics.get("alpha_img", []))
    if len(alpha_img) == 0:
        return {"collapsed": False, "reason": "no reliability weights"}

    mean_img = alpha_img.mean()
    std_img = alpha_img.std()

    collapsed = std_img < threshold and (mean_img > 0.95 or mean_img < 0.05)

    return {
        "collapsed": collapsed,
        "mean_alpha_img": float(mean_img),
        "mean_alpha_gene": float(1 - mean_img),
        "std_alpha_img": float(std_img),
        "reason": (
            f"img_weight={mean_img:.3f}±{std_img:.3f}"
            if not collapsed
            else f"COLLAPSED: all weight on {'image' if mean_img > 0.5 else 'gene'}"
        ),
    }


def reliability_utility_correlation(diagnostics):
    """
    Investigate whether reliability estimates correlate with prediction utility.

    Checks if higher reliability weight on a modality corresponds to
    better individual-modality prediction for that patient.

    Returns correlation statistics. Does NOT assume the correlation is positive.
    """
    from scipy import stats

    alpha_img = np.array(diagnostics.get("alpha_img", []))
    sens_img = np.array(diagnostics.get("sensitivity_img", []))
    sens_gene = np.array(diagnostics.get("sensitivity_gene", []))

    result = {}

    if len(alpha_img) > 5 and len(sens_img) > 5:
        # Correlation: reliability weight vs sensitivity (expect negative)
        r_img, p_img = stats.spearmanr(alpha_img, sens_img)
        r_gene, p_gene = stats.spearmanr(1 - alpha_img, sens_gene)
        result["alpha_img_vs_sensitivity_img"] = {
            "spearman_r": float(r_img),
            "p_value": float(p_img),
        }
        result["alpha_gene_vs_sensitivity_gene"] = {
            "spearman_r": float(r_gene),
            "p_value": float(p_gene),
        }

    return result
