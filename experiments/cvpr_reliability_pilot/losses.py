"""
Loss functions for the CVPR reliability pilot.

  cox_loss           — Cox partial negative log-likelihood
  reliability_loss   — Perturbation-calibrated reliability supervision
  combined_loss      — Unified loss computation for all methods
"""

import torch
import torch.nn.functional as F


def cox_loss(risk_scores, os_months, vital_status):
    """
    Negative partial log-likelihood for Cox PH model.

    Args:
        risk_scores:  [B] scalar risk per patient
        os_months:    [B] survival time
        vital_status: [B] event indicator (1=died, 0=censored)
    Returns:
        scalar loss
    """
    if vital_status.sum() == 0:
        return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)

    order = torch.argsort(os_months, descending=True)
    risk = risk_scores[order]
    events = vital_status[order].float()
    log_cum_sum = torch.logcumsumexp(risk, dim=0)
    loss = -torch.mean((risk - log_cum_sum) * events)
    return loss


def perturbation_sensitivity(aux_head, features, sigma=0.1, n_perturbations=5):
    """
    Compute perturbation sensitivity: mean squared prediction change under noise.

    d_m = E[(r_m(z_m + delta) - r_m(z_m))^2]

    All computation is detached (no gradient through encoder or aux head).

    Args:
        aux_head: AuxCoxHead module
        features: [B, D] encoder features (will be detached)
        sigma: noise standard deviation
        n_perturbations: number of Monte Carlo samples
    Returns:
        [B] sensitivity per sample
    """
    feat_det = features.detach()
    B = feat_det.size(0)

    with torch.no_grad():
        base_risk = aux_head(feat_det)  # [B]

    sensitivity = torch.zeros(B, device=feat_det.device)
    with torch.no_grad():
        for _ in range(n_perturbations):
            delta = torch.randn_like(feat_det) * sigma
            pert_risk = aux_head(feat_det + delta)
            sensitivity += (pert_risk - base_risk) ** 2

    return sensitivity / n_perturbations


def reliability_loss(model, reliability_logits, sigma=0.1, n_perturbations=5):
    """
    Perturbation-calibrated reliability loss.

    For each modality m:
      1. Compute sensitivity d_m via perturbation of aux head predictions
      2. Normalize: d_norm = (d - mean) / (std + eps)
      3. Target: t_m = sigmoid(-d_norm)  (lower sensitivity -> higher reliability)
      4. Loss: BCE(sigmoid(q_m), t_m.detach())

    Normalization uses batch statistics (approximation of training-set stats).
    Targets are fully detached: no gradient flows through sensitivity computation.

    Args:
        model: PCRFModel instance (must have aux_img, aux_gene, cached features)
        reliability_logits: [B, 2] from model forward pass
        sigma: perturbation noise level
        n_perturbations: MC samples
    Returns:
        scalar reliability loss
    """
    img_feat, gene_feat = model.get_encoder_features()

    # Perturbation sensitivity per modality
    d_img = perturbation_sensitivity(
        model.aux_img, img_feat, sigma, n_perturbations
    )
    d_gene = perturbation_sensitivity(
        model.aux_gene, gene_feat, sigma, n_perturbations
    )

    eps = 1e-8

    # Normalize within batch (training-only normalization)
    d_img_norm = (d_img - d_img.mean()) / (d_img.std() + eps)
    d_gene_norm = (d_gene - d_gene.mean()) / (d_gene.std() + eps)

    # Reliability targets: lower sensitivity -> higher reliability
    target_img = torch.sigmoid(-d_img_norm).detach()
    target_gene = torch.sigmoid(-d_gene_norm).detach()

    # Predicted reliability (sigmoid of individual logits)
    pred_img = torch.sigmoid(reliability_logits[:, 0])
    pred_gene = torch.sigmoid(reliability_logits[:, 1])

    # BCE loss
    loss_img = F.binary_cross_entropy(pred_img, target_img)
    loss_gene = F.binary_cross_entropy(pred_gene, target_gene)

    return loss_img + loss_gene


def combined_loss(outputs, os_months, vital_status, model=None,
                  use_aux=False, use_reliability=False,
                  aux_weight=0.1, reliability_weight=0.01,
                  perturbation_sigma=0.1, n_perturbations=5):
    """
    Compute total loss for any pilot method.

    L_total = L_joint_Cox
            + aux_weight * (L_img_aux + L_gene_aux)     [if use_aux]
            + reliability_weight * L_reliability          [if use_reliability]

    Args:
        outputs: model forward dict
        os_months: [B] survival times
        vital_status: [B] event indicators
        model: PCRFModel (needed for reliability loss)
        use_aux: include auxiliary Cox loss (flowing through encoders)
        use_reliability: include reliability calibration loss
        aux_weight: weight for each aux loss
        reliability_weight: gamma for reliability loss
        perturbation_sigma: noise sigma for perturbation calibration
        n_perturbations: MC samples for perturbation
    Returns:
        (total_loss, loss_dict) where loss_dict has component losses
    """
    risk = outputs["survival"][:, 1]
    loss_joint = cox_loss(risk, os_months, vital_status)

    loss_dict = {"joint": loss_joint.item()}
    total = loss_joint

    # Auxiliary Cox losses
    if use_aux and "aux_img_risk" in outputs:
        loss_aux_img = cox_loss(outputs["aux_img_risk"], os_months, vital_status)
        loss_aux_gene = cox_loss(outputs["aux_gene_risk"], os_months, vital_status)
        total = total + aux_weight * (loss_aux_img + loss_aux_gene)
        loss_dict["aux_img"] = loss_aux_img.item()
        loss_dict["aux_gene"] = loss_aux_gene.item()

    # For PCRF without aux_loss in total, still train aux heads separately
    # (they need to be trained for perturbation calibration to work)
    if not use_aux and hasattr(model, 'aux_img') and "aux_img_risk" in outputs:
        # Aux heads trained on detached features — loss only updates aux head weights
        loss_aux_img = cox_loss(outputs["aux_img_risk"], os_months, vital_status)
        loss_aux_gene = cox_loss(outputs["aux_gene_risk"], os_months, vital_status)
        total = total + aux_weight * (loss_aux_img + loss_aux_gene)
        loss_dict["aux_img_detached"] = loss_aux_img.item()
        loss_dict["aux_gene_detached"] = loss_aux_gene.item()

    # Reliability loss
    if use_reliability and "reliability_logits" in outputs:
        loss_rel = reliability_loss(
            model, outputs["reliability_logits"],
            sigma=perturbation_sigma,
            n_perturbations=n_perturbations,
        )
        total = total + reliability_weight * loss_rel
        loss_dict["reliability"] = loss_rel.item()

    loss_dict["total"] = total.item()
    return total, loss_dict
