"""RA-SAMO v0: branch-specific gradient diagnostics and auxiliary projection.

Call prepare_branch_gradients before loss.backward(), then apply_branch_gradients
after backward() and before optimizer.step(). Shared/fusion/head gradients remain
those of the original weighted loss. No validation/test labels are used.
"""
from dataclasses import dataclass
import torch


@dataclass
class BranchDiagnostics:
    cosine_aux: float
    cosine_rank: float
    projected_aux: bool
    projected_rank: bool
    primary_norm: float


def _grads(loss, params):
    if not loss.requires_grad:
        return [torch.zeros_like(p) for p in params]
    raw = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, raw)]


def _dot(a, b):
    return sum((x * y).sum() for x, y in zip(a, b))


def _cosine(a, b, eps=1e-12):
    na = _dot(a, a).clamp_min(0).sqrt()
    nb = _dot(b, b).clamp_min(0).sqrt()
    if na.item() <= eps or nb.item() <= eps:
        return 0.0
    return float((_dot(a, b) / (na * nb + eps)).item())


def _project(aux, primary, eps=1e-12):
    denom = _dot(primary, primary)
    conflict = bool(_dot(aux, primary).item() < 0 and denom.item() > eps)
    if not conflict:
        return aux, False
    coeff = _dot(aux, primary) / (denom + eps)
    return [a - coeff * p for a, p in zip(aux, primary)], True


def prepare_branch_gradients(model, fusion_loss, image_loss, gene_loss,
                             ranking_loss, weights, project=True):
    """Compute replacement encoder gradients using the SAME forward graph.

    weights = (image_aux, gene_aux, ranking). Only the image and gene
    encoder gradients are replaced. Auxiliary head and shared parameter
    gradients must be computed normally from weighted_loss.backward().
    """
    if len(weights) != 3:
        raise ValueError("weights must have three elements")
    result = {}
    for branch, aux_loss, aux_weight in (
        ("image", image_loss, weights[0]),
        ("gene", gene_loss, weights[1]),
    ):
        params = [p for p in getattr(model, branch + "_encoder").parameters()
                  if p.requires_grad]
        if not params:
            raise ValueError(f"No trainable {branch} encoder parameters")
        primary = _grads(fusion_loss, params)
        aux = _grads(aux_loss, params)
        rank = _grads(ranking_loss, params)
        cos_aux, cos_rank = _cosine(primary, aux), _cosine(primary, rank)
        if project:
            aux, projected_aux = _project(aux, primary)
            rank, projected_rank = _project(rank, primary)
        else:
            projected_aux = projected_rank = False
        combined = [p + aux_weight * a + weights[2] * r
                    for p, a, r in zip(primary, aux, rank)]
        result[branch] = (
            params,
            combined,
            BranchDiagnostics(cos_aux, cos_rank, projected_aux,
                              projected_rank, float(_dot(primary, primary).sqrt().item())),
        )
    return result


def apply_branch_gradients(prepared):
    """Replace encoder .grad AFTER weighted_loss.backward()."""
    for params, grads, _ in prepared.values():
        for param, grad in zip(params, grads):
            param.grad = grad.clone()
    return {branch: item[2] for branch, item in prepared.items()}
