"""Autograd block Hessian-vector diagnostics, independent of optimizer updates."""
import math
import torch


def block_hvp_curvatures(closure, groups, *, rng_before=None, rng_after=None,
                         set_rng_state=None):
    """Compute v^T H_mm v for each encoder, v=detached(g_m/||g_m||).

    Uses a single differentiable forward and nested autograd.grad calls.
    Does not write parameter .grad, perturb weights, or update the optimizer.
    Caller is responsible for keeping batch/model mode and mutable buffers fixed.
    """
    if rng_before is not None and set_rng_state is not None:
        set_rng_state(rng_before)
    try:
        with torch.enable_grad():
            loss=closure()
            if loss.ndim != 0:
                raise ValueError("HVP closure must return scalar loss")
            if not bool(torch.isfinite(loss.detach()).item()):
                return {name: {"status": "nonfinite_loss"} for name in groups}
            all_params=[p for params in groups.values() for p in params if p.requires_grad]
            if not all_params:
                return {name: {"status": "no_parameters"} for name in groups}
            grads=torch.autograd.grad(loss,all_params,create_graph=True,
                                      allow_unused=True)
            gradient_by_id={id(p):g for p,g in zip(all_params,grads)}
            results={}
            for name,params in groups.items():
                pairs=[(p,gradient_by_id.get(id(p))) for p in params if p.requires_grad]
                pairs=[(p,g) for p,g in pairs if g is not None]
                if not pairs:
                    results[name]={"status":"no_direction"}
                    continue
                norm_sq=sum(g.detach().double().square().sum().item() for _,g in pairs)
                norm=math.sqrt(norm_sq)
                if not math.isfinite(norm) or norm<1e-12:
                    results[name]={"status":"no_direction","gradient_norm":norm}
                    continue
                ps=[p for p,_ in pairs]
                gs=[g for _,g in pairs]
                vs=[(g.detach()/norm).detach() for g in gs]
                # Scalar directional first derivative; differentiate again.
                directional=sum((g*v).sum() for g,v in zip(gs,vs))
                if directional.requires_grad:
                    hv=torch.autograd.grad(directional,ps,allow_unused=True,
                                           retain_graph=True)
                    kappa=sum((v*h).sum().detach().double().item()
                              for v,h in zip(vs,hv) if h is not None)
                else:
                    kappa=0.0
                results[name]={
                    "status":"ok" if math.isfinite(kappa) else "nonfinite_curvature",
                    "gradient_norm":norm,
                    "directional_curvature_hvp":kappa}
            return results
    finally:
        if rng_after is not None and set_rng_state is not None:
            set_rng_state(rng_after)
