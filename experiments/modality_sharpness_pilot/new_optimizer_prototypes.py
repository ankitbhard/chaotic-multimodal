"""Experimental optimization baselines for multimodal Cox survival training.

These are research prototypes, not reproductions of RGM, DGL or Polyak-SAM.
"""
import torch
import torch.nn.functional as F


class OptimizerExperiment:
    def __init__(self,model,mode,lr=1e-3,weight_decay=0.,conflict_weight=0.25,
                 auxiliary_weight=0.1):
        if mode not in ("sgd","adamw","conflict_projection","encoder_auxiliary",
                        "adaptive_step"):
            raise ValueError(mode)
        self.model=model
        self.mode=mode
        self.conflict_weight=float(conflict_weight)
        self.auxiliary_weight=float(auxiliary_weight)
        self.base=(torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
                   if mode=="adamw" else
                   torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,
                                   weight_decay=weight_decay))
        self.steps=0
        self.history=[]
        self.reference_lr=float(lr)
        self._loss_ema=None

    def step(self,closure):
        self.base.zero_grad(set_to_none=True)
        loss=closure()
        loss.backward()
        grad_cos=None
        if self.mode=="conflict_projection":
            # Common space: gradient of Cox loss wrt modality-specific
            # projected fusion vectors. For the current attention fusion,
            # these are [B,256] and share coordinates. These are NOT
            # independent modality objectives; interpret as a diagnostic.
            fusion=self.model.fusion
            gi=getattr(fusion,"_research_img_grad",None)
            gg=getattr(fusion,"_research_gene_grad",None)
            if gi is not None and gg is not None:
                a=gi.flatten();b=gg.flatten()
                grad_cos=float(F.cosine_similarity(a[None],b[None]).detach())
                if grad_cos<0:
                    # Do not project parameter gradients across disjoint encoders.
                    # Projection of shared-space gradients requires custom
                    # backward plumbing, so diagnostic-only by design.
                    pass
        elif self.mode=="adaptive_step":
            # Bounded loss-EMA relative scaling; a heuristic baseline, not
            # the Polyak step size (which requires a valid lower bound).
            val=float(loss.detach())
            self._loss_ema=val if self._loss_ema is None else .9*self._loss_ema+.1*val
            ratio=self._loss_ema/max(abs(val),1e-8)
            scale=min(1.5,max(.5,ratio))
            for group in self.base.param_groups:
                group["lr"]=self.reference_lr*scale
        self.base.step()
        self.steps+=1
        self.history.append({"step":self.steps,"loss":float(loss.detach()),
                             "gradient_cosine":grad_cos,
                             "lr":self.base.param_groups[0]["lr"]})
        return loss.detach()


def install_shared_gradient_hooks(model):
    """Observe same-space modality gradients in attention fusion.

    Hooks are valid only for attention fusion. They retain no graph and
    require no extra forward/backward pass. Call before training.
    """
    fusion=model.fusion
    if not hasattr(fusion,"img_proj") or not hasattr(fusion,"gene_proj"):
        raise ValueError("Gradient diagnostic requires attention fusion")
    fusion._research_img_grad=None
    fusion._research_gene_grad=None
    def image_hook(module,inp,out):
        fusion._research_img_grad=None
        if out.requires_grad:
            out.register_hook(lambda g: setattr(fusion,"_research_img_grad",g.detach().clone()))
    def gene_hook(module,inp,out):
        fusion._research_gene_grad=None
        if out.requires_grad:
            out.register_hook(lambda g: setattr(fusion,"_research_gene_grad",g.detach().clone()))
    return (fusion.img_proj.register_forward_hook(image_hook),
            fusion.gene_proj.register_forward_hook(gene_hook))
