"""Classification loss adapter for SAMO/RA-SAMO.

Returns 4 loss tensors (fusion_ce, img_aux_ce, meta_aux_ce, zero_ranking)
matching the signature expected by `LossWeightController` and
`prepare_branch_gradients`.

Ranking loss is survival-specific (ordering by time) with no natural
classification analog — always returns a zero tensor.
"""

import torch
import torch.nn.functional as F


def classification_loss_parts(outputs, labels, method):
    """Compute 4-part loss matching SAMO interface.

    Args:
        outputs: dict with "logits" and optionally "aux_img_logits", "aux_meta_logits"
        labels: integer class labels [B]
        method: one of ("adamw", "sam", "adaptive_aux", "samo", "ra_samo",
                        "image_only_adamw", "meta_only_adamw")

    Returns:
        (fusion_ce, img_aux_ce, meta_aux_ce, zero_ranking) — 4 scalar tensors
    """
    fusion_ce = F.cross_entropy(outputs["logits"], labels)
    zero = fusion_ce * 0.0  # zero tensor on same device with grad connection

    has_aux = method in ("adaptive_aux", "samo", "ra_samo")

    if has_aux and "aux_img_logits" in outputs:
        img_aux_ce = F.cross_entropy(outputs["aux_img_logits"], labels)
    else:
        img_aux_ce = zero

    if has_aux and "aux_meta_logits" in outputs:
        meta_aux_ce = F.cross_entropy(outputs["aux_meta_logits"], labels)
    else:
        meta_aux_ce = zero

    # No ranking loss in classification — always zero
    ranking = zero

    return fusion_ce, img_aux_ce, meta_aux_ce, ranking
