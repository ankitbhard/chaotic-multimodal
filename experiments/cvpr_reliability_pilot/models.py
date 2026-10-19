"""
PCRF model architecture — Perturbation-Calibrated Reliability-Guided Fusion.

Extends the existing JointModel with:
  - Modality-specific reliability estimation networks
  - Projection into a common latent space
  - Patient-specific reliability-guided fusion
  - Explicit modality availability masks
  - Auxiliary unimodal Cox heads (always present for perturbation calibration)

The learned-gating baseline uses identical architecture (same capacity)
but without reliability supervision (gamma=0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models.encoders import SlideEmbeddingEncoder, GeneEncoder
from models.fusion import FUSED_DIM, CrossModalAttention


# ── Auxiliary components ──────────────────────────────────────────────────────

class ReliabilityNet(nn.Module):
    """Small MLP that maps encoder features to a scalar reliability logit."""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)  # [B, 1]


class AuxCoxHead(nn.Module):
    """Unimodal Cox prediction head: feat → hidden → risk scalar."""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)  # [B]


# ── PCRF Model ────────────────────────────────────────────────────────────────

class PCRFModel(nn.Module):
    """
    Perturbation-Calibrated Reliability-Guided Fusion.

    z_I = ImageEncoder(x_I)           -> [B, 512]
    z_G = GeneEncoder(x_G)            -> [B, 128]
    q_I = R_I(z_I), q_G = R_G(z_G)   -> reliability logits
    [alpha_I, alpha_G] = softmax([q_I, q_G])
    z_F = alpha_I * P_I(z_I) + alpha_G * P_G(z_G)
    survival = head(shared(z_F))

    Args:
        n_genes: number of gene features after preprocessing
        image_dim: slide embedding dim (1024 UNI, 1536 UNI2-h)
        dropout: dropout rate
        use_aux_loss: if True, aux Cox loss gradients flow through encoders
        use_reliability_loss: if True, enables perturbation calibration
        reliability_hidden: hidden dim for reliability MLPs
    """

    def __init__(
        self,
        n_genes: int,
        image_dim: int = 1024,
        dropout: float = 0.3,
        use_aux_loss: bool = True,
        use_reliability_loss: bool = True,
        reliability_hidden: int = 64,
    ):
        super().__init__()

        self.image_encoder = SlideEmbeddingEncoder(image_dim, dropout)
        self.gene_encoder = GeneEncoder(n_genes, dropout)

        img_dim = self.image_encoder.FEATURE_DIM   # 512
        gene_dim = self.gene_encoder.FEATURE_DIM   # 128
        d = FUSED_DIM                              # 256

        # Projection to common latent space
        self.proj_img = nn.Linear(img_dim, d)
        self.proj_gene = nn.Linear(gene_dim, d)

        # Reliability estimation networks
        self.reliability_img = ReliabilityNet(img_dim, reliability_hidden)
        self.reliability_gene = ReliabilityNet(gene_dim, reliability_hidden)

        # Shared representation + survival head (same as JointModel)
        self.shared = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.head_survival = nn.Linear(d, 2)

        # Auxiliary Cox heads (always present for perturbation calibration)
        self.aux_img = AuxCoxHead(img_dim)
        self.aux_gene = AuxCoxHead(gene_dim)

        self.use_aux_loss = use_aux_loss
        self.use_reliability_loss = use_reliability_loss

        # Cached for perturbation calibration and diagnostics
        self._img_feat = None
        self._gene_feat = None
        self._reliability_weights = None

    def forward(self, inputs, modality_mask=None):
        """
        Args:
            inputs: {"image": [B, image_dim], "gene": [B, n_genes]}
            modality_mask: optional [B, 2] binary (1=present, 0=missing)
        Returns:
            dict with keys: survival, reliability_weights, reliability_logits,
                           aux_img_risk, aux_gene_risk
        """
        img_feat = self.image_encoder(inputs["image"])    # [B, 512]
        gene_feat = self.gene_encoder(inputs["gene"])     # [B, 128]

        # Cache for perturbation calibration
        self._img_feat = img_feat
        self._gene_feat = gene_feat

        # Reliability logits
        q_img = self.reliability_img(img_feat)            # [B, 1]
        q_gene = self.reliability_gene(gene_feat)         # [B, 1]
        logits = torch.cat([q_img, q_gene], dim=1)       # [B, 2]

        # Apply modality mask (set missing to -inf before softmax)
        if modality_mask is not None:
            logits = logits + (modality_mask.float() - 1.0) * 1e9

        weights = F.softmax(logits, dim=1)                # [B, 2]
        self._reliability_weights = weights.detach()

        # Project to common space and fuse
        p_img = self.proj_img(img_feat)                   # [B, 256]
        p_gene = self.proj_gene(gene_feat)                # [B, 256]

        fused = weights[:, 0:1] * p_img + weights[:, 1:2] * p_gene

        shared = self.shared(fused)
        survival = self.head_survival(shared)

        # Auxiliary risk predictions
        # Detach encoder features if aux loss should NOT regularize encoders
        if self.use_aux_loss:
            aux_img_risk = self.aux_img(img_feat)
            aux_gene_risk = self.aux_gene(gene_feat)
        else:
            aux_img_risk = self.aux_img(img_feat.detach())
            aux_gene_risk = self.aux_gene(gene_feat.detach())

        return {
            "survival": survival,
            "reliability_weights": weights,
            "reliability_logits": logits,
            "aux_img_risk": aux_img_risk,
            "aux_gene_risk": aux_gene_risk,
        }

    def get_encoder_features(self):
        """Returns cached encoder outputs for perturbation calibration."""
        return self._img_feat, self._gene_feat

    def get_reliability_weights(self):
        return self._reliability_weights


# ── JointModel with modality dropout ─────────────────────────────────────────

class JointModelWithDropout(nn.Module):
    """
    Wraps the existing attention-fusion model with training-time modality dropout.

    During training, with probability p_drop:
      - 50% chance: zero out image features
      - 50% chance: zero out gene features
    Forces the model to handle missing modalities gracefully.
    """

    def __init__(self, n_genes, image_dim=1024, dropout=0.3, p_drop=0.15):
        super().__init__()
        self.image_encoder = SlideEmbeddingEncoder(image_dim, dropout)
        self.gene_encoder = GeneEncoder(n_genes, dropout)
        self.fusion = CrossModalAttention(
            self.image_encoder.FEATURE_DIM,
            self.gene_encoder.FEATURE_DIM,
        )
        d = FUSED_DIM
        self.shared = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.head_survival = nn.Linear(d, 2)
        self.p_drop = p_drop
        self._attn_weights = None

    def forward(self, inputs, modality_mask=None):
        img_feat = self.image_encoder(inputs["image"])
        gene_feat = self.gene_encoder(inputs["gene"])

        # Training-time modality dropout
        if self.training and self.p_drop > 0:
            B = img_feat.size(0)
            drop = torch.rand(B, device=img_feat.device) < self.p_drop
            which = torch.rand(B, device=img_feat.device) < 0.5
            drop_img = drop & which
            drop_gene = drop & ~which
            img_feat = img_feat * (~drop_img).float().unsqueeze(1)
            gene_feat = gene_feat * (~drop_gene).float().unsqueeze(1)

        # Apply explicit modality mask at test time
        if modality_mask is not None:
            img_feat = img_feat * modality_mask[:, 0:1]
            gene_feat = gene_feat * modality_mask[:, 1:2]

        fused, attn_w = self.fusion(img_feat, gene_feat)
        self._attn_weights = attn_w.detach()
        shared = self.shared(fused)
        return {"survival": self.head_survival(shared)}

    def get_attention_weights(self):
        return self._attn_weights


# ── JointModel with auxiliary Cox heads ──────────────────────────────────────

class JointModelWithAux(nn.Module):
    """
    Standard attention fusion + auxiliary unimodal Cox heads.
    From Phase 1: aux heads alone gave the only significant improvement (p=0.007).
    """

    def __init__(self, n_genes, image_dim=1024, dropout=0.3):
        super().__init__()
        self.image_encoder = SlideEmbeddingEncoder(image_dim, dropout)
        self.gene_encoder = GeneEncoder(n_genes, dropout)
        self.fusion = CrossModalAttention(
            self.image_encoder.FEATURE_DIM,
            self.gene_encoder.FEATURE_DIM,
        )
        d = FUSED_DIM
        self.shared = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.head_survival = nn.Linear(d, 2)

        # Auxiliary Cox heads (features NOT detached — gradient flows to encoder)
        self.aux_img = AuxCoxHead(self.image_encoder.FEATURE_DIM)
        self.aux_gene = AuxCoxHead(self.gene_encoder.FEATURE_DIM)
        self._attn_weights = None

    def forward(self, inputs, modality_mask=None):
        img_feat = self.image_encoder(inputs["image"])
        gene_feat = self.gene_encoder(inputs["gene"])

        if modality_mask is not None:
            img_feat = img_feat * modality_mask[:, 0:1]
            gene_feat = gene_feat * modality_mask[:, 1:2]

        fused, attn_w = self.fusion(img_feat, gene_feat)
        self._attn_weights = attn_w.detach()
        shared = self.shared(fused)

        return {
            "survival": self.head_survival(shared),
            "aux_img_risk": self.aux_img(img_feat),
            "aux_gene_risk": self.aux_gene(gene_feat),
        }

    def get_attention_weights(self):
        return self._attn_weights


# ── Factory ──────────────────────────────────────────────────────────────────

def build_pilot_model(method, n_genes, image_dim=1024, dropout=0.3):
    """
    Build model for a given pilot method.

    Returns: (model, method_config_dict)
    """
    from models.joint_model import JointModel

    configs = {
        "adamw": {
            "model_cls": "joint",
            "use_aux_loss": False,
            "use_reliability_loss": False,
            "mod_dropout": 0.0,
        },
        "aux_cox": {
            "model_cls": "joint_aux",
            "use_aux_loss": True,
            "use_reliability_loss": False,
            "mod_dropout": 0.0,
        },
        "mod_dropout": {
            "model_cls": "joint_dropout",
            "use_aux_loss": False,
            "use_reliability_loss": False,
            "mod_dropout": 0.15,
        },
        "learned_gating": {
            "model_cls": "pcrf",
            "use_aux_loss": False,
            "use_reliability_loss": False,
            "mod_dropout": 0.0,
        },
        "learned_gating_aux": {
            "model_cls": "pcrf",
            "use_aux_loss": True,
            "use_reliability_loss": False,
            "mod_dropout": 0.0,
        },
        "pcrf": {
            "model_cls": "pcrf",
            "use_aux_loss": False,
            "use_reliability_loss": True,
            "mod_dropout": 0.0,
        },
        "pcrf_aux": {
            "model_cls": "pcrf",
            "use_aux_loss": True,
            "use_reliability_loss": True,
            "mod_dropout": 0.0,
        },
        "pcrf_aux_warmup": {
            "model_cls": "pcrf",
            "use_aux_loss": True,
            "use_reliability_loss": True,
            "mod_dropout": 0.0,
            "warmup_epochs": 20,
        },
    }

    cfg = configs[method]

    if cfg["model_cls"] == "joint":
        model = JointModel(n_genes, image_dim, "attention", dropout)
    elif cfg["model_cls"] == "joint_aux":
        model = JointModelWithAux(n_genes, image_dim, dropout)
    elif cfg["model_cls"] == "joint_dropout":
        model = JointModelWithDropout(n_genes, image_dim, dropout, cfg["mod_dropout"])
    elif cfg["model_cls"] == "pcrf":
        model = PCRFModel(
            n_genes, image_dim, dropout,
            use_aux_loss=cfg["use_aux_loss"],
            use_reliability_loss=cfg["use_reliability_loss"],
        )
    else:
        raise ValueError(f"Unknown model class: {cfg['model_cls']}")

    total = sum(p.numel() for p in model.parameters())
    print(f"  Model [{method}]: {total:,} params")

    return model, cfg


ALL_METHODS = ["adamw", "aux_cox", "mod_dropout", "learned_gating", "pcrf", "pcrf_aux"]
PHASE2_METHODS = ["learned_gating_aux", "pcrf_aux", "pcrf_aux_warmup"]
BASELINE_METHODS = ["adamw", "aux_cox", "mod_dropout"]
ABLATION_METHODS = ALL_METHODS + ["learned_gating_aux", "pcrf_aux_warmup"]
