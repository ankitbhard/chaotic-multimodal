"""HAM10000 model variants for dermoscopic image + metadata classification.

Key design: expose `.gene_encoder` as a property aliased to `._meta_encoder`
so that `prepare_branch_gradients` (which accesses `model.image_encoder` and
`model.gene_encoder`) works without modifying shared SAMO code.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models.fusion import CrossModalAttention, FUSED_DIM


class ResNet18Encoder(nn.Module):
    """Pretrained ResNet-18 with final FC removed. Output: [B, 512]."""
    FEATURE_DIM = 512

    def __init__(self, pretrained=True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        base = tv_models.resnet18(weights=weights)
        # Remove the final FC layer
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.feature_dim = self.FEATURE_DIM

    def forward(self, x):
        """x: [B, 3, 224, 224] -> [B, 512]"""
        return self.features(x).squeeze(-1).squeeze(-1)


class MetadataEncoder(nn.Module):
    """3-layer MLP for tabular metadata. Output: [B, 64]."""
    FEATURE_DIM = 64

    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.feature_dim = self.FEATURE_DIM
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """x: [B, input_dim] -> [B, 64]"""
        return self.encoder(x)


class AuxClassHead(nn.Module):
    """Auxiliary classification head: feat -> hidden -> n_classes."""

    def __init__(self, input_dim, n_classes, hidden_dim=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.head(x)  # [B, n_classes]


class HAM10000Model(nn.Module):
    """Multimodal model: ResNet18 (image) + MetadataEncoder → CrossModalAttention → 7-class.

    Exposes `.gene_encoder` as a property aliased to `._meta_encoder`
    for compatibility with `prepare_branch_gradients`.
    """

    def __init__(self, meta_dim, n_classes=7, dropout=0.3, pretrained=True):
        super().__init__()
        self.image_encoder = ResNet18Encoder(pretrained=pretrained)
        self._meta_encoder = MetadataEncoder(meta_dim, dropout)
        self.fusion = CrossModalAttention(
            self.image_encoder.FEATURE_DIM,
            self._meta_encoder.FEATURE_DIM,
        )
        d = FUSED_DIM
        self.shared = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.head = nn.Linear(d, n_classes)
        self._attn_weights = None

    @property
    def gene_encoder(self):
        return self._meta_encoder

    def forward(self, inputs):
        img_feat = self.image_encoder(inputs["image"])
        meta_feat = self._meta_encoder(inputs["meta"])
        fused, attn_w = self.fusion(img_feat, meta_feat)
        self._attn_weights = attn_w.detach()
        shared = self.shared(fused)
        return {"logits": self.head(shared)}

    def get_attention_weights(self):
        if self._attn_weights is None:
            raise RuntimeError("Call forward() first.")
        return self._attn_weights


class HAM10000ModelWithAux(nn.Module):
    """Multimodal model with auxiliary classification heads for SAMO/RA-SAMO.

    Returns logits, aux_img_logits, aux_meta_logits.
    Exposes `.gene_encoder` property and `aux_gene` attribute for gradient code.
    """

    def __init__(self, meta_dim, n_classes=7, dropout=0.3, pretrained=True):
        super().__init__()
        self.image_encoder = ResNet18Encoder(pretrained=pretrained)
        self._meta_encoder = MetadataEncoder(meta_dim, dropout)
        self.fusion = CrossModalAttention(
            self.image_encoder.FEATURE_DIM,
            self._meta_encoder.FEATURE_DIM,
        )
        d = FUSED_DIM
        self.shared = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.head = nn.Linear(d, n_classes)

        # Auxiliary heads (gradients flow through to encoders)
        self.aux_img = AuxClassHead(
            self.image_encoder.FEATURE_DIM, n_classes)
        self.aux_gene = AuxClassHead(
            self._meta_encoder.FEATURE_DIM, n_classes)
        self._attn_weights = None

    @property
    def gene_encoder(self):
        return self._meta_encoder

    def forward(self, inputs):
        img_feat = self.image_encoder(inputs["image"])
        meta_feat = self._meta_encoder(inputs["meta"])
        fused, attn_w = self.fusion(img_feat, meta_feat)
        self._attn_weights = attn_w.detach()
        shared = self.shared(fused)
        return {
            "logits": self.head(shared),
            "aux_img_logits": self.aux_img(img_feat),
            "aux_meta_logits": self.aux_gene(meta_feat),
        }

    def get_attention_weights(self):
        if self._attn_weights is None:
            raise RuntimeError("Call forward() first.")
        return self._attn_weights


class ImageOnlyModel(nn.Module):
    """ResNet-18 image-only baseline."""

    def __init__(self, n_classes=7, pretrained=True):
        super().__init__()
        self.encoder = ResNet18Encoder(pretrained=pretrained)
        self.head = nn.Linear(self.encoder.FEATURE_DIM, n_classes)

    def forward(self, inputs):
        feat = self.encoder(inputs["image"])
        return {"logits": self.head(feat)}


class MetaOnlyModel(nn.Module):
    """Metadata-only baseline."""

    def __init__(self, meta_dim, n_classes=7, dropout=0.3):
        super().__init__()
        self.encoder = MetadataEncoder(meta_dim, dropout)
        self.head = nn.Linear(self.encoder.FEATURE_DIM, n_classes)

    def forward(self, inputs):
        feat = self.encoder(inputs["meta"])
        return {"logits": self.head(feat)}
