"""CLAM-style gated attention MIL models for survival prediction.

Replaces mean-pooling with learned attention aggregation while preserving
the downstream gene encoder, cross-modal fusion, and survival head.

Key design: CLAMImageEncoder is assigned to model.image_encoder so that
RA-SAMO's prepare_branch_gradients (which calls getattr(model, "image_encoder"))
naturally includes all attention parameters in gradient projection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models.encoders import GeneEncoder
from models.fusion import CrossModalAttention, FUSED_DIM
from experiments.cvpr_reliability_pilot.models import AuxCoxHead


class GatedAttentionAggregator(nn.Module):
    """Gated attention MIL aggregator (CLAM-SB, no instance clustering).

    a_k = softmax( V^T (tanh(W1 h_k) * sigmoid(W2 h_k)) )
    z   = sum_k  a_k * h_k

    Input:  patches [N, input_dim]
    Output: z [input_dim], attention_weights [N]
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256,
                 dropout: float = 0.25):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout))
        self.attention_b = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.attention_v = nn.Linear(hidden_dim, 1)

    def forward(self, patches: torch.Tensor):
        a = self.attention_a(patches)           # [N, hidden]
        b = self.attention_b(patches)           # [N, hidden]
        scores = self.attention_v(a * b)        # [N, 1]
        weights = F.softmax(scores, dim=0)      # [N, 1]
        z = (weights * patches).sum(dim=0)      # [input_dim]
        return z, weights.squeeze(-1)           # [input_dim], [N]


class CLAMImageEncoder(nn.Module):
    """Attention MIL aggregation + linear projection.

    Replaces SlideEmbeddingEncoder for patch-level input.
    Outputs [B, 512] — same FEATURE_DIM as SlideEmbeddingEncoder.

    Input:  all_patches [total_patches, input_dim], patch_counts [B]
    Output: [B, FEATURE_DIM]
    """
    FEATURE_DIM = 512

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256,
                 dropout: float = 0.25):
        super().__init__()
        self.feature_dim = self.FEATURE_DIM
        self.aggregator = GatedAttentionAggregator(input_dim, hidden_dim, dropout)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, self.FEATURE_DIM),
            nn.LayerNorm(self.FEATURE_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self._last_attentions = None

    def forward(self, all_patches: torch.Tensor,
                patch_counts: torch.Tensor) -> torch.Tensor:
        slides = torch.split(all_patches, patch_counts.tolist())
        embeddings = []
        attentions = []
        for slide in slides:
            z, att = self.aggregator(slide)
            embeddings.append(z)
            attentions.append(att.detach())
        self._last_attentions = attentions
        stacked = torch.stack(embeddings)       # [B, input_dim]
        return self.projection(stacked)         # [B, FEATURE_DIM]

    def get_attention_weights(self):
        """Per-slide attention weights from the last forward pass."""
        return self._last_attentions


class JointModelCLAM(nn.Module):
    """CLAM image encoder + gene encoder + fusion + survival head. No aux."""

    def __init__(self, n_genes: int, input_dim: int = 1024,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.image_encoder = CLAMImageEncoder(input_dim, hidden_dim, dropout)
        self.gene_encoder = GeneEncoder(n_genes, dropout)
        self.fusion = CrossModalAttention(
            self.image_encoder.FEATURE_DIM, self.gene_encoder.FEATURE_DIM)
        d = FUSED_DIM
        self.shared = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.head_survival = nn.Linear(d, 2)
        self._attn_weights = None

    def forward(self, inputs: dict) -> dict:
        img_feat = self.image_encoder(
            inputs["patches"], inputs["patch_counts"])
        gene_feat = self.gene_encoder(inputs["gene"])
        fused, attn_w = self.fusion(img_feat, gene_feat)
        self._attn_weights = attn_w.detach()
        shared = self.shared(fused)
        return {"survival": self.head_survival(shared)}

    def get_attention_weights(self):
        return self._attn_weights


class JointModelCLAMWithAux(nn.Module):
    """CLAM image encoder + gene encoder + fusion + survival + aux Cox heads."""

    def __init__(self, n_genes: int, input_dim: int = 1024,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.image_encoder = CLAMImageEncoder(input_dim, hidden_dim, dropout)
        self.gene_encoder = GeneEncoder(n_genes, dropout)
        self.fusion = CrossModalAttention(
            self.image_encoder.FEATURE_DIM, self.gene_encoder.FEATURE_DIM)
        d = FUSED_DIM
        self.shared = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.head_survival = nn.Linear(d, 2)
        self.aux_img = AuxCoxHead(self.image_encoder.FEATURE_DIM)
        self.aux_gene = AuxCoxHead(self.gene_encoder.FEATURE_DIM)
        self._attn_weights = None

    def forward(self, inputs: dict) -> dict:
        img_feat = self.image_encoder(
            inputs["patches"], inputs["patch_counts"])
        gene_feat = self.gene_encoder(inputs["gene"])
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
