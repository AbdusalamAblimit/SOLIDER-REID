"""Cross-Part Self-Attention (CPSA) module.

After Gaussian pooling extracts part features, this module applies self-attention
so that visible parts can share information with occluded parts. Unlike PPTD
(which uses cross-attention from random queries), CPSA operates on already
meaningful Gaussian-pooled features, requiring minimal learning.

Key design choices:
1. Input is already good (Gaussian-pooled part features, not random queries)
2. Only self-attention (no cross-attention) → much fewer parameters
3. Visibility-aware: attention is modulated by part visibility scores
4. Residual connection: output = input + refined, so at worst it's identity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossPartSelfAttention(nn.Module):
    """Single self-attention layer for cross-part feature refinement.

    Args:
        d_model: Feature dimension (must match part feature dim)
        nhead: Number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        use_vis_mask: Whether to mask attention based on part visibility
    """

    def __init__(self, d_model=768, nhead=8, dim_feedforward=768, dropout=0.1,
                 use_vis_mask=True):
        super().__init__()
        self.use_vis_mask = use_vis_mask

        # Self-attention: parts attend to each other
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Learnable gate: controls how much the refined features contribute
        # Initialized to -3 → sigmoid(-3) ≈ 0.05 → near-identity at start
        self.gate = nn.Parameter(torch.tensor(-3.0))

    def forward(self, part_feats, part_vis=None):
        """
        Args:
            part_feats: [B, K, C] Gaussian-pooled part features
            part_vis: [B, K] part visibility scores (optional)

        Returns:
            refined_feats: [B, K, C] refined part features
        """
        # Pre-norm self-attention
        q = self.norm1(part_feats)

        # Optional visibility-aware attention mask
        # Low-visibility parts should attend to high-visibility parts
        # but not vice versa (high-vis parts are reliable, don't corrupt them)
        attn_mask = None
        if self.use_vis_mask and part_vis is not None:
            # Soft mask: scale attention by source visibility
            # key_vis [B, K] → [B, 1, K] (broadcast over queries)
            # Higher vis = higher attention weight
            # Use additive bias: log(vis) added to attention logits
            vis_bias = torch.log(part_vis.clamp(min=0.1))  # [B, K]
            vis_bias = vis_bias.unsqueeze(1).expand(-1, part_feats.shape[1], -1)  # [B, K, K]
            # Expand for multihead: [B*nhead, K, K]
            nhead = self.self_attn.num_heads
            attn_mask = vis_bias.unsqueeze(1).expand(-1, nhead, -1, -1)
            attn_mask = attn_mask.reshape(-1, part_feats.shape[1], part_feats.shape[1])

        attn_out, _ = self.self_attn(q, q, q, attn_mask=attn_mask)

        # Gated residual: gate starts at 0 (identity), learns to blend in refinement
        gate = torch.sigmoid(self.gate)
        part_feats = part_feats + gate * attn_out

        # FFN with residual
        part_feats = part_feats + gate * self.ffn(self.norm2(part_feats))

        return part_feats
