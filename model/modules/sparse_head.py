"""SPLADE-style learned sparse projection for ReID features.

Produces sparse high-dimensional representations where most dimensions are zero.
Occluded body parts naturally activate fewer dimensions, enabling
automatic partial matching through dot-product similarity.

Reference: Formal et al., "SPLADE: Sparse Lexical and Expansion Model
for First Stage Ranking", SIGIR 2021.
"""

import torch
import torch.nn as nn


class SparseProjectionHead(nn.Module):
    """Projects dense features to sparse high-dimensional space."""

    def __init__(self, input_dim=768, sparse_dim=2048):
        super().__init__()
        self.proj = nn.Linear(input_dim, sparse_dim)
        # Initialize small to encourage initial sparsity
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        """
        Args:
            x: (B, D) dense feature
        Returns:
            sparse_feat: (B, sparse_dim) sparse feature (log1p-relu activated)
            sparsity: scalar, mean activation (for regularization logging)
        """
        sparse_feat = torch.log1p(torch.relu(self.proj(x)))
        with torch.no_grad():
            sparsity = (sparse_feat > 0).float().mean().item()
        return sparse_feat, sparsity
