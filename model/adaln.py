"""
Adaptive Layer Normalization (AdaLN).

Modulates LayerNorm scale and shift using the diffusion timestep embedding,
following architecture.md Section 6.4.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class AdaLayerNorm(nn.Module):
    """LayerNorm whose affine parameters are predicted from a timestep embedding."""

    def __init__(self, d_model: int, t_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(t_dim, 2 * d_model)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        Args:
            x:     [B, L, d_model]
            t_emb: [B, t_dim]
        Returns:
            [B, L, d_model]
        """
        scale, shift = self.proj(t_emb).chunk(2, dim=-1)   # each [B, d_model]
        return self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
