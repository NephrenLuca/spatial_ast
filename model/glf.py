"""
Global-Local Gated Fusion (GLF).

Hadamard gating that fuses Mamba (global) and Attention (local) outputs,
following architecture.md Section 6.6 / GeoFusion GSM gating.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GlobalLocalGatedFusion(nn.Module):
    """
    out = LayerNorm( out_proj( h_global * sigmoid(gate_proj(h_local)) ) )

    h_global (Mamba):     captures long-range structural dependencies
    h_local  (Attention): captures precise local geometry
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_global: Tensor, h_local: Tensor) -> Tensor:
        """
        Args:
            h_global: [B, L, D]  (Mamba output)
            h_local:  [B, L, D]  (Attention output)
        Returns:
            [B, L, D]
        """
        gate = torch.sigmoid(self.gate_proj(h_local))
        fused = h_global * gate
        return self.norm(self.out_proj(fused))
