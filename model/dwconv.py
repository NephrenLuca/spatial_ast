"""
Depth-wise 1-D convolution for local smoothing.

Captures adjacent-token continuity (e.g. consecutive curve segments),
following architecture.md Section 6.5.1 / GeoFusion DWConv.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class DepthwiseConv1d(nn.Module):
    """Per-channel 1-D convolution with SiLU activation."""

    def __init__(self, d_model: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, L, D] -> [B, L, D]"""
        return self.act(self.conv(x.transpose(1, 2)).transpose(1, 2))
