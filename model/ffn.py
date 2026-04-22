"""
SwiGLU feed-forward network.

Implements  out = W_down( SiLU(W_gate(x)) * W_up(x) )
following architecture.md Section 6.3 FFN sub-layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SwiGLU_FFN(nn.Module):

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, L, d_model] -> [B, L, d_model]"""
        return self.w_down(self.act(self.w_gate(x)) * self.w_up(x))
