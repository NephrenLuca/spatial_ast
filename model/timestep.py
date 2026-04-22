"""
Timestep encoder: sinusoidal positional encoding -> 2-layer MLP.

Produces a per-sample conditioning vector from the diffusion timestep,
following architecture.md Section 6.8.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class TimestepEncoder(nn.Module):

    def __init__(self, d_model: int, t_dim: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: [B] integer timesteps
        Returns:
            [B, t_dim]
        """
        half_dim = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / half_dim
        )
        emb = t[:, None].float() * freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # [B, d_model]
        return self.mlp(emb)
