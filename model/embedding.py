"""
Input embedding layer with Rotary Positional Encoding.

Seven additive embeddings (token, depth, type, role, parent, sibling, geom)
are summed, then RoPE is applied, followed by LayerNorm + Dropout.
Follows architecture.md Section 6.2.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from model.config import ModelConfig


class RotaryPositionalEncoding(nn.Module):
    """Standard Rotary Positional Encoding (RoPE) applied to hidden states."""

    def __init__(self, dim: int, max_len: int = 4096) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._max_len = max_len
        self._cos_cache: Tensor | None = None
        self._sin_cache: Tensor | None = None

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        if self._cos_cache is not None and self._cos_cache.size(0) >= seq_len:
            return
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)          # [L, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)         # [L, D]
        self._cos_cache = emb.cos()[None, :, :]          # [1, L, D]
        self._sin_cache = emb.sin()[None, :, :]

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, L, D] -> [B, L, D]"""
        seq_len = x.size(1)
        self._build_cache(seq_len, x.device)
        cos = self._cos_cache[:, :seq_len, :].to(x.dtype)
        sin = self._sin_cache[:, :seq_len, :].to(x.dtype)
        return x * cos + self._rotate_half(x) * sin


class SpatialASTEmbedding(nn.Module):
    """7-way additive embedding + RoPE + LayerNorm + Dropout."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model

        self.token_embed   = nn.Embedding(config.vocab_size, d)
        self.depth_embed   = nn.Embedding(config.max_depth + 1, d)
        self.type_embed    = nn.Embedding(config.num_node_types, d)
        self.role_embed    = nn.Embedding(config.num_roles, d)
        self.parent_embed  = nn.Embedding(config.num_node_types, d)
        self.sibling_embed = nn.Embedding(config.max_siblings, d)
        self.geom_proj     = nn.Linear(config.geom_desc_dim, d)

        self.rope = RotaryPositionalEncoding(d)
        self.layer_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.embed_dropout)

    def forward(
        self,
        token_ids: Tensor,      # [B, L]
        depth_ids: Tensor,      # [B, L]
        type_ids: Tensor,       # [B, L]
        role_ids: Tensor,       # [B, L]
        parent_ids: Tensor,     # [B, L]
        sibling_ids: Tensor,    # [B, L]
        geom_desc: Tensor,      # [B, L, 4]
    ) -> Tensor:
        """Returns [B, L, d_model]."""
        x = (
            self.token_embed(token_ids)
            + self.depth_embed(depth_ids)
            + self.type_embed(type_ids)
            + self.role_embed(role_ids)
            + self.parent_embed(parent_ids)
            + self.sibling_embed(sibling_ids)
            + self.geom_proj(geom_desc)
        )
        x = self.rope(x)
        return self.dropout(self.layer_norm(x))
