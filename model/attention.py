"""
Multi-head self-attention and cross-attention.

Uses PyTorch 2.0+ ``F.scaled_dot_product_attention`` which auto-dispatches
to FlashAttention / memory-efficient kernels on GPU.
Follows architecture.md Section 6.3.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.embedding import RotaryPositionalEncoding


class MultiHeadAttention(nn.Module):
    """Self-attention with optional RoPE on Q/K."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        use_rope: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        self.q_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.v_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, d_model, bias=False)

        self.rope = RotaryPositionalEncoding(head_dim) if use_rope else None
        self.dropout = dropout

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            q, k, v: [B, L, d_model]
            mask:     [B, L] bool — True for valid positions
        Returns:
            [B, L, d_model]
        """
        B, L, _ = q.shape

        q = self.q_proj(q).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: [B, H, L, D_h]

        if self.rope is not None:
            q = self._apply_rope(q)
            k = self._apply_rope(k)

        attn_mask = None
        if mask is not None:
            # mask: [B, L] -> [B, 1, 1, L_k]  (broadcast over heads and query positions)
            attn_mask = mask[:, None, None, :].to(dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf"))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out)

    def _apply_rope(self, x: Tensor) -> Tensor:
        """Apply RoPE independently per head: x is [B, H, L, D_h]."""
        B, H, L, D = x.shape
        x = x.reshape(B * H, L, D)      # treat each head as a batch item
        x = self.rope(x)
        return x.view(B, H, L, D)


class MultiHeadCrossAttention(nn.Module):
    """Cross-attention: Q from decoder, K/V from condition encoder."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        self.q_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.v_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, d_model, bias=False)
        self.dropout = dropout

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        """
        Args:
            query: [B, L_q, d_model]
            key:   [B, L_kv, d_model]
            value: [B, L_kv, d_model]
        Returns:
            [B, L_q, d_model]
        """
        B, L_q, _ = query.shape
        L_kv = key.size(1)

        q = self.q_proj(query).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, L_kv, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, L_q, -1)
        return self.out_proj(out)
