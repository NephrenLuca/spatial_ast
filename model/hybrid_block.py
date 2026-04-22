"""
HybridBlock: DWConv -> (MHA || GC-Mamba) -> GLF -> CrossAttn -> FFN.

The attention / mamba dimension ratio shifts across the network depth:
  - early blocks (phase < 1/3):  70% attn, 30% mamba  (local geometry)
  - middle blocks:               50% / 50%             (balanced)
  - late blocks (phase > 2/3):   30% attn, 70% mamba   (global structure)

Follows architecture.md Section 6.3.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from model.config import ModelConfig
from model.adaln import AdaLayerNorm
from model.dwconv import DepthwiseConv1d
from model.attention import MultiHeadAttention, MultiHeadCrossAttention
from model.gc_mamba import GeometryConditionedMamba
from model.glf import GlobalLocalGatedFusion
from model.ffn import SwiGLU_FFN


class HybridBlock(nn.Module):
    """
    v2.0 hybrid block with depth-wise convolution, parallel MHA + GC-Mamba
    branches, Hadamard gated fusion, cross-attention, and SwiGLU FFN.
    """

    def __init__(self, config: ModelConfig, block_idx: int) -> None:
        super().__init__()
        total_blocks = config.num_blocks
        phase = block_idx / total_blocks

        if phase < 1 / 3:
            attn_heads = max(1, int(config.d_model * 0.7) // config.head_dim)
            mamba_d_state = max(1, int(config.d_model * 0.3))
        elif phase < 2 / 3:
            attn_heads = max(1, int(config.d_model * 0.5) // config.head_dim)
            mamba_d_state = max(1, int(config.d_model * 0.5))
        else:
            attn_heads = max(1, int(config.d_model * 0.3) // config.head_dim)
            mamba_d_state = max(1, int(config.d_model * 0.7))

        # Pre-smoothing depth-wise convolution
        self.dwconv = DepthwiseConv1d(config.d_model, kernel_size=config.dwconv_kernel)
        self.dwconv_norm = nn.LayerNorm(config.d_model)

        # MHA branch (local geometry)
        self.norm1 = AdaLayerNorm(config.d_model, config.time_embed_dim)
        self.attn = MultiHeadAttention(
            d_model=config.d_model,
            num_heads=attn_heads,
            head_dim=config.head_dim,
            use_rope=True,
        )

        # GC-Mamba branch (global structure)
        self.norm2 = AdaLayerNorm(config.d_model, config.time_embed_dim)
        self.gc_mamba = GeometryConditionedMamba(
            d_model=config.d_model,
            d_state=mamba_d_state,
            d_conv=config.mamba_conv_dim,
            expand=config.mamba_expand,
            geom_dim=config.geom_desc_dim,
        )

        # Global-local gated fusion
        self.glf = GlobalLocalGatedFusion(config.d_model)

        # Cross-attention (condition injection)
        self.norm3 = AdaLayerNorm(config.d_model, config.time_embed_dim)
        self.cross_attn = MultiHeadCrossAttention(
            d_model=config.d_model,
            num_heads=config.cross_attn_heads,
            head_dim=config.head_dim,
        )

        # FFN
        self.norm4 = AdaLayerNorm(config.d_model, config.time_embed_dim)
        self.ffn = SwiGLU_FFN(d_model=config.d_model, d_ff=config.d_ff)

    def forward(
        self,
        x: Tensor,
        t_emb: Tensor,
        cond_kv: Tensor,
        geom_desc: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x:         [B, L, d_model]
            t_emb:     [B, t_dim]
            cond_kv:   [B, L_cond, d_model]
            geom_desc: [B, L, geom_dim]
            mask:      [B, L] bool — True for valid positions
        Returns:
            [B, L, d_model]
        """
        # Step 0: depth-wise conv local smoothing
        x = x + self.dwconv_norm(self.dwconv(x))

        # Step 1: MHA (local geometry details)
        h_attn = self.norm1(x, t_emb)
        h_attn = self.attn(h_attn, h_attn, h_attn, mask=mask)

        # Step 2: GC-Mamba (global structure propagation)
        h_mamba = self.norm2(x, t_emb)
        h_mamba = self.gc_mamba(h_mamba, geom_desc)

        # Step 3: gated fusion
        h_fused = self.glf(h_global=h_mamba, h_local=h_attn)
        x = x + h_fused

        # Step 4: cross-attention (condition injection)
        h = self.norm3(x, t_emb)
        h = self.cross_attn(query=h, key=cond_kv, value=cond_kv)
        x = x + h

        # Step 5: FFN
        h = self.norm4(x, t_emb)
        h = self.ffn(h)
        x = x + h

        return x
