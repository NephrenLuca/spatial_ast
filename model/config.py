"""
Model configuration dataclass.

Mirrors architecture.md Section 6.10 with all hyperparameters for
SpatialASTDenoiser and its sub-modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # ── Vocabulary ────────────────────────────────────────────────
    vocab_size: int = 304
    max_seq_len: Optional[int] = None  # variable length; DataLoader pads dynamically

    # ── Embedding (7-way additive) ────────────────────────────────
    d_model: int = 768
    max_depth: int = 5
    num_node_types: int = 17
    num_roles: int = 6
    max_siblings: int = 64
    geom_desc_dim: int = 4
    embed_dropout: float = 0.1

    # ── Transformer ───────────────────────────────────────────────
    num_blocks: int = 18
    head_dim: int = 64
    d_ff: int = 3072              # 4 * d_model
    cross_attn_heads: int = 8

    # ── Mamba (geometry-conditioned) ──────────────────────────────
    mamba_conv_dim: int = 4
    mamba_expand: int = 2

    # ── Depth-wise convolution ────────────────────────────────────
    dwconv_kernel: int = 5

    # ── Timestep conditioning ─────────────────────────────────────
    time_embed_dim: int = 512

    # ── Condition encoder ─────────────────────────────────────────
    text_encoder_name: str = "google/flan-t5-base"
    image_encoder_name: str = "facebook/convnext-base-224"
    freeze_text_encoder: bool = True
    freeze_image_encoder: bool = True
    cond_text_dim: int = 768      # stub text encoder output dim
    cond_image_dim: int = 768     # stub image encoder output dim
    cond_seq_len: int = 16        # stub condition sequence length

    @classmethod
    def small(cls) -> ModelConfig:
        """Tiny config for unit tests and single-sample overfit smoke tests."""
        return cls(
            d_model=64,
            num_blocks=2,
            head_dim=32,
            d_ff=256,
            cross_attn_heads=2,
            mamba_expand=2,
            time_embed_dim=64,
            embed_dropout=0.0,
            cond_text_dim=64,
            cond_image_dim=64,
            cond_seq_len=4,
        )
