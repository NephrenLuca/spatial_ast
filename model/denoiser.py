"""
SpatialASTDenoiser: the complete hybrid Mamba-Transformer denoiser network.

Assembles embedding, timestep encoder, condition encoder, N hybrid blocks,
final layer-norm, and vocabulary projection head.
Follows architecture.md Section 6.9.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from model.config import ModelConfig
from model.embedding import SpatialASTEmbedding
from model.timestep import TimestepEncoder
from model.condition import ConditionEncoder
from model.hybrid_block import HybridBlock


class SpatialASTDenoiser(nn.Module):
    """Full denoiser network: input tokens -> vocab logits."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = SpatialASTEmbedding(config)
        self.time_encoder = TimestepEncoder(config.d_model, config.time_embed_dim)
        self.condition_encoder = ConditionEncoder(config)

        self.blocks = nn.ModuleList([
            HybridBlock(config, i) for i in range(config.num_blocks)
        ])

        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        token_ids: Tensor,
        depth_ids: Tensor,
        type_ids: Tensor,
        role_ids: Tensor,
        parent_ids: Tensor,
        sibling_ids: Tensor,
        geom_desc: Tensor,
        t: Tensor,
        text_tokens: Tensor | None = None,
        image: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            token_ids:    [B, L]  token vocabulary IDs
            depth_ids:    [B, L]  AST depth per token
            type_ids:     [B, L]  node type per token
            role_ids:     [B, L]  token role (NODE_TAG / PARAM_VALUE / ...)
            parent_ids:   [B, L]  parent node type per token
            sibling_ids:  [B, L]  sibling index per token
            geom_desc:    [B, L, 4]  geometry descriptors
            t:            [B]    diffusion timestep
            text_tokens:  [B, L_text] (optional) text condition token IDs
            image:        [B, 3, 224, 224] (optional) image condition
            mask:         [B, L]  bool — True for valid positions

        Returns:
            logits: [B, L, vocab_size]
        """
        x = self.embedding(
            token_ids, depth_ids, type_ids, role_ids,
            parent_ids, sibling_ids, geom_desc,
        )

        t_emb = self.time_encoder(t)
        cond_kv = self.condition_encoder(text_tokens, image)

        for block in self.blocks:
            x = block(x, t_emb, cond_kv, geom_desc, mask)

        x = self.final_norm(x)
        return self.output_proj(x)
