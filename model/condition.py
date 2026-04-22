"""
Condition encoder: text and/or image -> condition KV sequence.

**Current implementation**: lightweight stub encoders that produce the
correct output shapes without downloading pretrained models.  This lets
the architecture be validated locally (CPU) and in unit tests.

**Production implementation** (for remote GPU training):
Replace ``StubTextEncoder`` / ``StubImageEncoder`` with the real wrappers
below (uncomment and install ``transformers``):

    from transformers import T5EncoderModel, T5Tokenizer
    from transformers import ConvNextModel

    class T5EncoderWrapper(nn.Module):
        def __init__(self, model_name, freeze, proj_dim):
            super().__init__()
            self.encoder = T5EncoderModel.from_pretrained(model_name)
            if freeze:
                for p in self.encoder.parameters():
                    p.requires_grad = False
            hidden = self.encoder.config.d_model
            self.proj = nn.Linear(hidden, proj_dim)

        def forward(self, input_ids, attention_mask=None):
            out = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask).last_hidden_state
            return self.proj(out)

    class ViTEncoderWrapper(nn.Module):
        def __init__(self, model_name, freeze, proj_dim):
            super().__init__()
            self.encoder = ConvNextModel.from_pretrained(model_name)
            if freeze:
                for p in self.encoder.parameters():
                    p.requires_grad = False
            hidden = self.encoder.config.hidden_sizes[-1]
            self.proj = nn.Linear(hidden, proj_dim)
            self.pool = nn.AdaptiveAvgPool1d(1)

        def forward(self, pixel_values):
            features = self.encoder(pixel_values).last_hidden_state
            B, C, H, W = features.shape
            features = features.view(B, C, H * W).permute(0, 2, 1)
            return self.proj(features)

Follows architecture.md Section 6.7.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from model.config import ModelConfig


class StubTextEncoder(nn.Module):
    """Placeholder text encoder that projects random token IDs to d_model."""

    def __init__(self, vocab_size: int, d_model: int, seq_len: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.seq_len = seq_len

    def forward(self, input_ids: Tensor) -> Tensor:
        """input_ids: [B, L_text] -> [B, L_text, d_model]"""
        return self.proj(self.embed(input_ids))


class StubImageEncoder(nn.Module):
    """Placeholder image encoder that projects a flat image tensor to d_model."""

    def __init__(self, d_model: int, seq_len: int) -> None:
        super().__init__()
        self.proj = nn.Linear(3 * 224 * 224, seq_len * d_model)
        self.seq_len = seq_len
        self.d_model = d_model

    def forward(self, image: Tensor) -> Tensor:
        """image: [B, 3, 224, 224] -> [B, seq_len, d_model]"""
        B = image.size(0)
        flat = image.reshape(B, -1)
        return self.proj(flat).view(B, self.seq_len, self.d_model)


class ConditionEncoder(nn.Module):
    """
    Multi-modal condition encoder: text + image -> condition KV.

    Uses stub encoders locally.  For production, swap in T5/ConvNeXt
    wrappers documented at the top of this file.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model

        self.text_encoder = StubTextEncoder(
            vocab_size=32128,  # T5 vocab size
            d_model=d,
            seq_len=config.cond_seq_len,
        )
        self.image_encoder = StubImageEncoder(
            d_model=d,
            seq_len=config.cond_seq_len,
        )

        self.modality_embed = nn.Embedding(2, d)  # 0 = text, 1 = image
        self.fuse = nn.Linear(d, d)

    def forward(
        self,
        text_tokens: Tensor | None = None,
        image: Tensor | None = None,
    ) -> Tensor:
        """
        Returns condition KV of shape [B, L_cond, d_model].
        At least one of text_tokens / image must be provided.
        """
        features: list[Tensor] = []

        if text_tokens is not None:
            t_feat = self.text_encoder(text_tokens)
            mod_ids = torch.zeros(
                t_feat.shape[:2], dtype=torch.long, device=t_feat.device
            )
            t_feat = t_feat + self.modality_embed(mod_ids)
            features.append(t_feat)

        if image is not None:
            i_feat = self.image_encoder(image)
            mod_ids = torch.ones(
                i_feat.shape[:2], dtype=torch.long, device=i_feat.device
            )
            i_feat = i_feat + self.modality_embed(mod_ids)
            features.append(i_feat)

        if not features:
            raise ValueError("ConditionEncoder requires at least one of text_tokens or image")

        cond = torch.cat(features, dim=1) if len(features) > 1 else features[0]
        return self.fuse(cond)
