"""
Condition encoder: text and/or image -> condition KV sequence.

Supports two modes controlled by ``ModelConfig.use_real_encoder``:

* **Stub mode** (default, ``use_real_encoder=False``):
  Lightweight random projections that produce correct output shapes.
  Used for unit tests and architecture validation on CPU/GPU without
  downloading pretrained weights.

* **Real mode** (``use_real_encoder=True``):
  Loads frozen ``google/flan-t5-base`` (text) and
  ``facebook/convnext-base-224`` (image) from local paths specified in
  ``ModelConfig.text_encoder_name`` / ``ModelConfig.image_encoder_name``.
  Requires the ``transformers`` package.

Follows architecture.md Section 6.7.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from model.config import ModelConfig


# ═══════════════════════════════════════════════════════════════════════
# Real encoders (require ``transformers``)
# ═══════════════════════════════════════════════════════════════════════

class T5EncoderWrapper(nn.Module):
    """Frozen Flan-T5 encoder with a learned projection to ``proj_dim``."""

    def __init__(self, model_name: str, freeze: bool, proj_dim: int) -> None:
        super().__init__()
        from transformers import T5EncoderModel

        self.encoder = T5EncoderModel.from_pretrained(
            model_name, use_safetensors=True,
        )
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        hidden = self.encoder.config.d_model
        self.proj = nn.Linear(hidden, proj_dim)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """input_ids: [B, L_text] -> [B, L_text, proj_dim]"""
        out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        return self.proj(out)


class ConvNextEncoderWrapper(nn.Module):
    """Frozen ConvNeXt encoder with spatial tokens projected to ``proj_dim``."""

    def __init__(self, model_name: str, freeze: bool, proj_dim: int) -> None:
        super().__init__()
        from transformers import ConvNextModel

        self.encoder = ConvNextModel.from_pretrained(
            model_name, use_safetensors=True,
        )
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        hidden = self.encoder.config.hidden_sizes[-1]
        self.proj = nn.Linear(hidden, proj_dim)

    def forward(self, pixel_values: Tensor) -> Tensor:
        """pixel_values: [B, 3, 224, 224] -> [B, H*W, proj_dim]"""
        features = self.encoder(pixel_values).last_hidden_state  # [B, C, H, W]
        B, C, H, W = features.shape
        features = features.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        return self.proj(features)


# ═══════════════════════════════════════════════════════════════════════
# Stub encoders (no external dependencies)
# ═══════════════════════════════════════════════════════════════════════

class StubTextEncoder(nn.Module):
    """Placeholder text encoder that projects random token IDs to d_model."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """input_ids: [B, L_text] -> [B, L_text, d_model]"""
        return self.proj(self.embed(input_ids))


class StubImageEncoder(nn.Module):
    """Placeholder image encoder that projects a flat image tensor to d_model."""

    def __init__(self, d_model: int, seq_len: int) -> None:
        super().__init__()
        self.proj = nn.Linear(3 * 224 * 224, seq_len * d_model)
        self.seq_len = seq_len
        self.d_model = d_model

    def forward(self, pixel_values: Tensor) -> Tensor:
        """pixel_values: [B, 3, 224, 224] -> [B, seq_len, d_model]"""
        B = pixel_values.size(0)
        flat = pixel_values.reshape(B, -1)
        return self.proj(flat).view(B, self.seq_len, self.d_model)


# ═══════════════════════════════════════════════════════════════════════
# Unified ConditionEncoder
# ═══════════════════════════════════════════════════════════════════════

class ConditionEncoder(nn.Module):
    """
    Multi-modal condition encoder: text + image -> condition KV.

    When ``config.use_real_encoder`` is True, loads pretrained T5 and ConvNeXt
    from local paths in ``config.text_encoder_name`` / ``config.image_encoder_name``.
    Otherwise falls back to lightweight stubs.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model

        if config.use_real_encoder:
            self.text_encoder = T5EncoderWrapper(
                model_name=config.text_encoder_name,
                freeze=config.freeze_text_encoder,
                proj_dim=d,
            )
            self.image_encoder = ConvNextEncoderWrapper(
                model_name=config.image_encoder_name,
                freeze=config.freeze_image_encoder,
                proj_dim=d,
            )
        else:
            self.text_encoder = StubTextEncoder(vocab_size=32128, d_model=d)
            self.image_encoder = StubImageEncoder(d_model=d, seq_len=config.cond_seq_len)

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
