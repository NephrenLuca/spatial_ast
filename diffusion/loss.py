"""
Training loss for hierarchical corruption diffusion.

Combines four terms following architecture.md Section 5.5:
  1. Depth-weighted cross-entropy (main reconstruction loss)
  2. Q8 parameter L2 regression
  3. Node-type auxiliary CE  (structure supervision)
  4. Param-value auxiliary CE (precision supervision)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from core.tokenizer import Q8_OFFSET, TokenRole


@dataclass
class LossConfig:
    """Weights for the composite training loss."""
    reg_weight: float = 0.1
    depth_weight_alpha: float = 0.2
    node_ce_weight: float = 0.5
    param_ce_weight: float = 2.0


def compute_loss(
    model_output: Tensor,
    target_tokens: Tensor,
    corruption_mask: Tensor,
    depth_map: Tensor,
    role_map: Tensor,
    config: LossConfig,
) -> dict[str, Tensor]:
    """
    Compute the composite diffusion training loss.

    Parameters
    ----------
    model_output    : [B, L, V]  logits from the denoiser
    target_tokens   : [B, L]     ground-truth token IDs
    corruption_mask : [B, L]     bool — True for corrupted positions
    depth_map       : [B, L]     int — AST depth per token
    role_map        : [B, L]     int — TokenRole per token
    config          : LossConfig

    Returns
    -------
    dict with keys:
      ``"total"``, ``"ce"``, ``"reg"``, ``"node_ce"``, ``"param_ce"``
    """
    V = model_output.shape[-1]
    device = model_output.device
    zero = torch.tensor(0.0, device=device)

    if not corruption_mask.any():
        return {
            "total": zero, "ce": zero, "reg": zero,
            "node_ce": zero, "param_ce": zero,
        }

    # 1. Depth-weighted cross-entropy on corrupted positions
    masked_logits = model_output[corruption_mask].view(-1, V)
    masked_targets = target_tokens[corruption_mask].view(-1)
    ce_per_token = F.cross_entropy(masked_logits, masked_targets, reduction="none")

    depth_weights = 1.0 + config.depth_weight_alpha * depth_map[corruption_mask].float()
    weighted_ce = (ce_per_token * depth_weights).mean()

    # 2. Q8 parameter L2 regression
    is_q8 = (target_tokens >= Q8_OFFSET) & (target_tokens < Q8_OFFSET + 256)
    q8_mask = corruption_mask & is_q8
    if q8_mask.any():
        pred_vals = model_output[q8_mask].argmax(-1).float() - Q8_OFFSET
        true_vals = target_tokens[q8_mask].float() - Q8_OFFSET
        reg_loss = F.mse_loss(pred_vals, true_vals)
    else:
        reg_loss = zero

    # 3. Node-type auxiliary CE
    is_node_tag = (role_map == TokenRole.NODE_TAG)
    node_mask = corruption_mask & is_node_tag
    if node_mask.any():
        node_ce = F.cross_entropy(
            model_output[node_mask].view(-1, V),
            target_tokens[node_mask].view(-1),
        )
    else:
        node_ce = zero

    # 4. Param-value auxiliary CE
    is_param = (role_map == TokenRole.PARAM_VALUE)
    param_mask = corruption_mask & is_param
    if param_mask.any():
        param_ce = F.cross_entropy(
            model_output[param_mask].view(-1, V),
            target_tokens[param_mask].view(-1),
        )
    else:
        param_ce = zero

    total = (
        weighted_ce
        + config.reg_weight * reg_loss
        + config.node_ce_weight * node_ce
        + config.param_ce_weight * param_ce
    )

    return {
        "total": total,
        "ce": weighted_ce,
        "reg": reg_loss,
        "node_ce": node_ce,
        "param_ce": param_ce,
    }
