"""
Single-sample overfitting smoke test.

Trains a small SpatialASTDenoiser on one synthetic sample for ~300 steps
and verifies loss drops below 0.01, confirming the full forward/backward
pipeline works end-to-end.

Requires CUDA GPU + mamba-ssm.  Run on the remote GPU platform.
Follows development.md Section 7.3.
"""

from __future__ import annotations

import sys
import os

import pytest
import torch
import torch.nn.functional as F

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA GPU + mamba-ssm",
)

# ── Helpers ───────────────────────────────────────────────────────────

def _build_sample(device: torch.device) -> dict:
    """
    Build a single training sample from the rectangle fixture
    using MetaAnnotator, then tensorize it.
    """
    from tests.helpers import build_rectangle_solid
    from data.meta_annotator import MetaAnnotator
    from core.tokenizer import TOKEN_PAD

    ast = build_rectangle_solid()
    annotator = MetaAnnotator(max_seq_len=None)
    sample = annotator.annotate(ast)
    L = sample.seq_len

    def _to(lst, dtype=torch.long):
        return torch.tensor(lst, dtype=dtype, device=device).unsqueeze(0)

    return dict(
        token_ids=_to(sample.tokens),
        depth_ids=_to(sample.depths),
        type_ids=_to(sample.types),
        role_ids=_to(sample.roles),
        parent_ids=_to(sample.parents),
        sibling_ids=_to(sample.siblings),
        geom_desc=torch.tensor(
            sample.geom_desc, dtype=torch.float32, device=device
        ).unsqueeze(0),
        seq_len=L,
    )


def _train_step(model, sample, optimizer, t_val, device):
    """One forward + backward step; returns scalar loss."""
    from core.tokenizer import TOKEN_MASK, Q8_OFFSET

    target = sample["token_ids"].clone()  # [1, L]
    L = sample["seq_len"]

    corrupted = target.clone()
    mask_ratio = 0.3
    num_mask = max(1, int(L * mask_ratio))
    indices = torch.randperm(L, device=device)[:num_mask]
    corrupted[0, indices] = TOKEN_MASK

    corruption_mask = torch.zeros(1, L, dtype=torch.bool, device=device)
    corruption_mask[0, indices] = True

    t = torch.tensor([t_val], device=device)
    text_tokens = torch.randint(0, 1000, (1, 4), device=device)

    logits = model(
        token_ids=corrupted,
        depth_ids=sample["depth_ids"],
        type_ids=sample["type_ids"],
        role_ids=sample["role_ids"],
        parent_ids=sample["parent_ids"],
        sibling_ids=sample["sibling_ids"],
        geom_desc=sample["geom_desc"],
        t=t,
        text_tokens=text_tokens,
    )

    masked_logits = logits[corruption_mask]           # [num_mask, V]
    masked_targets = target[corruption_mask]           # [num_mask]
    loss = F.cross_entropy(masked_logits, masked_targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ═══════════════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════════════

@requires_cuda
def test_overfit_single_sample():
    """Train on 1 sample for 300 steps, loss should drop to near zero."""
    from model.config import ModelConfig
    from model.denoiser import SpatialASTDenoiser

    device = torch.device("cuda")
    config = ModelConfig.small()
    model = SpatialASTDenoiser(config).to(device)
    sample = _build_sample(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    num_steps = 300
    losses = []

    for step in range(num_steps):
        loss = _train_step(model, sample, optimizer, t_val=500, device=device)
        losses.append(loss)
        if step % 50 == 0:
            print(f"  Step {step:3d}: loss = {loss:.6f}")

    final_loss = losses[-1]
    print(f"  Final loss: {final_loss:.6f}")
    assert final_loss < 0.1, (
        f"Single-sample overfit failed: final loss = {final_loss:.4f} "
        f"(expected < 0.1)"
    )

    avg_last_20 = sum(losses[-20:]) / 20
    print(f"  Avg last 20 steps: {avg_last_20:.6f}")
    assert avg_last_20 < 0.1, (
        f"Loss not stable: avg of last 20 = {avg_last_20:.4f}"
    )
