"""
Unit tests for the diffusion/ package: schedule, corruption, loss.
"""

from __future__ import annotations

import random
import pytest
import torch

from core.ast_node import reset_id_counter
from core.serializer import ASTSerializer
from core.tokenizer import (
    TOKEN_BOS,
    TOKEN_EOS,
    TOKEN_MASK,
    TOKEN_NOISE,
    TOKEN_PAD,
    Q8_OFFSET,
    TokenRole,
)
from diffusion.schedule import DiffusionConfig, corruption_probability
from diffusion.corruption import (
    CorruptionMode,
    SubtreeCorruptor,
    hierarchical_corrupt,
    batch_corrupt,
)
from diffusion.loss import LossConfig, compute_loss
from tests.helpers import build_rectangle_solid, build_triangle_solid


# ═══════════════════════════════════════════════════════════════════════
# Schedule tests
# ═══════════════════════════════════════════════════════════════════════

class TestCorruptionProbability:
    def setup_method(self):
        self.cfg = DiffusionConfig(T=1000, max_depth=5, tau_scale=0.9, beta_scale=0.1)

    def test_zero_timestep_shallow_low_probability(self):
        for d in range(3):
            p = corruption_probability(d, 0, 1000, self.cfg)
            assert p < 0.01, f"depth={d}: p={p} should be near 0 at t=0"

    def test_zero_timestep_deep_moderate_probability(self):
        p5 = corruption_probability(5, 0, 1000, self.cfg)
        assert 0.4 < p5 < 0.6, (
            f"depth=5 at t=0: p={p5} should be ~0.5 (tau_5=0, sigmoid(0)=0.5)"
        )

    def test_max_timestep_shallow_near_one(self):
        p = corruption_probability(0, 1000, 1000, self.cfg)
        assert p > 0.7, f"p={p} for depth=0, t=T should be high"

    def test_deeper_nodes_corrupted_earlier(self):
        t = 200
        p_deep = corruption_probability(5, t, 1000, self.cfg)
        p_shallow = corruption_probability(0, t, 1000, self.cfg)
        assert p_deep > p_shallow, (
            f"At t={t}, deeper nodes (p={p_deep}) should be more likely "
            f"to be corrupted than shallow (p={p_shallow})"
        )

    def test_monotonic_in_time(self):
        for d in range(6):
            prev_p = 0.0
            for t in range(0, 1001, 100):
                p = corruption_probability(d, t, 1000, self.cfg)
                assert p >= prev_p - 1e-9, (
                    f"depth={d}: p({t})={p} < p({t-100})={prev_p}"
                )
                prev_p = p

    def test_monotonic_in_depth_at_fixed_t(self):
        for t in [100, 300, 500, 700]:
            probs = [corruption_probability(d, t, 1000, self.cfg) for d in range(6)]
            for i in range(len(probs) - 1):
                assert probs[i] <= probs[i + 1] + 1e-9, (
                    f"t={t}: p(d={i})={probs[i]} > p(d={i+1})={probs[i+1]}"
                )


# ═══════════════════════════════════════════════════════════════════════
# Corruption tests
# ═══════════════════════════════════════════════════════════════════════

def _serialize_fixture():
    """Serialize the rectangle solid and return tokens, depth_map, role_map."""
    reset_id_counter()
    ast = build_rectangle_solid()
    serializer = ASTSerializer()
    tokens, metas = serializer.serialize(ast, pad=False)
    depth_map = [m.depth for m in metas]
    role_map = [int(m.role) for m in metas]
    return tokens, depth_map, role_map


class TestHierarchicalCorrupt:
    def setup_method(self):
        self.cfg = DiffusionConfig(T=1000, max_depth=5, tau_scale=0.9, beta_scale=0.1)

    def test_no_corruption_at_t0(self):
        tokens, depths, roles = _serialize_fixture()
        corrupted, mask = hierarchical_corrupt(tokens, depths, roles, t=0, config=self.cfg)
        assert not any(mask), "Nothing should be corrupted at t=0"
        assert corrupted == tokens

    def test_some_corruption_at_high_t(self):
        tokens, depths, roles = _serialize_fixture()
        found_corruption = False
        for seed in range(10):
            corrupted, mask = hierarchical_corrupt(
                tokens, depths, roles, t=950, config=self.cfg,
                rng=random.Random(seed),
            )
            if any(mask):
                found_corruption = True
                break
        assert found_corruption, "At t=950 with 10 seeds, some tokens should be corrupted"

    def test_original_unchanged(self):
        tokens, depths, roles = _serialize_fixture()
        original = list(tokens)
        hierarchical_corrupt(tokens, depths, roles, t=500, config=self.cfg)
        assert tokens == original, "Original token list must not be modified"

    def test_mask_length_matches_output(self):
        tokens, depths, roles = _serialize_fixture()
        for t in [100, 500, 900]:
            corrupted, mask = hierarchical_corrupt(
                tokens, depths, roles, t=t, config=self.cfg,
                rng=random.Random(123),
            )
            assert len(mask) == len(corrupted)


class TestSubtreeCorruptor:
    def test_mask_mode_replaces_tag(self):
        tokens, depths, roles = _serialize_fixture()
        corruptor = SubtreeCorruptor()
        first_tag_idx = next(
            i for i, r in enumerate(roles) if r == TokenRole.NODE_TAG
        )
        result = corruptor.corrupt_span(
            list(tokens), roles, first_tag_idx, first_tag_idx + 1,
            CorruptionMode.MASK,
        )
        assert result[first_tag_idx] == TOKEN_MASK

    def test_resample_changes_q8(self):
        tokens, depths, roles = _serialize_fixture()
        q8_indices = [i for i, t in enumerate(tokens) if Q8_OFFSET <= t < Q8_OFFSET + 256]
        if not q8_indices:
            pytest.skip("No Q8 tokens in fixture")
        corruptor = SubtreeCorruptor()
        start = q8_indices[0]
        end = q8_indices[-1] + 1
        original_vals = [tokens[i] for i in q8_indices]
        result = corruptor.corrupt_span(
            list(tokens), roles, start, end, CorruptionMode.RESAMPLE,
        )
        resampled_vals = [result[i] for i in q8_indices if i < len(result)]
        assert any(a != b for a, b in zip(original_vals, resampled_vals)), (
            "RESAMPLE should change at least some Q8 tokens"
        )


class TestBatchCorrupt:
    def test_batch_output_shapes(self):
        tokens, depths, roles = _serialize_fixture()
        L = len(tokens)
        B = 4
        token_batch = torch.tensor([tokens] * B, dtype=torch.long)
        depth_batch = torch.tensor([depths] * B, dtype=torch.long)
        role_batch = torch.tensor([roles] * B, dtype=torch.long)
        t_batch = torch.tensor([500] * B, dtype=torch.long)

        cfg = DiffusionConfig(T=1000)
        out_tokens, out_mask = batch_corrupt(
            token_batch, depth_batch, role_batch, t_batch, cfg,
        )
        assert out_tokens.shape[0] == B
        assert out_mask.shape[0] == B
        assert out_tokens.shape[1] == out_mask.shape[1]
        assert out_tokens.shape[1] >= L


# ═══════════════════════════════════════════════════════════════════════
# Loss tests
# ═══════════════════════════════════════════════════════════════════════

class TestComputeLoss:
    def test_basic_loss_shape(self):
        B, L, V = 2, 16, 304
        logits = torch.randn(B, L, V)
        targets = torch.randint(0, V, (B, L))
        mask = torch.ones(B, L, dtype=torch.bool)
        depths = torch.randint(0, 6, (B, L))
        roles = torch.full((B, L), TokenRole.PARAM_VALUE, dtype=torch.long)

        losses = compute_loss(logits, targets, mask, depths, roles, LossConfig())
        assert "total" in losses
        assert losses["total"].dim() == 0
        assert losses["total"].item() > 0

    def test_no_corruption_gives_zero(self):
        B, L, V = 2, 16, 304
        logits = torch.randn(B, L, V)
        targets = torch.randint(0, V, (B, L))
        mask = torch.zeros(B, L, dtype=torch.bool)
        depths = torch.randint(0, 6, (B, L))
        roles = torch.zeros(B, L, dtype=torch.long)

        losses = compute_loss(logits, targets, mask, depths, roles, LossConfig())
        assert losses["total"].item() == 0.0

    def test_node_ce_fires_on_tag_tokens(self):
        B, L, V = 1, 8, 304
        logits = torch.randn(B, L, V)
        targets = torch.randint(8, 22, (B, L))
        mask = torch.ones(B, L, dtype=torch.bool)
        depths = torch.ones(B, L, dtype=torch.long)
        roles = torch.full((B, L), TokenRole.NODE_TAG, dtype=torch.long)

        losses = compute_loss(logits, targets, mask, depths, roles, LossConfig())
        assert losses["node_ce"].item() > 0

    def test_param_ce_fires_on_param_tokens(self):
        B, L, V = 1, 8, 304
        logits = torch.randn(B, L, V)
        targets = torch.randint(Q8_OFFSET, Q8_OFFSET + 256, (B, L))
        mask = torch.ones(B, L, dtype=torch.bool)
        depths = torch.ones(B, L, dtype=torch.long)
        roles = torch.full((B, L), TokenRole.PARAM_VALUE, dtype=torch.long)

        losses = compute_loss(logits, targets, mask, depths, roles, LossConfig())
        assert losses["param_ce"].item() > 0
        assert losses["reg"].item() >= 0

    def test_depth_weighting_increases_loss(self):
        B, L, V = 1, 8, 304
        logits = torch.randn(B, L, V)
        targets = torch.randint(0, V, (B, L))
        mask = torch.ones(B, L, dtype=torch.bool)
        roles = torch.full((B, L), TokenRole.PARAM_VALUE, dtype=torch.long)

        depths_shallow = torch.zeros(B, L, dtype=torch.long)
        depths_deep = torch.full((B, L), 5, dtype=torch.long)

        loss_shallow = compute_loss(
            logits, targets, mask, depths_shallow, roles, LossConfig()
        )
        loss_deep = compute_loss(
            logits, targets, mask, depths_deep, roles, LossConfig()
        )
        assert loss_deep["ce"].item() >= loss_shallow["ce"].item()
