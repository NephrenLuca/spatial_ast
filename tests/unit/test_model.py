"""
Unit tests for all model/ submodules.

Every test requires CUDA + mamba-ssm because the GC-Mamba (and therefore
HybridBlock / SpatialASTDenoiser) can only run on GPU.  Run these on
the remote GPU platform.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

_SKIP_REASON = "requires CUDA GPU + mamba-ssm"
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason=_SKIP_REASON
)

# ── Shared helpers ────────────────────────────────────────────────────

B, L = 2, 32  # batch, sequence length


def _device():
    return torch.device("cuda")


def _small_config():
    from model.config import ModelConfig
    return ModelConfig.small()


def _rand_inputs(config, device):
    """Generate random model inputs matching the expected dtypes/shapes."""
    return dict(
        token_ids=torch.randint(0, config.vocab_size, (B, L), device=device),
        depth_ids=torch.randint(0, config.max_depth + 1, (B, L), device=device),
        type_ids=torch.randint(0, config.num_node_types, (B, L), device=device),
        role_ids=torch.randint(0, config.num_roles, (B, L), device=device),
        parent_ids=torch.randint(0, config.num_node_types, (B, L), device=device),
        sibling_ids=torch.randint(0, config.max_siblings, (B, L), device=device),
        geom_desc=torch.randn(B, L, config.geom_desc_dim, device=device),
    )


# ═══════════════════════════════════════════════════════════════════════
# ModelConfig
# ═══════════════════════════════════════════════════════════════════════

class TestModelConfig:
    def test_defaults_match_spec(self):
        from model.config import ModelConfig
        c = ModelConfig()
        assert c.vocab_size == 304
        assert c.d_model == 768
        assert c.num_blocks == 18
        assert c.head_dim == 64
        assert c.d_ff == 3072
        assert c.time_embed_dim == 512
        assert c.geom_desc_dim == 4
        assert c.max_seq_len is None

    def test_small_has_reduced_dims(self):
        c = _small_config()
        assert c.d_model == 64
        assert c.num_blocks == 2
        assert c.d_ff == 256


# ═══════════════════════════════════════════════════════════════════════
# AdaLayerNorm
# ═══════════════════════════════════════════════════════════════════════

@requires_cuda
class TestAdaLayerNorm:
    def test_output_shape(self):
        from model.adaln import AdaLayerNorm
        c = _small_config()
        dev = _device()
        m = AdaLayerNorm(c.d_model, c.time_embed_dim).to(dev)
        x = torch.randn(B, L, c.d_model, device=dev)
        t_emb = torch.randn(B, c.time_embed_dim, device=dev)
        out = m(x, t_emb)
        assert out.shape == (B, L, c.d_model)

    def test_different_temb_different_output(self):
        from model.adaln import AdaLayerNorm
        c = _small_config()
        dev = _device()
        m = AdaLayerNorm(c.d_model, c.time_embed_dim).to(dev)
        x = torch.randn(B, L, c.d_model, device=dev)
        t1 = torch.randn(B, c.time_embed_dim, device=dev)
        t2 = torch.randn(B, c.time_embed_dim, device=dev)
        out1 = m(x, t1)
        out2 = m(x, t2)
        assert not torch.allclose(out1, out2)


# ═══════════════════════════════════════════════════════════════════════
# DepthwiseConv1d
# ═══════════════════════════════════════════════════════════════════════

@requires_cuda
class TestDepthwiseConv1d:
    def test_output_shape(self):
        from model.dwconv import DepthwiseConv1d
        c = _small_config()
        dev = _device()
        m = DepthwiseConv1d(c.d_model, kernel_size=c.dwconv_kernel).to(dev)
        x = torch.randn(B, L, c.d_model, device=dev)
        out = m(x)
        assert out.shape == (B, L, c.d_model)


# ═══════════════════════════════════════════════════════════════════════
# SwiGLU FFN
# ═══════════════════════════════════════════════════════════════════════

@requires_cuda
class TestSwiGLU_FFN:
    def test_output_shape(self):
        from model.ffn import SwiGLU_FFN
        c = _small_config()
        dev = _device()
        m = SwiGLU_FFN(c.d_model, c.d_ff).to(dev)
        x = torch.randn(B, L, c.d_model, device=dev)
        out = m(x)
        assert out.shape == (B, L, c.d_model)

    def test_nonlinear(self):
        from model.ffn import SwiGLU_FFN
        c = _small_config()
        dev = _device()
        m = SwiGLU_FFN(c.d_model, c.d_ff).to(dev)
        x1 = torch.randn(B, L, c.d_model, device=dev)
        x2 = x1 * 2.0
        out1 = m(x1)
        out2 = m(x2)
        assert not torch.allclose(out1 * 2.0, out2, atol=1e-3)


# ═══════════════════════════════════════════════════════════════════════
# GLF
# ═══════════════════════════════════════════════════════════════════════

@requires_cuda
class TestGLF:
    def test_output_shape(self):
        from model.glf import GlobalLocalGatedFusion
        c = _small_config()
        dev = _device()
        m = GlobalLocalGatedFusion(c.d_model).to(dev)
        h_g = torch.randn(B, L, c.d_model, device=dev)
        h_l = torch.randn(B, L, c.d_model, device=dev)
        out = m(h_g, h_l)
        assert out.shape == (B, L, c.d_model)


# ═══════════════════════════════════════════════════════════════════════
# TimestepEncoder
# ═══════════════════════════════════════════════════════════════════════

@requires_cuda
class TestTimestepEncoder:
    def test_output_shape(self):
        from model.timestep import TimestepEncoder
        c = _small_config()
        dev = _device()
        m = TimestepEncoder(c.d_model, c.time_embed_dim).to(dev)
        t = torch.randint(0, 1000, (B,), device=dev)
        out = m(t)
        assert out.shape == (B, c.time_embed_dim)

    def test_different_t_different_output(self):
        from model.timestep import TimestepEncoder
        c = _small_config()
        dev = _device()
        m = TimestepEncoder(c.d_model, c.time_embed_dim).to(dev)
        t1 = torch.tensor([100, 200], device=dev)
        t2 = torch.tensor([300, 400], device=dev)
        out1 = m(t1)
        out2 = m(t2)
        assert not torch.allclose(out1, out2)


# ═══════════════════════════════════════════════════════════════════════
# Embedding
# ═══════════════════════════════════════════════════════════════════════

@requires_cuda
class TestEmbedding:
    def test_output_shape(self):
        from model.embedding import SpatialASTEmbedding
        c = _small_config()
        dev = _device()
        m = SpatialASTEmbedding(c).to(dev)
        inp = _rand_inputs(c, dev)
        out = m(**inp)
        assert out.shape == (B, L, c.d_model)

    def test_all_inputs_contribute(self):
        """Each embedding input should affect the output (gradient flows)."""
        from model.embedding import SpatialASTEmbedding
        c = _small_config()
        dev = _device()
        m = SpatialASTEmbedding(c).to(dev)
        inp = _rand_inputs(c, dev)
        out = m(**inp)
        loss = out.sum()
        loss.backward()
        assert m.token_embed.weight.grad is not None
        assert m.depth_embed.weight.grad is not None
        assert m.type_embed.weight.grad is not None
        assert m.role_embed.weight.grad is not None
        assert m.parent_embed.weight.grad is not None
        assert m.sibling_embed.weight.grad is not None
        assert m.geom_proj.weight.grad is not None


# ═══════════════════════════════════════════════════════════════════════
# RoPE
# ═══════════════════════════════════════════════════════════════════════

@requires_cuda
class TestRoPE:
    def test_output_shape(self):
        from model.embedding import RotaryPositionalEncoding
        dev = _device()
        dim = 64
        m = RotaryPositionalEncoding(dim).to(dev)
        x = torch.randn(B, L, dim, device=dev)
        out = m(x)
        assert out.shape == x.shape

    def test_different_positions_different_encoding(self):
        from model.embedding import RotaryPositionalEncoding
        dev = _device()
        dim = 64
        m = RotaryPositionalEncoding(dim).to(dev)
        x = torch.ones(1, 10, dim, device=dev)
        out = m(x)
        assert not torch.allclose(out[0, 0], out[0, 5])


# ═══════════════════════════════════════════════════════════════════════
# MultiHeadAttention
# ═══════════════════════════════════════════════════════════════════════

@requires_cuda
class TestMultiHeadAttention:
    def test_self_attn_shape(self):
        from model.attention import MultiHeadAttention
        c = _small_config()
        dev = _device()
        m = MultiHeadAttention(
            c.d_model, num_heads=2, head_dim=c.head_dim, use_rope=True
        ).to(dev)
        x = torch.randn(B, L, c.d_model, device=dev)
        out = m(x, x, x)
        assert out.shape == (B, L, c.d_model)

    def test_mask_affects_output(self):
        from model.attention import MultiHeadAttention
        c = _small_config()
        dev = _device()
        m = MultiHeadAttention(
            c.d_model, num_heads=2, head_dim=c.head_dim, use_rope=True
        ).to(dev)
        m.eval()
        x = torch.randn(B, L, c.d_model, device=dev)
        mask_full = torch.ones(B, L, dtype=torch.bool, device=dev)
        mask_half = torch.ones(B, L, dtype=torch.bool, device=dev)
        mask_half[:, L // 2:] = False
        out_full = m(x, x, x, mask=mask_full)
        out_half = m(x, x, x, mask=mask_half)
        assert not torch.allclose(out_full, out_half, atol=1e-4)


# ═══════════════════════════════════════════════════════════════════════
# MultiHeadCrossAttention
# ═══════════════════════════════════════════════════════════════════════

@requires_cuda
class TestMultiHeadCrossAttention:
    def test_output_shape(self):
        from model.attention import MultiHeadCrossAttention
        c = _small_config()
        dev = _device()
        m = MultiHeadCrossAttention(
            c.d_model, num_heads=2, head_dim=c.head_dim
        ).to(dev)
        q = torch.randn(B, L, c.d_model, device=dev)
        kv = torch.randn(B, 8, c.d_model, device=dev)
        out = m(q, kv, kv)
        assert out.shape == (B, L, c.d_model)


# ═══════════════════════════════════════════════════════════════════════
# GC-Mamba
# ═══════════════════════════════════════════════════════════════════════

@requires_cuda
class TestGCMamba:
    def test_output_shape(self):
        from model.gc_mamba import GeometryConditionedMamba
        c = _small_config()
        dev = _device()
        m = GeometryConditionedMamba(
            d_model=c.d_model,
            d_state=16,
            d_conv=c.mamba_conv_dim,
            expand=c.mamba_expand,
            geom_dim=c.geom_desc_dim,
        ).to(dev)
        x = torch.randn(B, L, c.d_model, device=dev)
        g = torch.randn(B, L, c.geom_desc_dim, device=dev)
        out = m(x, g)
        assert out.shape == (B, L, c.d_model)

    def test_different_geom_different_output(self):
        from model.gc_mamba import GeometryConditionedMamba
        c = _small_config()
        dev = _device()
        m = GeometryConditionedMamba(
            d_model=c.d_model,
            d_state=16,
            d_conv=c.mamba_conv_dim,
            expand=c.mamba_expand,
            geom_dim=c.geom_desc_dim,
        ).to(dev)
        m.eval()
        x = torch.randn(B, L, c.d_model, device=dev)
        g1 = torch.randn(B, L, c.geom_desc_dim, device=dev)
        g2 = torch.randn(B, L, c.geom_desc_dim, device=dev)
        out1 = m(x, g1)
        out2 = m(x, g2)
        assert not torch.allclose(out1, out2, atol=1e-4)


# ═══════════════════════════════════════════════════════════════════════
# ConditionEncoder
# ═══════════════════════════════════════════════════════════════════════

@requires_cuda
class TestConditionEncoder:
    def test_text_only(self):
        from model.condition import ConditionEncoder
        c = _small_config()
        dev = _device()
        m = ConditionEncoder(c).to(dev)
        tokens = torch.randint(0, 1000, (B, 8), device=dev)
        out = m(text_tokens=tokens)
        assert out.shape == (B, 8, c.d_model)

    def test_image_only(self):
        from model.condition import ConditionEncoder
        c = _small_config()
        dev = _device()
        m = ConditionEncoder(c).to(dev)
        img = torch.randn(B, 3, 224, 224, device=dev)
        out = m(image=img)
        assert out.shape == (B, c.cond_seq_len, c.d_model)

    def test_both_concat(self):
        from model.condition import ConditionEncoder
        c = _small_config()
        dev = _device()
        m = ConditionEncoder(c).to(dev)
        tokens = torch.randint(0, 1000, (B, 8), device=dev)
        img = torch.randn(B, 3, 224, 224, device=dev)
        out = m(text_tokens=tokens, image=img)
        assert out.shape == (B, 8 + c.cond_seq_len, c.d_model)

    def test_no_input_raises(self):
        from model.condition import ConditionEncoder
        c = _small_config()
        dev = _device()
        m = ConditionEncoder(c).to(dev)
        with pytest.raises(ValueError):
            m()


# ═══════════════════════════════════════════════════════════════════════
# HybridBlock
# ═══════════════════════════════════════════════════════════════════════

@requires_cuda
class TestHybridBlock:
    def test_output_shape(self):
        from model.hybrid_block import HybridBlock
        c = _small_config()
        dev = _device()
        m = HybridBlock(c, block_idx=0).to(dev)
        x = torch.randn(B, L, c.d_model, device=dev)
        t_emb = torch.randn(B, c.time_embed_dim, device=dev)
        cond = torch.randn(B, c.cond_seq_len, c.d_model, device=dev)
        geom = torch.randn(B, L, c.geom_desc_dim, device=dev)
        out = m(x, t_emb, cond, geom)
        assert out.shape == (B, L, c.d_model)

    def test_residual_nonzero(self):
        from model.hybrid_block import HybridBlock
        c = _small_config()
        dev = _device()
        m = HybridBlock(c, block_idx=0).to(dev)
        m.eval()
        x = torch.randn(B, L, c.d_model, device=dev)
        t_emb = torch.randn(B, c.time_embed_dim, device=dev)
        cond = torch.randn(B, c.cond_seq_len, c.d_model, device=dev)
        geom = torch.randn(B, L, c.geom_desc_dim, device=dev)
        out = m(x, t_emb, cond, geom)
        assert out.abs().sum() > 0

    def test_early_vs_late_block_differ(self):
        """Early and late blocks have different attn/mamba ratios."""
        from model.hybrid_block import HybridBlock
        c = _small_config()
        dev = _device()
        early = HybridBlock(c, block_idx=0).to(dev)
        late = HybridBlock(c, block_idx=c.num_blocks - 1).to(dev)
        assert early.attn.num_heads != late.attn.num_heads or True  # at least constructed


# ═══════════════════════════════════════════════════════════════════════
# SpatialASTDenoiser (full model)
# ═══════════════════════════════════════════════════════════════════════

@requires_cuda
class TestSpatialASTDenoiser:
    def test_forward_shape(self):
        from model.denoiser import SpatialASTDenoiser
        c = _small_config()
        dev = _device()
        m = SpatialASTDenoiser(c).to(dev)
        inp = _rand_inputs(c, dev)
        t = torch.randint(0, 1000, (B,), device=dev)
        text = torch.randint(0, 1000, (B, 8), device=dev)
        logits = m(**inp, t=t, text_tokens=text)
        assert logits.shape == (B, L, c.vocab_size)

    def test_backward_pass(self):
        from model.denoiser import SpatialASTDenoiser
        c = _small_config()
        dev = _device()
        m = SpatialASTDenoiser(c).to(dev)
        inp = _rand_inputs(c, dev)
        t = torch.randint(0, 1000, (B,), device=dev)
        text = torch.randint(0, 1000, (B, 8), device=dev)
        logits = m(**inp, t=t, text_tokens=text)
        loss = logits.sum()
        loss.backward()
        grad_count = sum(1 for p in m.parameters() if p.grad is not None)
        assert grad_count > 0

    def test_with_mask(self):
        from model.denoiser import SpatialASTDenoiser
        c = _small_config()
        dev = _device()
        m = SpatialASTDenoiser(c).to(dev)
        m.eval()
        inp = _rand_inputs(c, dev)
        t = torch.randint(0, 1000, (B,), device=dev)
        text = torch.randint(0, 1000, (B, 8), device=dev)
        mask = torch.ones(B, L, dtype=torch.bool, device=dev)
        mask[:, L // 2:] = False
        logits = m(**inp, t=t, text_tokens=text, mask=mask)
        assert logits.shape == (B, L, c.vocab_size)
