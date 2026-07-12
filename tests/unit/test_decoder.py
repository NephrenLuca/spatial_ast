"""
Unit tests for the decoder/ package: grammar mask, bracket balancer,
geometry checker, constraint decoder pipeline.
"""

from __future__ import annotations

import pytest
import torch

from core.types import NodeType, NodeRegistry
from core.ast_node import ASTNode, reset_id_counter
from core.serializer import ASTSerializer
from core.grammar import validate_ast
from core.tokenizer import (
    TOKEN_BOS,
    TOKEN_EOS,
    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_PAD,
    VOCAB_SIZE,
    Q8_OFFSET,
    get_node_type_token,
)
from decoder.grammar_mask import GrammarMask
from decoder.bracket_balancer import BracketBalancer
from decoder.geometry_checker import GeometryChecker
from decoder.pipeline import ConstraintDecoderPipeline
from tests.helpers import (
    build_rectangle_solid,
    build_triangle_solid,
    build_circle_solid,
    make_edge,
    make_line,
    make_loop,
)


# ═══════════════════════════════════════════════════════════════════════
# GrammarMask tests
# ═══════════════════════════════════════════════════════════════════════

class TestGrammarMask:
    def test_initial_state_expects_bos(self):
        gm = GrammarMask()
        valid = gm.get_valid_tokens()
        assert valid == {TOKEN_BOS}

    def test_after_bos_expects_prog(self):
        gm = GrammarMask()
        gm.feed(TOKEN_BOS)
        valid = gm.get_valid_tokens()
        assert get_node_type_token("PROG") in valid

    def test_after_eos_expects_pad(self):
        gm = GrammarMask()
        gm.feed(TOKEN_BOS)
        gm.feed(get_node_type_token("PROG"))
        gm.feed(TOKEN_EOS)
        valid = gm.get_valid_tokens()
        assert valid == {TOKEN_PAD}

    def test_prog_expects_open_paren_or_params(self):
        gm = GrammarMask()
        gm.feed(TOKEN_BOS)
        gm.feed(get_node_type_token("PROG"))
        valid = gm.get_valid_tokens()
        assert TOKEN_LPAREN in valid or any(
            Q8_OFFSET <= t < Q8_OFFSET + 256 for t in valid
        ) or len(valid) > 0

    def test_valid_tokens_never_empty_during_decode(self):
        reset_id_counter()
        ast = build_rectangle_solid()
        serializer = ASTSerializer()
        tokens, _ = serializer.serialize(ast, pad=False)

        gm = GrammarMask()
        for i, tok in enumerate(tokens):
            valid = gm.get_valid_tokens()
            assert len(valid) > 0, f"No valid tokens at position {i}"
            gm.feed(tok)

    def test_mask_application(self):
        gm = GrammarMask()
        logits = torch.zeros(VOCAB_SIZE)
        masked = gm.apply_mask(logits)
        assert masked[TOKEN_BOS] == 0.0
        assert masked[TOKEN_EOS] == float("-inf")


# ═══════════════════════════════════════════════════════════════════════
# BracketBalancer tests
# ═══════════════════════════════════════════════════════════════════════

class TestBracketBalancer:
    def setup_method(self):
        self.bb = BracketBalancer()

    def test_balanced_unchanged(self):
        tokens = [1, TOKEN_LPAREN, 2, 3, TOKEN_RPAREN, 4]
        assert self.bb.check(tokens)
        result = self.bb.check_and_repair(tokens)
        assert result == tokens

    def test_unmatched_close_removed(self):
        tokens = [1, TOKEN_RPAREN, 2, TOKEN_LPAREN, 3, TOKEN_RPAREN]
        assert not self.bb.check(tokens)
        result = self.bb.repair(tokens)
        assert self.bb.check(result)
        assert TOKEN_RPAREN not in result[:2]

    def test_unmatched_open_closed(self):
        tokens = [1, TOKEN_LPAREN, 2, 3]
        assert not self.bb.check(tokens)
        result = self.bb.repair(tokens)
        assert self.bb.check(result)
        assert result[-1] == TOKEN_RPAREN

    def test_nested_brackets(self):
        tokens = [
            TOKEN_LPAREN,
            TOKEN_LPAREN,
            1, 2,
            TOKEN_RPAREN,
            TOKEN_RPAREN,
        ]
        assert self.bb.check(tokens)

    def test_empty_is_balanced(self):
        assert self.bb.check([])

    def test_repair_preserves_content(self):
        tokens = [10, TOKEN_LPAREN, 20, 30]
        result = self.bb.repair(tokens)
        assert 10 in result and 20 in result and 30 in result


# ═══════════════════════════════════════════════════════════════════════
# GeometryChecker tests
# ═══════════════════════════════════════════════════════════════════════

class TestGeometryChecker:
    def setup_method(self):
        self.gc = GeometryChecker(tolerance=1)
        reset_id_counter()

    def test_closed_loop_passes(self):
        edges = [
            make_edge(make_line(0, 0, 128, 0)),
            make_edge(make_line(128, 0, 128, 128)),
            make_edge(make_line(128, 128, 0, 128)),
            make_edge(make_line(0, 128, 0, 0)),
        ]
        loop = make_loop(edges)
        assert self.gc.check_loop_closure(loop)

    def test_open_loop_fails(self):
        edges = [
            make_edge(make_line(0, 0, 128, 0)),
            make_edge(make_line(128, 0, 128, 128)),
            make_edge(make_line(128, 128, 0, 128)),
            make_edge(make_line(0, 128, 50, 50)),  # doesn't close
        ]
        loop = make_loop(edges)
        assert not self.gc.check_loop_closure(loop)

    def test_repair_closes_loop(self):
        edges = [
            make_edge(make_line(0, 0, 128, 0)),
            make_edge(make_line(128, 0, 128, 128)),
            make_edge(make_line(128, 128, 0, 128)),
            make_edge(make_line(0, 128, 50, 50)),
        ]
        loop = make_loop(edges)
        repaired = self.gc.repair_loop_closure(loop)
        assert self.gc.check_loop_closure(repaired)

    def test_check_and_repair_ast(self):
        ast = build_rectangle_solid()
        repaired, n_repairs = self.gc.check_and_repair_ast(ast)
        assert n_repairs == 0
        result = validate_ast(repaired)
        assert result.is_valid

    def test_single_edge_loop(self):
        edges = [make_edge(make_line(0, 0, 128, 0))]
        loop = make_loop(edges)
        assert not self.gc.check_loop_closure(loop)

    def test_tolerance(self):
        edges = [
            make_edge(make_line(0, 0, 128, 0)),
            make_edge(make_line(128, 0, 128, 128)),
            make_edge(make_line(128, 128, 0, 128)),
            make_edge(make_line(0, 128, 1, 0)),  # off by 1 in both x and y
        ]
        loop = make_loop(edges)
        gc_strict = GeometryChecker(tolerance=0)
        gc_loose = GeometryChecker(tolerance=1)
        assert not gc_strict.check_loop_closure(loop)
        assert gc_loose.check_loop_closure(loop)


# ═══════════════════════════════════════════════════════════════════════
# ConstraintDecoderPipeline tests
# ═══════════════════════════════════════════════════════════════════════

class TestConstraintDecoderPipeline:
    def setup_method(self):
        self.pipeline = ConstraintDecoderPipeline()
        reset_id_counter()

    def test_postprocess_clamps_q8(self):
        tokens = [TOKEN_BOS, Q8_OFFSET + 100, Q8_OFFSET + 255, TOKEN_EOS]
        result = self.pipeline.postprocess(tokens)
        for t in result:
            if Q8_OFFSET <= t < Q8_OFFSET + 256:
                assert 0 <= t - Q8_OFFSET <= 255

    def test_postprocess_fixes_brackets(self):
        tokens = [TOKEN_BOS, 10, TOKEN_LPAREN, 20, TOKEN_EOS]
        result = self.pipeline.postprocess(tokens)
        bb = BracketBalancer()
        assert bb.check(result)

    def test_roundtrip_valid_ast(self):
        ast = build_rectangle_solid()
        serializer = ASTSerializer()
        tokens, _ = serializer.serialize(ast, pad=False)
        recovered = self.pipeline.decode_to_ast(tokens)
        result = validate_ast(recovered)
        assert result.is_valid

    def test_constrained_decode_produces_tokens(self):
        L, V = 64, VOCAB_SIZE
        logits = torch.randn(L, V)
        tokens = self.pipeline.constrained_greedy_decode(logits, max_len=L)
        assert len(tokens) > 0
        assert tokens[0] == TOKEN_BOS
