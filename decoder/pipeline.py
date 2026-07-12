"""
Constraint decoder pipeline: chains grammar mask, bracket balancer,
and geometry checker into a single post-processing pass.

Follows architecture.md Section 7.1.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from core.tokenizer import (
    TOKEN_BOS,
    TOKEN_EOS,
    TOKEN_PAD,
    VOCAB_SIZE,
    Q8_OFFSET,
    is_q8_token,
    get_node_type_from_token,
)
from core.serializer import ASTSerializer
from core.ast_node import ASTNode
from core.grammar import validate_ast, ValidationResult
from decoder.grammar_mask import GrammarMask
from decoder.bracket_balancer import BracketBalancer
from decoder.geometry_checker import GeometryChecker


class ConstraintDecoderPipeline:
    """
    End-to-end constrained decoding pipeline.

    Stages:
      1. **Grammar mask** — during autoregressive generation, masks
         logits at each step so only structurally legal tokens remain.
      2. **Bracket balancer** — post-hoc repair of bracket mismatches.
      3. **Value range clipper** — clamp Q8 tokens to [Q8_OFFSET, Q8_OFFSET+255].
      4. **Geometry checker** — repair open loops.
    """

    def __init__(self, geometry_tolerance: int = 1) -> None:
        self.bracket_balancer = BracketBalancer()
        self.geometry_checker = GeometryChecker(tolerance=geometry_tolerance)
        self.serializer = ASTSerializer()

    def constrained_greedy_decode(
        self,
        logits: Tensor,
        max_len: Optional[int] = None,
    ) -> List[int]:
        """
        Greedy-decode a full sequence from ``[L, V]`` logits with
        grammar constraints applied at each step.

        Parameters
        ----------
        logits : [L, V]
            Pre-computed logits for each position (teacher-forcing style).
        max_len : int, optional
            Maximum output length (defaults to L).

        Returns
        -------
        tokens : list[int]
            Decoded token IDs (BOS … EOS).
        """
        if max_len is None:
            max_len = logits.shape[0]

        grammar = GrammarMask()
        tokens: List[int] = []

        for pos in range(max_len):
            masked_logits = grammar.apply_mask(logits[pos])
            token_id = int(masked_logits.argmax(-1).item())
            tokens.append(token_id)
            grammar.feed(token_id)

            if token_id == TOKEN_EOS:
                break

        return tokens

    def postprocess(self, tokens: List[int]) -> List[int]:
        """
        Apply all post-hoc repairs to a decoded token sequence.

        1. Bracket balancing
        2. Q8 value range clamping
        """
        tokens = self.bracket_balancer.check_and_repair(tokens)
        tokens = self._clamp_q8(tokens)
        return tokens

    def decode_to_ast(self, tokens: List[int]) -> ASTNode:
        """
        Full pipeline: postprocess tokens → deserialize → geometry repair.
        """
        tokens = self.postprocess(tokens)
        ast = self.serializer.deserialize(tokens)
        ast, _ = self.geometry_checker.check_and_repair_ast(ast)
        return ast

    def validate(self, ast: ASTNode) -> ValidationResult:
        """Run grammar validation on the decoded AST."""
        return validate_ast(ast)

    def full_pipeline(
        self, logits: Tensor, max_len: Optional[int] = None
    ) -> tuple[List[int], ASTNode, ValidationResult]:
        """
        End-to-end: logits → constrained decode → postprocess → AST → validate.

        Returns (tokens, ast, validation_result).
        """
        tokens = self.constrained_greedy_decode(logits, max_len)
        tokens = self.postprocess(tokens)
        ast = self.serializer.deserialize(tokens)
        ast, _ = self.geometry_checker.check_and_repair_ast(ast)
        result = validate_ast(ast)
        return tokens, ast, result

    @staticmethod
    def _clamp_q8(tokens: List[int]) -> List[int]:
        """Ensure Q8 tokens are within the valid range."""
        out: List[int] = []
        for t in tokens:
            if is_q8_token(t):
                val = t - Q8_OFFSET
                val = max(0, min(255, val))
                out.append(Q8_OFFSET + val)
            else:
                out.append(t)
        return out
