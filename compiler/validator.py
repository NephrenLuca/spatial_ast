"""
Pre-compile semantic validation for AST trees.

Runs the grammar-level ``validate_ast`` plus additional compiler-specific
checks (e.g. EXT/REV parameter ranges, loop closure plausibility) to
ensure the AST is safe to lower into IR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from core.types import NodeType
from core.ast_node import ASTNode
from core.grammar import validate_ast, ValidationResult
from compiler.quantize import dequantize


@dataclass
class CompileValidationResult:
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.is_valid = False
        self.errors.append(msg)


class CompileValidator:
    """
    Extended validator that combines grammar checks with compiler-specific
    constraints.
    """

    def validate(self, root: ASTNode) -> CompileValidationResult:
        result = CompileValidationResult()

        # 1. Grammar-level validation
        grammar_result = validate_ast(root)
        if not grammar_result.is_valid:
            result.is_valid = False
            result.errors.extend(grammar_result.errors)

        # 2. Compiler-specific checks
        for node in root.dfs():
            self._check_extrude(node, result)
            self._check_revolve(node, result)
            self._check_loop_non_empty(node, result)
            self._check_sketch_has_geometry(node, result)

        return result

    @staticmethod
    def _check_extrude(node: ASTNode, result: CompileValidationResult) -> None:
        if node.node_type != NodeType.EXT:
            return
        fwd = node.params.get("distance_fwd", 0)
        bwd = node.params.get("distance_bwd", 0)
        fwd_f = dequantize(fwd)
        bwd_f = dequantize(bwd)
        if abs(fwd_f) < 0.01 and abs(bwd_f) < 0.01:
            result.add_error(
                f"EXT (id={node.node_id}): both distances are zero — "
                f"degenerate extrude"
            )

    @staticmethod
    def _check_revolve(node: ASTNode, result: CompileValidationResult) -> None:
        if node.node_type != NodeType.REV:
            return
        angle = node.params.get("angle", 0)
        angle_f = dequantize(angle)
        if abs(angle_f) < 1e-9:
            result.add_error(
                f"REV (id={node.node_id}): angle is zero — degenerate revolve"
            )

    @staticmethod
    def _check_loop_non_empty(node: ASTNode, result: CompileValidationResult) -> None:
        if node.node_type != NodeType.LOOP:
            return
        edges = [c for c in node.children if c.node_type == NodeType.EDGE]
        if len(edges) == 0:
            result.add_error(
                f"LOOP (id={node.node_id}): contains no EDGE children"
            )

    @staticmethod
    def _check_sketch_has_geometry(
        node: ASTNode, result: CompileValidationResult,
    ) -> None:
        if node.node_type != NodeType.SKT:
            return
        curve_count = sum(
            1 for n in node.dfs()
            if n.node_type in (NodeType.LN, NodeType.ARC, NodeType.CIR)
        )
        if curve_count == 0:
            result.add_error(
                f"SKT (id={node.node_id}): sketch contains no curves"
            )
