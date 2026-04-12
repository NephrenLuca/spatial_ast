"""
AST -> IR emitter.

Walks the AST tree (DFS) and produces a flat list of ``IRInstruction``
objects.  The emitter handles dequantisation of Q8 parameters back to
continuous floats.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from core.types import NodeType
from core.ast_node import ASTNode
from compiler.quantize import dequantize
from compiler.ir import IRInstruction, EXTRUDE_OP_MAP


class IREmitter:
    """Convert an AST tree into a list of ``IRInstruction``."""

    def emit(self, root: ASTNode) -> List[IRInstruction]:
        instructions: List[IRInstruction] = []
        self._visit(root, instructions)
        return instructions

    # ── dispatcher ──────────────────────────────────────────────────

    def _visit(self, node: ASTNode, out: List[IRInstruction]) -> None:
        handler = self._DISPATCH.get(node.node_type)
        if handler is not None:
            handler(self, node, out)

    # ── per-node emitters ───────────────────────────────────────────

    def _emit_program(self, node: ASTNode, out: List[IRInstruction]) -> None:
        for child in node.children:
            self._visit(child, out)

    def _emit_solid(self, node: ASTNode, out: List[IRInstruction]) -> None:
        sketch = node.children[0]
        self._visit(sketch, out)
        for op in node.children[1:]:
            self._visit(op, out)

    def _emit_bool(self, node: ASTNode, out: List[IRInstruction]) -> None:
        op_type = node.params.get("op_type", "union")
        from compiler.ir import BOOLEAN_OP_MAP
        op_code = BOOLEAN_OP_MAP.get(op_type, 0.0)
        out.append(IRInstruction("boolean", [op_code]))
        for child in node.children:
            self._visit(child, out)

    def _emit_sketch(self, node: ASTNode, out: List[IRInstruction]) -> None:
        out.append(IRInstruction("sketch_start", []))
        for face in node.children:
            self._visit(face, out)
        out.append(IRInstruction("sketch_end", []))

    def _emit_face(self, node: ASTNode, out: List[IRInstruction]) -> None:
        for loop in node.children:
            self._visit(loop, out)

    def _emit_loop(self, node: ASTNode, out: List[IRInstruction]) -> None:
        out.append(IRInstruction("loop_start", []))
        for edge in node.children:
            self._visit(edge, out)
        out.append(IRInstruction("loop_end", []))

    def _emit_edge(self, node: ASTNode, out: List[IRInstruction]) -> None:
        for curve in node.children:
            self._visit(curve, out)

    def _emit_line(self, node: ASTNode, out: List[IRInstruction]) -> None:
        start, end = node.children
        sx, sy = self._deq_coord(start)
        ex, ey = self._deq_coord(end)
        out.append(IRInstruction("line", [sx, sy, ex, ey]))

    def _emit_arc(self, node: ASTNode, out: List[IRInstruction]) -> None:
        start, mid, end = node.children
        sx, sy = self._deq_coord(start)
        mx, my = self._deq_coord(mid)
        ex, ey = self._deq_coord(end)
        out.append(IRInstruction("arc", [sx, sy, mx, my, ex, ey]))

    def _emit_circle(self, node: ASTNode, out: List[IRInstruction]) -> None:
        center = node.children[0]
        radius_node = node.children[1]
        cx, cy = self._deq_coord(center)
        r = dequantize(radius_node.params.get("value", 0))
        out.append(IRInstruction("circle", [cx, cy, r]))

    def _emit_extrude(self, node: ASTNode, out: List[IRInstruction]) -> None:
        d_fwd = dequantize(node.params.get("distance_fwd", 0))
        d_bwd = dequantize(node.params.get("distance_bwd", 0))
        op_type = node.params.get("op_type", "new")
        op_code = EXTRUDE_OP_MAP.get(op_type, 0.0)
        out.append(IRInstruction("extrude", [d_fwd, d_bwd, op_code]))

    def _emit_revolve(self, node: ASTNode, out: List[IRInstruction]) -> None:
        angle = dequantize(node.params.get("angle", 0))
        op_type = node.params.get("op_type", "new")
        op_code = EXTRUDE_OP_MAP.get(op_type, 0.0)
        out.append(IRInstruction("revolve", [angle, op_code]))

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _deq_coord(crd_node: ASTNode) -> tuple:
        x = dequantize(crd_node.params.get("x", 0))
        y = dequantize(crd_node.params.get("y", 0))
        return x, y

    # ── dispatch table ──────────────────────────────────────────────

    _DISPATCH: Dict[NodeType, Callable] = {
        NodeType.PROG: _emit_program,
        NodeType.SOL:  _emit_solid,
        NodeType.BOOL: _emit_bool,
        NodeType.SKT:  _emit_sketch,
        NodeType.FACE: _emit_face,
        NodeType.LOOP: _emit_loop,
        NodeType.EDGE: _emit_edge,
        NodeType.LN:   _emit_line,
        NodeType.ARC:  _emit_arc,
        NodeType.CIR:  _emit_circle,
        NodeType.EXT:  _emit_extrude,
        NodeType.REV:  _emit_revolve,
    }
