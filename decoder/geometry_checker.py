"""
Geometry consistency checker for decoded ASTs.

Validates and optionally repairs Loop closure: edges in a Loop must
form a connected chain where each edge's start coincides with the
previous edge's end, and the last edge closes back to the first.

Follows architecture.md Section 7.4.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Tuple

from core.types import NodeType
from core.ast_node import ASTNode


class GeometryChecker:
    """Check and repair geometric consistency of decoded AST structures."""

    def __init__(self, tolerance: int = 1) -> None:
        self.tolerance = tolerance

    def check_loop_closure(self, loop: ASTNode) -> bool:
        """
        Return True if the edges in *loop* form a closed ring.

        Each Edge node contains exactly one curve child (LN, ARC, CIR),
        whose CRD children define start and end coordinates.
        """
        edges = loop.children
        if not edges:
            return False

        first_start = self._get_start_coord(edges[0])
        if first_start is None:
            return False

        prev_end = first_start
        for edge in edges:
            start = self._get_start_coord(edge)
            if start is None or not self._coords_close(prev_end, start):
                return False
            end = self._get_end_coord(edge)
            if end is None:
                return False
            prev_end = end

        return self._coords_close(prev_end, first_start)

    def repair_loop_closure(self, loop: ASTNode) -> ASTNode:
        """
        Repair an open loop by adjusting the last edge's end coordinate
        to match the first edge's start coordinate.
        """
        edges = list(loop.children)
        if len(edges) < 2:
            return loop

        first_start = self._get_start_coord(edges[0])
        if first_start is None:
            return loop

        last_edge = edges[-1]
        repaired_edge = self._set_end_coord(last_edge, first_start)
        if repaired_edge is None:
            return loop

        edges[-1] = repaired_edge
        return replace(loop, children=tuple(edges))

    def check_and_repair_ast(self, root: ASTNode) -> Tuple[ASTNode, int]:
        """
        Walk the full AST and repair every Loop that isn't closed.

        Returns the (possibly modified) AST and the number of repairs made.
        """
        repairs = [0]

        def _fix(node: ASTNode) -> ASTNode:
            if node.node_type == NodeType.LOOP:
                if not self.check_loop_closure(node):
                    repairs[0] += 1
                    return self.repair_loop_closure(node)
            return node

        fixed = root.map(_fix)
        return fixed, repairs[0]

    # ── Coordinate helpers ─────────────────────────────────────────

    def _get_curve(self, edge: ASTNode) -> Optional[ASTNode]:
        """Get the curve child (LN/ARC/CIR) of an Edge node."""
        if edge.node_type != NodeType.EDGE:
            return None
        for child in edge.children:
            if child.node_type in (NodeType.LN, NodeType.ARC, NodeType.CIR):
                return child
        return None

    def _get_crds(self, curve: ASTNode) -> List[ASTNode]:
        """Get all CRD children of a curve node."""
        return [c for c in curve.children if c.node_type == NodeType.CRD]

    def _get_start_coord(self, edge: ASTNode) -> Optional[Tuple[int, int]]:
        """Return (x, y) of the first CRD in the edge's curve."""
        curve = self._get_curve(edge)
        if curve is None:
            return None

        if curve.node_type == NodeType.CIR:
            crds = self._get_crds(curve)
            if crds:
                return (crds[0].params.get("x", 0), crds[0].params.get("y", 0))
            return None

        crds = self._get_crds(curve)
        if crds:
            return (crds[0].params.get("x", 0), crds[0].params.get("y", 0))
        return None

    def _get_end_coord(self, edge: ASTNode) -> Optional[Tuple[int, int]]:
        """Return (x, y) of the last CRD in the edge's curve."""
        curve = self._get_curve(edge)
        if curve is None:
            return None

        if curve.node_type == NodeType.CIR:
            return self._get_start_coord(edge)

        crds = self._get_crds(curve)
        if crds:
            return (crds[-1].params.get("x", 0), crds[-1].params.get("y", 0))
        return None

    def _set_end_coord(
        self, edge: ASTNode, target: Tuple[int, int]
    ) -> Optional[ASTNode]:
        """Set the last CRD's (x, y) in *edge*'s curve to *target*."""
        curve = self._get_curve(edge)
        if curve is None:
            return None

        if curve.node_type == NodeType.CIR:
            return edge

        crds = self._get_crds(curve)
        if not crds:
            return None

        last_crd = crds[-1]
        new_crd = replace(last_crd, params={"x": target[0], "y": target[1]})

        new_children = list(curve.children)
        for i in range(len(new_children) - 1, -1, -1):
            if new_children[i].node_type == NodeType.CRD:
                new_children[i] = new_crd
                break
        new_curve = replace(curve, children=tuple(new_children))

        new_edge_children = list(edge.children)
        for i, c in enumerate(new_edge_children):
            if c.node_type in (NodeType.LN, NodeType.ARC, NodeType.CIR):
                new_edge_children[i] = new_curve
                break
        return replace(edge, children=tuple(new_edge_children))

    def _coords_close(
        self, a: Tuple[int, int], b: Tuple[int, int]
    ) -> bool:
        return abs(a[0] - b[0]) <= self.tolerance and abs(a[1] - b[1]) <= self.tolerance
